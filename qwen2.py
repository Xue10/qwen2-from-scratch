import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Config:
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: Optional[int] = 2
    vocab_size: int = 151936
    intermediate_size: int = 4864
    rms_norm_eps: float = 1e-06

    rope_theta: float = 1000000.0

    max_batch_size: int = 4 # training batch size unknown
    max_position_embeddings: int = 32768

    device:str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device:str, theta: float = 10000.0): # in config theta=1M, but transformers lib uses 10000.0

    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator/head_dim)).to(device)

    m = torch.arange(seq_len, device=device)
    # Shape: (seq_len) outer (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, h, head_dim / 2) -> (B, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x.unsqueeze(3)
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim)
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    
class SelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()    
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        # QKV bias: bias = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.cache_k = torch.zeros((config.max_batch_size, config.max_position_embeddings, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((config.max_batch_size, config.max_position_embeddings, self.n_kv_heads, self.head_dim))
    def forward(self, x, start_pos, freqs_complex):

        batch_size, seq_len, _ = x.shape
        
        xq = self.q_proj(x)
        
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:batch_size, start_pos: start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos+seq_len] = xv
        # (B, start_pos+seq_len, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]
        # (B, start_pos+seq_len, n_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep).to(x.device)
        values = repeat_kv(values, self.n_rep).to(x.device)
        # (B, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # (B, n_heads, start_pos+seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # (B, n_heads, seq_len, start_pos+seq_len)
        # print(f'xq {xq.device}| keys {keys.device}')
        scores = xq @ keys.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
        scores = F.softmax(scores.float(), dim=-1)
        # (B, h_heads, seq_len, head_dim)
        out = scores @ values
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads*self.head_dim)
        return self.o_proj(out)

class MLP(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        # Normalization before self attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Normalization before mlp
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, start_pos: int, freqs_complex):
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_complex)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out
    
class Model(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers):
            self.layers.append(DecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.freqs_complex = precompute_theta_pos_frequencies(self.config.hidden_size // self.config.num_attention_heads, self.config.max_position_embeddings*2, device=self.config.device)
       
    def forward(self, tokens, start_pos: int):
        batch_size, seq_len = tokens.shape 
        assert seq_len == 1, "Using kv cache, only one token at a time can be processed"

        h = self.embed_tokens(tokens)

        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        
        return h

class Qwen2(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, x):
        x = self.model(x, 0)
        logits = self.lm_head(x)
        return logits
    
    @classmethod
    def from_pretrained(cls, device):
        t0 = time.time()
     
        model = Qwen2(Config(device=device))
        
        sd = model.state_dict()

        # load from local
        qwen = AutoModelForCausalLM.from_pretrained('./Qwen2-0.5B')

        # load from remote
        # qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B')
       
        sd_trained = qwen.state_dict()
        print(sd_trained.keys() == sd.keys())
        for k in sd_trained.keys():
            # print(k)
            assert sd[k].shape == sd_trained[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_trained[k])
                # print(f'{k} copied')
        t1 = time.time()
        dt = t1 - t0
        print(f"model loaded, using {dt:3f}s")
        return model
    
model = Qwen2.from_pretrained(device='cuda')
model.to('cuda')
print('cuda loaded')

#--------------------------------------------
# test by input one token

#load from local
tokenizer = AutoTokenizer.from_pretrained('./Qwen2-0.5B')

# load from remote
# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')

tokens = tokenizer.encode("Nice")
logits = model(torch.tensor([tokens], device='cuda'))
print(logits)
next_token = torch.argmax(logits, dim=-1)
print(next_token)
output = tokenizer.decode([next_token.item()])
print(output)

