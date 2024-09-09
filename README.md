# Qwen2 from scratch
This repository is to build qwen2 0.5B model from scratch. 

The project is inspired by https://github.com/hkproj/pytorch-llama
## Model features
According to the qwen2 paper, the following features has been implemented:
- *Grouped Query Attention*
- *KV cache*
- *SwiGLU*
- *Rotary Positional Embeddings*
- *QKV bias*
- *RMSNorm*
- *Pre-normalization*

Currently the qwen2.py file is to load 0.5B model.
## Todo
- Inference code
- Training code
## References
[Qwen2 Model](https://huggingface.co/Qwen/Qwen2-0.5B) in huggingface

[Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

[modeling_qwen2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) in transformers library