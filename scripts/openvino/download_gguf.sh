#!/bin/bash

# Llama-3.2-1B-Instruct
wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf \
     -O models/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf

# Llama-3.1-8B-Instruct
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf \
     -O models/gguf/Llama-3.1-8B-Instruct-Q4_0.gguf

# Phi-3-mini-4k-instruct
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf \
     -O models/gguf/Phi-3-mini-4k-instruct-Q4_0.gguf

# Qwen2.5-1.5B-Instruct
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf \
     -O models/gguf/Qwen2.5-1.5B-Instruct-Q4_0.gguf

# Qwen3-8B
wget https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_0.gguf \
     -O models/gguf/Qwen3-8B-Q4_0.gguf

# MiniCPM-1B
wget https://huggingface.co/openbmb/MiniCPM-S-1B-sft-gguf/resolve/main/MiniCPM-S-1B-sft-bf16.gguf \
     -O models/gguf/MiniCPM-1B-Q4_0.gguf

# Hunyuan-7B-Instruct
wget https://huggingface.co/bartowski/tencent_Hunyuan-7B-Instruct-GGUF/resolve/main/tencent_Hunyuan-7B-Instruct-Q4_0.gguf \
     -O models/gguf/Hunyuan-7B-Instruct-Q4_0.gguf

# Mistral-7B-Instruct-v0.3
wget https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_0.gguf \
     -O models/gguf/Mistral-7B-Instruct-v0.3-Q4_0.gguf

# DeepSeek-R1-Distill-Llama-8B
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf \
     -O models/gguf/DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf