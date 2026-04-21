#!/bin/bash
MODELS=(
  "models/Qwen_Qwen2.5-1.5B-Instruct"
  "models/Qwen_Qwen3-8B"
  "models/meta-llama_Llama-3.1-8B-Instruct"
  "models/microsoft_Phi-3-mini-4k-instruct"
  "models/microsoft_Phi-3.5-mini-instruct"
  "models/LiquidAI_LFM2-2.6B"
  "models/tencent_Hunyuan-7B-Instruct"
  "models/openbmb_MiniCPM-1B-sft-bf16"
  "models/deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
  "models/mistralai_Mistral-7B-Instruct-v0.3"
)

#"models/meta-llama_Llama-3.2-1B-Instruct"

for MODEL_ID in "${MODELS[@]}"; do
  MODEL_NAME=$(basename $MODEL_ID)

  # INT4 SYM Channel-wise (group_size=-1 = no grouping, per-channel)
  optimum-cli export openvino \
    --model $MODEL_ID \
    --task text-generation-with-past \
    --library transformers \
    --weight-format int4 \
    --sym \
    --group-size -1 \
    models/ir/${MODEL_NAME}/INT4_SYM_CW

  # INT4 SYM Group-size 32
  optimum-cli export openvino \
    --model $MODEL_ID \
    --task text-generation-with-past \
    --library transformers \
    --weight-format int4 \
    --sym \
    --group-size 32 \
    models/ir/${MODEL_NAME}/INT4_SYM_GS32

  # INT4 SYM Group-size default
  optimum-cli export openvino \
    --model $MODEL_ID \
    --task text-generation-with-past \
    --library transformers \
    --weight-format int4 \
    models/ir/${MODEL_NAME}/INT4_DEFAULT
done