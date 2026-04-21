name=run_llm_bench_all_models.sh
#!/bin/bash
declare -A MODELS=(
  ["Llama-3.2-1B-Instruct"]="Llama-3.2-1B-Instruct"
  ["Llama-3.1-8B-Instruct"]="Llama-3.1-8B-Instruct"
  ["Phi-3-mini-4k-instruct"]="Phi-3-mini-4k-instruct"
  ["Qwen2.5-1.5B-Instruct"]="Qwen2.5-1.5B-Instruct"
  ["DeepSeek-R1-Distill-Llama-8B"]="DeepSeek-R1-Distill-Llama-8B"
)

# ["Hunyuan-7B-Instruct"]="Hunyuan-7B-Instruct"
# ["Qwen3-8B"]="Qwen3-8B"
# ["MiniCPM-1B"]="MiniCPM-1B"
# ["Mistral-7B-Instruct-v0.3"]="Mistral-7B-Instruct-v0.3"

for MODEL_TAG in "${!MODELS[@]}"; do
  MODEL_IR=models/ir/$MODEL_TAG
  if [ "$MODEL_TAG" = "Llama-3.1-8B-Instruct" ]; then
    MODEL_GGUF=models/gguf/${MODEL_TAG}-Q4_0_4_4.gguf
    else  MODEL_GGUF=models/gguf/${MODEL_TAG}-Q4_0.gguf
  fi
  export MODEL_IR MODEL_GGUF MODEL_TAG
  bash run_llm_bench_all.sh
  
done
