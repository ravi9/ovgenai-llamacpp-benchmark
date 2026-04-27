#!/bin/bash

# Check if file argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <models_file>"
  exit 1
fi

MODELS_FILE=$1

# Read models from file
while IFS= read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  
  # Remove quotes and trim whitespace
  MODEL=$(echo "$line" | tr -d '"' | xargs)
  
  # Extract model name (part after "/")
  MODEL_NAME="${MODEL#*/}"
  
  # Construct model path
  MODEL_ID="models/${MODEL_NAME}"

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
done < "$MODELS_FILE"