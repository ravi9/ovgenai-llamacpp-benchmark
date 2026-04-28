#!/bin/bash
# ============================================================
# llm_bench: All backends for all models in models/ir and models/gguf
# Usage: ./run_llm_bench_all.sh
# ============================================================

set -o pipefail  # Ensure pipe failures are caught

source env/bin/activate

PROMPT_FILE=prompts/prompt_32tok.jsonl
COMMON_ARGS="-pf $PROMPT_FILE -n 3 -bs 1 --disable_prompt_permutation -s 42 --task text_gen"

# -----------------------------------------------------------
# Helper function
# -----------------------------------------------------------
run_bench() {
  local TAG=$1; local MODEL=$2; local DEVICE=$3; local IC=$4; local QUANT=$5
  local LOG="$LOGDIR/${TAG}_ic${IC}.log"
  local CSV="$LOGDIR/${TAG}_ic${IC}.csv"
  echo ">>> Running: $TAG (ic=$IC)"
  python openvino.genai/tools/llm_bench/benchmark.py -m $MODEL -d $DEVICE -ic $IC $COMMON_ARGS \
    -r $CSV 2>&1 | tee $LOG \
    || echo "FAILED: $TAG ic$IC" >> $LOGDIR/FAILED.log
}

# -----------------------------------------------------------
# Process each model in models/ir
# -----------------------------------------------------------
for MODEL_IR_PATH in models/ir/*; do
  # Skip if not a directory
  [[ ! -d "$MODEL_IR_PATH" ]] && continue
  
  # Extract model tag (directory name)
  MODEL_TAG=$(basename "$MODEL_IR_PATH")
  
  # Find corresponding GGUF file
  # Try common patterns for GGUF filenames
  MODEL_GGUF=""
  for pattern in "${MODEL_TAG}-Q4_0.gguf" "${MODEL_TAG}-q4.gguf" "Meta-${MODEL_TAG}-Q4_0.gguf" "${MODEL_TAG}"-*.gguf; do
    if [ -f "models/gguf/$pattern" ]; then
      MODEL_GGUF="models/gguf/$pattern"
      break
    fi
  done
  
  # If no exact match, try to find any GGUF file with similar name
  if [ -z "$MODEL_GGUF" ]; then
    # Extract core model name (first part before any version number)
    CORE_NAME=$(echo "$MODEL_TAG" | sed -E 's/(-[0-9]+(\.[0-9]+)*[A-Z]?(-Instruct)?)$//')
    for gguf_file in models/gguf/*.gguf; do
      if [[ $(basename "$gguf_file") == *"$CORE_NAME"* ]]; then
        MODEL_GGUF="$gguf_file"
        break
      fi
    done
  fi
  
  # Create log directory for this model
  LOGDIR="logs/${MODEL_TAG}"
  mkdir -p $LOGDIR
  
  echo ""
  echo "========================================================"
  echo "Processing model: $MODEL_TAG"
  echo "IR path: $MODEL_IR_PATH"
  if [ -n "$MODEL_GGUF" ]; then
    echo "GGUF path: $MODEL_GGUF"
  else
    echo "GGUF path: NOT FOUND (will skip GGUF benchmarks)"
  fi
  echo "========================================================"
  echo ""

  # -----------------------------------------------------------
  # A. OV GenAI IR — INT4_SYM_CW
  # -----------------------------------------------------------
  IR_CW=${MODEL_IR_PATH}/INT4_SYM_CW
  if [ -d "$IR_CW" ]; then
    for IC in 32 128; do
      run_bench "ov_genai_ir_cpu_cw"  $IR_CW  CPU  $IC  INT4_SYM_CW
      run_bench "ov_genai_ir_gpu_cw"  $IR_CW  GPU  $IC  INT4_SYM_CW
      run_bench "ov_genai_ir_npu_cw"  $IR_CW  NPU  $IC  INT4_SYM_CW
    done
  fi

  # -----------------------------------------------------------
  # B. OV GenAI IR — INT4_SYM_GS32
  # -----------------------------------------------------------
  IR_GS32=${MODEL_IR_PATH}/INT4_SYM_GS32
  if [ -d "$IR_GS32" ]; then
    for IC in 32 128; do
      run_bench "ov_genai_ir_cpu_gs32"  $IR_GS32  CPU  $IC  INT4_SYM_GS32
      run_bench "ov_genai_ir_gpu_gs32"  $IR_GS32  GPU  $IC  INT4_SYM_GS32
      run_bench "ov_genai_ir_npu_gs32"  $IR_GS32  NPU  $IC  INT4_SYM_GS32
    done
  fi

  # -----------------------------------------------------------
  # C. OV GenAI IR — INT4_DEFAULT
  # -----------------------------------------------------------
  IR_DEFAULT=${MODEL_IR_PATH}/INT4_DEFAULT
  if [ -d "$IR_DEFAULT" ]; then
    for IC in 32 128; do
      run_bench "ov_genai_ir_cpu_default"  $IR_DEFAULT  CPU  $IC  INT4_DEFAULT
      run_bench "ov_genai_ir_gpu_default"  $IR_DEFAULT  GPU  $IC  INT4_DEFAULT
      run_bench "ov_genai_ir_npu_default"  $IR_DEFAULT  NPU  $IC  INT4_DEFAULT
    done
  fi

  # -----------------------------------------------------------
  # D. OV GenAI GGUF Reader
  # -----------------------------------------------------------
  if [ -n "$MODEL_GGUF" ] && [ -f "$MODEL_GGUF" ]; then
    for IC in 32 128; do
      run_bench "ov_genai_gguf_cpu"  $MODEL_GGUF  CPU  $IC  Q4_0
      run_bench "ov_genai_gguf_gpu"  $MODEL_GGUF  GPU  $IC  Q4_0
      run_bench "ov_genai_gguf_npu"  $MODEL_GGUF  NPU  $IC  Q4_0
    done
  fi

  echo ">>> Completed benchmarks for $MODEL_TAG. Logs in $LOGDIR/"
  
done

echo ""
echo "========================================================"
echo "All llm_bench runs complete!"
echo "========================================================"