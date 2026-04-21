#!/bin/bash
# ============================================================
# llm_bench: All backends for one model
# Usage: 
#MODEL_IR=models/ir/Phi-3-mini-4k-instruct
#        
#MODEL_GGUF=models/gguf/Phi-3-mini-4k-instruct-Q4_0.gguf
#        
#MODEL_TAG=Phi-3-mini-4k-instruct  
#bash run_llm_bench_all.sh
# ============================================================

set -o pipefail  # Ensure pipe failures are caught

MODEL_TAG=${MODEL_TAG:-model}
LOGDIR=logs/${MODEL_TAG}
mkdir -p $LOGDIR

#cd openvino.genai/tools/llm_bench
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
# A. OV GenAI IR — INT4_SYM_CW
# -----------------------------------------------------------
IR_CW=${MODEL_IR}/INT4_SYM_CW
for IC in 32 128; do
  run_bench "ov_genai_ir_cpu_cw"  $IR_CW  CPU  $IC  INT4_SYM_CW
  run_bench "ov_genai_ir_gpu_cw"  $IR_CW  GPU  $IC  INT4_SYM_CW
  run_bench "ov_genai_ir_npu_cw"  $IR_CW  NPU  $IC  INT4_SYM_CW
done

# -----------------------------------------------------------
# B. OV GenAI IR — INT4_SYM_GS32
# -----------------------------------------------------------
IR_GS32=${MODEL_IR}/INT4_SYM_GS32
for IC in 32 128; do
  run_bench "ov_genai_ir_cpu_gs32"  $IR_GS32  CPU  $IC  INT4_SYM_GS32
  run_bench "ov_genai_ir_gpu_gs32"  $IR_GS32  GPU  $IC  INT4_SYM_GS32
  run_bench "ov_genai_ir_npu_gs32"  $IR_GS32  NPU  $IC  INT4_SYM_GS32
done

# -----------------------------------------------------------
# C. OV GenAI IR — INT4_DEFAULT
# -----------------------------------------------------------
IR_DEFAULT=${MODEL_IR}/INT4_DEFAULT
for IC in 32 128; do
  run_bench "ov_genai_ir_cpu_default"  $IR_DEFAULT  CPU  $IC  INT4_DEFAULT
  run_bench "ov_genai_ir_gpu_default"  $IR_DEFAULT  GPU  $IC  INT4_DEFAULT
  run_bench "ov_genai_ir_npu_default"  $IR_DEFAULT  NPU  $IC  INT4_DEFAULT
done

# -----------------------------------------------------------
# D. OV GenAI GGUF Reader
# -----------------------------------------------------------
for IC in 32 128; do
  run_bench "ov_genai_gguf_cpu"  $MODEL_GGUF  CPU  $IC  Q4_0
  run_bench "ov_genai_gguf_gpu"  $MODEL_GGUF  GPU  $IC  Q4_0
  run_bench "ov_genai_gguf_npu"  $MODEL_GGUF  NPU  $IC  Q4_0
done

echo ">>> All llm_bench runs complete. Logs in $LOGDIR/"