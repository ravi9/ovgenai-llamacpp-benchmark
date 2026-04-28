#!/bin/bash
# ============================================================
# llama-bench: All backends for all models in models/gguf/
# Usage: bash run_llama_bench_all.sh
# ============================================================

# Common bench args (no -fa here; added per-backend below)
BENCH_ARGS="-p 32 -n 32,128 -b 32 -r 3 -fitc 256 -o csv"

# Get all GGUF models
GGUF_DIR="models/gguf"
MODELS=($(ls -1 $GGUF_DIR/*.gguf 2>/dev/null))

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No GGUF models found in $GGUF_DIR"
    exit 1
fi

echo "Found ${#MODELS[@]} models to benchmark"
echo "========================================"

# Loop through all models
for MODEL_PATH in "${MODELS[@]}"; do
    # Extract model name without path and extension
    MODEL_FILENAME=$(basename "$MODEL_PATH")
    MODEL_TAG="${MODEL_FILENAME%.gguf}"
    
    # Remove quantization suffixes from model tag
    MODEL_TAG=$(echo "$MODEL_TAG" | sed -E 's/(-[Qq]4[_0-9]*)?$//' | sed -E 's/Q4_0_4_4$//' | sed -E 's/-bf16$//')
    
    echo ""
    echo "========================================"
    echo "Processing: $MODEL_TAG"
    echo "Model: $MODEL_PATH"
    echo "========================================"
    
    LOGDIR="logs/${MODEL_TAG}"
    mkdir -p "$LOGDIR"
    
    # -----------------------------------------------------------
    # 1. Default GGML CPU backend
    # -----------------------------------------------------------
    echo ">>> Running: llama_ggml_cpu"
    ./llama.cpp/build/Release/bin/llama-bench \
      -m "$MODEL_PATH" $BENCH_ARGS \
      2>&1 | tee "$LOGDIR/llama_ggml_cpu.log"
    
    # -----------------------------------------------------------
    # 2. Vulkan backend (GPU)
    # -----------------------------------------------------------
    echo ">>> Running: llama_vulkan_gpu"
    ./llama.cpp/build/ReleaseVulkan/bin/llama-bench \
      -m "$MODEL_PATH" $BENCH_ARGS \
      2>&1 | tee "$LOGDIR/llama_vulkan_gpu.log"
    
    # -----------------------------------------------------------
    # 3. OpenVINO backend — CPU
    # -----------------------------------------------------------
    echo ">>> Running: llama_ov_cpu"
    source /opt/intel/openvino_2026.1.0/setupvars.sh
    export GGML_OPENVINO_DEVICE=CPU
    export GGML_OPENVINO_STATEFUL_EXECUTION=1
    export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache
    
    ./llama.cpp/build/ReleaseOV/bin/llama-bench \
      -m "$MODEL_PATH" $BENCH_ARGS -fa 1 \
      2>&1 | tee "$LOGDIR/llama_ov_cpu.log"
    
    # -----------------------------------------------------------
    # 4. OpenVINO backend — GPU
    # -----------------------------------------------------------
    echo ">>> Running: llama_ov_gpu"
    export GGML_OPENVINO_DEVICE=GPU
    export GGML_OPENVINO_STATEFUL_EXECUTION=1   # Required; stateless has known issues on GPU
    export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache
    
    ./llama.cpp/build/ReleaseOV/bin/llama-bench \
      -m "$MODEL_PATH" $BENCH_ARGS -fa 1 \
      2>&1 | tee "$LOGDIR/llama_ov_gpu.log"
    
    # -----------------------------------------------------------
    # 5. OpenVINO backend — NPU
    # IMPORTANT: unset GGML_OPENVINO_STATEFUL_EXECUTION for NPU.
    # Setting it to 0 does NOT work — you must fully unset it.
    # GGML_OPENVINO_CACHE_DIR is also NOT supported on NPU.
    # -----------------------------------------------------------
    echo ">>> Running: llama_ov_npu"
    export GGML_OPENVINO_DEVICE=NPU
    unset GGML_OPENVINO_STATEFUL_EXECUTION
    unset GGML_OPENVINO_CACHE_DIR
    
    ./llama.cpp/build/ReleaseOV/bin/llama-bench \
      -m "$MODEL_PATH" $BENCH_ARGS -fa 1 \
      2>&1 | tee "$LOGDIR/llama_ov_npu.log"
    
    echo ">>> Completed benchmarking for $MODEL_TAG. Logs in $LOGDIR/"
done

echo ""
echo "========================================"
echo "All llama-bench runs complete!"
echo "Processed ${#MODELS[@]} models"
echo "Logs saved in logs/ directory"
echo "========================================"