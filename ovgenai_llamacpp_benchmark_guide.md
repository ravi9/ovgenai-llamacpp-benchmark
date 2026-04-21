# LLM Benchmarking Guide: OpenVINO GenAI `llm_bench` vs llama.cpp `llama-bench`

**Benchmark Configuration:** Input tokens: 32 | Output tokens: 32 & 128 | Warmup: 1 | Measured runs: 3 | Context: 256

---

## 📋 Table of Contents
1. [Benchmark Configuration Decisions](#1-benchmark-configuration-decisions)
2. [Parameter Mapping Table](#2-parameter-mapping-table)
3. [Setup & Build](#3-setup--build)
4. [Model Preparation](#4-model-preparation)
5. [Prompt File Setup for llm_bench](#5-prompt-file-setup-for-llm_bench)
6. [llama-bench Sample Commands](#6-llama-bench-sample-commands)
7. [llm_bench Sample Commands](#7-llm_bench-sample-commands)
8. [Output Metrics Reference](#8-output-metrics-reference)
9. [Excel Results Sheet Columns](#9-excel-results-sheet-columns)
10. [Environment Capture Script](#10-environment-capture-script)
11. [Known Issues & Failure Documentation](#11-known-issues--failure-documentation)

---

## 1. Benchmark Configuration Decisions

### ✅ Final Standard Config

| Parameter | Value | Rationale |
|---|---|---|
| **Input tokens** | **32** | Realistic short-prompt workload; `0` is not supported (silently skipped by both tools) |
| **Output tokens** | **32 and 128** | Two points: short-burst vs sustained generation |
| **Batch size** | See note below | Different semantics per tool — read carefully |
| **Warmup runs** | **1** (automatic default in both tools) | Ensures JIT compilation, caches are warm before measuring |
| **Measured runs** | **3** | Enough for stable average; use min/avg/median |
| **Context size** | **256** | 32 + 128 + margin = 160 minimum; 256 fits all cases; safe for NPU |

### ⚠️ Batch Size — Critical Distinction

The term "batch size" means **completely different things** in the two tools:

| Tool | Parameter | What it controls | Recommended value |
|---|---|---|---|
| **llama-bench** | `-b N` (`--batch-size`) | **Prompt processing chunk size** — how many prompt tokens are processed per decode step. Setting `-b 1` forces one-token-at-a-time prefill, which artificially collapses PP throughput and is unrealistic. | **`-b 32`** (match prompt length) |
| **llm_bench** | `-bs N` (`--batch_size`) | **Inference batch = number of parallel sequences**. `-bs 1` means single-sequence latency mode — this IS the correct setting for latency benchmarking. | **`-bs 1`** |

### ⚠️ Input Tokens = 0 — Not Supported

Both tools **silently skip** zero-length prompt or generation entries:
- `llama-bench -p 0` → skipped, no PP row generated
- `llama-bench -n 0` → skipped, no TG row generated
- `llm_bench` with an empty prompt → undefined/error behaviour

**Use `-p 32` (llama-bench) and a verified ~32-token prompt file (llm_bench).** If you want a pure TG-only test in llama-bench, use `-pg 32,128` (combined prefill+generate) rather than trying to zero out `-p`.

---

## 2. Parameter Mapping Table

| Concept | llama-bench | llm_bench | Notes |
|---|---|---|---|
| **Model path** | `-m <path.gguf>` | `-m <ir_dir_or_path.gguf>` | llm_bench accepts both IR dirs and GGUF files |
| **Input tokens** | `-p 32` (synthetic token IDs) | `-pf prompt_32tok.jsonl` (real text, ~32 tokens) | Verify actual token count with tokenizer |
| **Output tokens** | `-n 32` or `-n 128` | `-ic 32` or `-ic 128` | Direct equivalent: max output tokens |
| **Combined PP+TG** | `-pg 32,128` | `-pf prompt_32tok.jsonl -ic 128` | Most comparable combined mode |
| **Warmup** | Automatic (1 run before `-r` reps) | Iteration 0 always = warmup (auto) | Do **not** add extra warmup flags — both default to 1 warmup |
| **Disable warmup** | `--no-warmup` | N/A (cannot skip iter 0) | Only use `--no-warmup` in llama-bench if explicitly needed |
| **Measured iterations** | `-r 3` (`--repetitions`) | `-n 3` (`--num_iters`) | Equivalent; both exclude the warmup from averages |
| **Context size** | `-c 256` |Implicit from model config; use `-lc '{"MAX_KV_CACHE_SIZE": N}'` or model config  | Set `-c 256` in llama-bench; GenAI use `-lc '{"MAX_KV_CACHE_SIZE": N}'` |
| **Batch size (prefill chunk)** | `-b 32` | N/A (internal) | Match to prompt length in llama-bench |
| **Batch size (sequences)** | N/A | `-bs 1` | Single-sequence latency mode in llm_bench |
| **Device** | Env var `GGML_OPENVINO_DEVICE` or build flag | `-d CPU / GPU / NPU` | |
| **Flash Attention** | `-fa 1` (**required** for OV backend) | Enabled by default in GenAI | Always use `-fa 1` with llama-bench OV backend |
| **Output format** | `-o csv` or `-o json` or `-o md` (default) | `-r report.csv` or `-rj report.json` | Both support CSV/JSON |
| **Seed** | N/A (synthetic tokens, no seed needed) | `-s 42` (default) | Fix seed in llm_bench for reproducibility |
| **Disable prompt permutation** | N/A (synthetic; no permutation) | `--disable_prompt_permutation` | **Always use** in llm_bench for reproducible runs |
| **Memory measurement** | Not built-in (use OS tools) | `-mc 2 -mc_dir ./mem_logs` | Run separately from perf benchmarks |
| **OV model caching** | `GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache` | Automatic (GenAI manages) | Enable cache for faster re-runs; **not supported on NPU** |
| **Stateful KV cache** | `GGML_OPENVINO_STATEFUL_EXECUTION=1` | Always stateful in GenAI | Required for CPU/GPU OV backend in llama-bench; `unset` for NPU |

---

## 3. Setup & Build

### A. llama.cpp — Build All Backends

```bash name=setup_llama_cpp.sh
# Clone
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Save commit for reproducibility
git log -1 --oneline > ../logs/llama_cpp_commit.txt

# --- Prerequisites (Linux) ---
sudo apt-get update
sudo apt-get install -y build-essential libcurl4-openssl-dev libtbb12 \
    cmake ninja-build python3-pip curl wget tar \
    ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd

# --- Build 1: Default CPU backend ---
cmake -B build/Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --parallel

# --- Build 2: Vulkan backend (GPU) ---
cmake -B build/ReleaseVulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build/ReleaseVulkan --parallel

# --- Build 3: OpenVINO backend (CPU, GPU, NPU) ---
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

**Windows (x64 Native Tools Command Prompt for VS 2022):**
```cmd name=setup_llama_cpp_windows.cmd
REM Default CPU
cmake -B build\Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build\Release --parallel

REM OpenVINO backend
"C:\Program Files (x86)\Intel\openvino_2026.0\setupvars.bat"
cmake -B build\ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON -DLLAMA_CURL=OFF -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build\ReleaseOV --parallel
```

### B. OpenVINO GenAI llm_bench — Python Setup

```bash name=setup_llm_bench.sh
python3 -m venv ov-llm-bench-env
source ov-llm-bench-env/bin/activate
pip install --upgrade pip

git clone https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt

# Save commit for reproducibility
git -C ../.. log -1 --oneline > ../../../logs/openvino_genai_commit.txt

# Optional: Hugging Face login for gated models
huggingface-cli login
```

---

## 4. Model Preparation

### A. Download GGUF Models (Q4_0) for llama-bench

```bash name=download_gguf_models.sh
mkdir -p ~/models/gguf

# Llama-3.2-1B-Instruct
wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf \
     -O ~/models/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf

```

### B. Convert Models to OpenVINO IR Format

Three INT4 quantization variants as specified in the task:

```bash name=convert_models_to_ir.sh
source ov-llm-bench-env/bin/activate

MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen3-8B"
  "openbmb/MiniCPM-1B-sft-bf16"
  "tencent/Hunyuan-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "bartowski/DeepSeek-R1-Distill-Llama-8B"
)

for MODEL_ID in "${MODELS[@]}"; do
  MODEL_NAME=$(basename $MODEL_ID)

  # INT4 SYM Channel-wise (group_size=-1 (channel-wise))
  optimum-cli export openvino \
    --model $MODEL_ID \
    --weight-format int4 \
    --sym \
    --group-size -1 \
    ~/models/ir/${MODEL_NAME}/INT4_SYM_CW

  # INT4 SYM Group-size 32
  optimum-cli export openvino \
    --model $MODEL_ID \
    --weight-format int4 \
    --sym \
    --group-size 32 \
    ~/models/ir/${MODEL_NAME}/INT4_SYM_GS32

  # INT4 Group-size default optimum config
  optimum-cli export openvino \
    --model $MODEL_ID \
    --weight-format int4 \
    ~/models/ir/${MODEL_NAME}/INT4_DEFAULT

  echo "✅ Done: $MODEL_NAME"
done
```

---

## 5. Prompt File Setup for llm_bench

llm_bench uses **real text** and a real tokenizer, so you must verify the prompt produces the target token count (~32 tokens).

### Verify Token Count

```python name=verify_token_count.py
from transformers import AutoTokenizer

PROMPT = "Describe the key innovations in artificial intelligence over the last decade, including deep learning, transformers, and large language models."

# Test against a few model tokenizers since token counts vary per model
model_ids = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

for model_id in model_ids:
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        count = len(tok.encode(PROMPT))
        print(f"{model_id:<45}: {count} tokens")
    except Exception as e:
        print(f"{model_id}: ERROR - {e}")
```

### Prompt Files

```jsonl name=prompts/prompt_32tok.jsonl
{"prompt": "Describe the key innovations in artificial intelligence over the last decade, including deep learning, transformers, and large language models."}
```

> **Note:** Token counts vary by model tokenizer (typically ±5 tokens for this prompt). This is acceptable — document the actual token count as reported in llm_bench's `input_size` CSV column. The goal is a reasonably comparable short-prompt scenario, not exact token parity.

---

## 6. llama-bench Sample Commands

### Configuration Notes
- `-p 32` — 32 synthetic prompt tokens
- `-n 32,128` — run TG with 32 then 128 output tokens in one invocation
- `-b 32` — prefill chunk = prompt length (realistic, not artificially slow)
- `-r 3` — 3 measured repetitions (1 automatic warmup precedes these)
- `-c 256` — context window (fits 32+128+margin; safe for NPU)
- `-fa 1` — **required** for OpenVINO backend only
- `-o csv` — machine-readable output for Excel import

```bash name=run_llama_bench_all.sh
#!/bin/bash
# ============================================================
# llama-bench: All backends for one model
# Usage: MODEL=~/models/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf
#        MODEL_TAG=Llama-3.2-1B  bash run_llama_bench_all.sh
# ============================================================

MODEL=${MODEL:-~/models/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf}
MODEL_TAG=${MODEL_TAG:-model}
LOGDIR=logs/${MODEL_TAG}
mkdir -p $LOGDIR

# Common bench args (no -fa here; added per-backend below)
BENCH_ARGS="-p 32 -n 32,128 -b 32 -r 3 -c 256 -o csv"

# -----------------------------------------------------------
# 1. Default GGML CPU backend
# -----------------------------------------------------------
echo ">>> Running: llama_ggml_cpu"
./build/Release/bin/llama-bench \
  -m $MODEL $BENCH_ARGS \
  2>&1 | tee $LOGDIR/llama_ggml_cpu.log

# -----------------------------------------------------------
# 2. Vulkan backend (GPU)
# -----------------------------------------------------------
echo ">>> Running: llama_vulkan_gpu"
./build/ReleaseVulkan/bin/llama-bench \
  -m $MODEL $BENCH_ARGS \
  2>&1 | tee $LOGDIR/llama_vulkan_gpu.log

# -----------------------------------------------------------
# 3. OpenVINO backend — CPU
# -----------------------------------------------------------
echo ">>> Running: llama_ov_cpu"
source /opt/intel/openvino/setupvars.sh
export GGML_OPENVINO_DEVICE=CPU
export GGML_OPENVINO_STATEFUL_EXECUTION=1
export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache

./build/ReleaseOV/bin/llama-bench \
  -m $MODEL $BENCH_ARGS -fa 1 \
  2>&1 | tee $LOGDIR/llama_ov_cpu.log

# -----------------------------------------------------------
# 4. OpenVINO backend — GPU
# -----------------------------------------------------------
echo ">>> Running: llama_ov_gpu"
export GGML_OPENVINO_DEVICE=GPU
export GGML_OPENVINO_STATEFUL_EXECUTION=1   # Required; stateless has known issues on GPU
export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache

./build/ReleaseOV/bin/llama-bench \
  -m $MODEL $BENCH_ARGS -fa 1 \
  2>&1 | tee $LOGDIR/llama_ov_gpu.log

# -----------------------------------------------------------
# 5. OpenVINO backend — NPU
# IMPORTANT: unset GGML_OPENVINO_STATEFUL_EXECUTION for NPU.
# Setting it to 0 does NOT work — you must fully unset it.
# GGML_OPENVINO_CACHE_DIR is also NOT supported on NPU.
# -----------------------------------------------------------
echo ">>> Running: llama_ov_npu"
export GGML_OPENVINO_DEVICE=NPU
unset GGML_OPENVINO_CACHE_DIR

./build/ReleaseOV/bin/llama-bench \
  -m $MODEL $BENCH_ARGS -fa 1 \
  2>&1 | tee $LOGDIR/llama_ov_npu.log

echo ">>> All llama-bench runs complete. Logs in $LOGDIR/"
```

**Windows equivalents (Command Prompt):**
```cmd name=run_llama_bench_windows.cmd
set MODEL=C:\models\gguf\Llama-3.2-1B-Instruct-Q4_0.gguf
set BENCH_ARGS=-p 32 -n 32,128 -b 32 -r 3 -c 256 -o csv

REM --- OV CPU ---
"C:\Program Files (x86)\Intel\openvino_2026.0\setupvars.bat"
set GGML_OPENVINO_DEVICE=CPU
set GGML_OPENVINO_STATEFUL_EXECUTION=1
set GGML_OPENVINO_CACHE_DIR=C:\tmp\ov_cache
build\ReleaseOV\bin\llama-bench.exe -m %MODEL% %BENCH_ARGS% -fa 1

REM --- OV GPU ---
set GGML_OPENVINO_DEVICE=GPU
set GGML_OPENVINO_STATEFUL_EXECUTION=1
build\ReleaseOV\bin\llama-bench.exe -m %MODEL% %BENCH_ARGS% -fa 1

REM --- OV NPU (unset stateful — "set =0" does NOT work) ---
set GGML_OPENVINO_DEVICE=NPU
set GGML_OPENVINO_STATEFUL_EXECUTION=
set GGML_OPENVINO_CACHE_DIR=
build\ReleaseOV\bin\llama-bench.exe -m %MODEL% %BENCH_ARGS% -fa 1
```

### Running All Models (Batch Script)

```bash name=run_llama_bench_all_models.sh
#!/bin/bash
# Run all models across all backends

MODELS=(
  "Llama-3.2-1B-Instruct-Q4_0.gguf"
  "Llama-3.1-8B-Instruct-Q4_0.gguf"
  "Phi-3-mini-4k-instruct-Q4_0.gguf"
  "Qwen2.5-1.5B-Instruct-Q4_0.gguf"
  "Qwen3-8B-Q4_0.gguf"
  "MiniCPM-1B-Q4_0.gguf"
  "Hunyuan-7B-Instruct-Q4_0.gguf"
  "Mistral-7B-Instruct-v0.3-Q4_0.gguf"
  "DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf"
)

BENCH_ARGS="-p 32 -n 32,128 -b 32 -r 3 -c 256 -o csv"
source /opt/intel/openvino/setupvars.sh

for GGUF in "${MODELS[@]}"; do
  TAG="${GGUF%.gguf}"
  MODEL=~/models/gguf/$GGUF
  LOGDIR=logs/$TAG
  mkdir -p $LOGDIR

  if [ ! -f "$MODEL" ]; then
    echo "⚠️  Skipping (not found): $MODEL" | tee -a $LOGDIR/FAILED.log
    continue
  fi

  echo "=== $TAG ===" | tee -a logs/run_summary.log

  # GGML CPU
  ./build/Release/bin/llama-bench -m $MODEL $BENCH_ARGS \
    2>&1 | tee $LOGDIR/llama_ggml_cpu.log || echo "FAILED: ggml_cpu $TAG" >> logs/run_summary.log

  # Vulkan GPU
  ./build/ReleaseVulkan/bin/llama-bench -m $MODEL $BENCH_ARGS \
    2>&1 | tee $LOGDIR/llama_vulkan_gpu.log || echo "FAILED: vulkan_gpu $TAG" >> logs/run_summary.log

  # OV CPU
  export GGML_OPENVINO_DEVICE=CPU; export GGML_OPENVINO_STATEFUL_EXECUTION=1; export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache
  ./build/ReleaseOV/bin/llama-bench -m $MODEL $BENCH_ARGS -fa 1 \
    2>&1 | tee $LOGDIR/llama_ov_cpu.log || echo "FAILED: ov_cpu $TAG" >> logs/run_summary.log

  # OV GPU
  export GGML_OPENVINO_DEVICE=GPU
  ./build/ReleaseOV/bin/llama-bench -m $MODEL $BENCH_ARGS -fa 1 \
    2>&1 | tee $LOGDIR/llama_ov_gpu.log || echo "FAILED: ov_gpu $TAG" >> logs/run_summary.log

  # OV NPU
  export GGML_OPENVINO_DEVICE=NPU; unset GGML_OPENVINO_STATEFUL_EXECUTION; unset GGML_OPENVINO_CACHE_DIR
  ./build/ReleaseOV/bin/llama-bench -m $MODEL $BENCH_ARGS -fa 1 \
    2>&1 | tee $LOGDIR/llama_ov_npu.log || echo "FAILED: ov_npu $TAG" >> logs/run_summary.log

done

echo "All done. See logs/run_summary.log for failures."
```

---

## 7. llm_bench Sample Commands

### Configuration Notes
- `-pf prompts/prompt_32tok.jsonl` — real text prompt (~32 tokens)
- `-ic 32` or `-ic 128` — output token limit
- `-bs 1` — single-sequence latency mode (correct for latency benchmarking)
- `-n 3` — 3 measured iterations (iteration 0 = warmup, excluded from averages automatically)
- `--disable_prompt_permutation` — same prompt every run (reproducible; disables anti-prefix-cache shuffling)
- `-s 42` — fixed seed

```bash name=run_llm_bench_all.sh
#!/bin/bash
# ============================================================
# llm_bench: All backends for one model
# Usage: MODEL_IR=~/models/ir/Llama-3.2-1B-Instruct
#        MODEL_GGUF=~/models/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf
#        MODEL_TAG=Llama-3.2-1B  bash run_llm_bench_all.sh
# ============================================================

MODEL_TAG=${MODEL_TAG:-model}
LOGDIR=logs/${MODEL_TAG}
mkdir -p $LOGDIR

cd openvino.genai/tools/llm_bench
source ../../../ov-llm-bench-env/bin/activate

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
  python benchmark.py -m $MODEL -d $DEVICE -ic $IC $COMMON_ARGS \
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
IR_GS128=${MODEL_IR}/INT4_DEFAULT
for IC in 32 128; do
  run_bench "ov_genai_ir_cpu_gsDefault"  $IR_GSDefault  CPU  $IC  INT4_Default
  run_bench "ov_genai_ir_gpu_gsDefault"  $IR_GSDefault  GPU  $IC  INT4_Default
  run_bench "ov_genai_ir_npu_gsDefault"  $IR_GSDefault  NPU  $IC  INT4_Default
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
```

### Calling the Script per Model

```bash name=run_llm_bench_all_models.sh
#!/bin/bash
declare -A MODELS=(
  ["Llama-3.2-1B-Instruct"]="meta-llama/Llama-3.2-1B-Instruct"
  ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
  ["Phi-3-mini-4k-instruct"]="microsoft/Phi-3-mini-4k-instruct"
  ["Qwen2.5-1.5B-Instruct"]="Qwen/Qwen2.5-1.5B-Instruct"
  ["Qwen3-8B"]="Qwen/Qwen3-8B"
  ["MiniCPM-1B"]="openbmb/MiniCPM-1B-sft-bf16"
  ["Hunyuan-7B-Instruct"]="tencent/Hunyuan-7B-Instruct"
  ["Mistral-7B-Instruct-v0.3"]="mistralai/Mistral-7B-Instruct-v0.3"
  ["DeepSeek-R1-Distill-Llama-8B"]="bartowski/DeepSeek-R1-Distill-Llama-8B"
)

for MODEL_TAG in "${!MODELS[@]}"; do
  MODEL_IR=~/models/ir/$MODEL_TAG
  MODEL_GGUF=~/models/gguf/${MODEL_TAG}-Q4_0.gguf
  export MODEL_IR MODEL_GGUF MODEL_TAG
  bash run_llm_bench_all.sh
done
```

---

## 8. Output Metrics Reference

### llama-bench Output Metrics

llama-bench runs two test types and labels them in its output:
- `pp32` = prompt-processing (prefill) of 32 tokens
- `tg32` / `tg128` = token-generation of 32 or 128 tokens

| **Metric** | **llama-bench output** | **Description** |
|---|---|---|
| Test type | `test` column (`pp32`, `tg32`, `tg128`) | What was measured |
| Input tokens | `n_prompt` field | Prompt token count |
| Output tokens | `n_gen` field | Generation token count |
| **Throughput** | **`avg_ts` (t/s)** | **Average tokens/second across reps** |
| Throughput std dev | `stddev_ts` (t/s) | Variation across repetitions |
| Average time | `avg_ns` (ns) | Average run time in nanoseconds |
| Std dev time | `stddev_ns` (ns) | Variation in nanoseconds |
| **PP speed (tok/s)** | **`avg_ts` where `n_prompt>0, n_gen=0`** | **Prompt eval / prefill speed** |
| **TG speed (tok/s)** | **`avg_ts` where `n_gen>0, n_prompt=0`** | **Token generation speed** |
| **TTFT proxy (ms)** | **`avg_ns / 1e6 / n_prompt`** | **Approx first-token time** (not exact; see note) |
| Model size | `model_size` | Model file size in bytes |
| Model params | `model_n_params` | Parameter count |
| Backend | `backends` | Registered backends (OPENVINO, Vulkan, CPU, etc.) |
| Build commit | `build_commit` | llama.cpp git commit |

> ⚠️ **TTFT in llama-bench:** llama-bench measures total prefill time, not true TTFT. `pp32` avg_ts gives you the prefill processing rate; to get approximate TTFT: `(32 / pp_tok_s) * 1000 ms`. For true TTFT, use llm_bench's `1st_latency(ms)`.

### llm_bench Output Metrics (CSV columns)

| **Metric** | **llm_bench CSV column** | **Description** |
|---|---|---|
| **Load + Compile time** | `pretrain_time(s)` | Full model load + compile time (seconds) |
| **Input tokens** | `input_size` | Actual tokenized input count (real tokenizer) |
| **Output token limit** | `infer_count` | The `-ic` value set |
| **Actual output tokens** | `output_size` | Tokens actually generated |
| **TTFT** | `1st_latency(ms)` | True time-to-first-token (ms) |
| **Token generation speed** | `2nd_avg_latency(ms)` | Average latency per output token (tokens 2..N) — convert: `1000 / 2nd_avg_latency = tok/s` |
| Overall latency | `latency(ms)` | ms/token across entire output (incl. first token) |
| Total generation time | `generation_time(s)` | End-to-end time for one inference |
| First inference latency | `1st_infer_latency(ms)` | First model forward-pass latency |
| Avg subsequent latency | `2nd_infer_avg_latency(ms)` | Avg of subsequent forward passes |
| Memory (RSS) | `max_rss_mem(GiB)` | Peak RSS memory during inference |
| Memory (System) | `max_sys_mem(GiB)` | Peak system memory during inference |
| Compile memory | `compile_max_rss_mem(GiB)` | Peak RSS during model compilation |
| Memory increase | `max_increase_rss_mem(GiB)` | Increase vs pre-run baseline |
| Precision/Quantization | `precision` | Inferred from model path (e.g., INT4_SYM_GS32) |
| Iteration | `iteration` | `0`=warmup, `1-3`=measured, `avg`/`mini`/`median` = summaries |
| Framework | `framework` | `ov` or `ov(gguf)` etc. |

---

## 9. Excel Results Sheet Columns

Use these normalized column names. Both tools' outputs map into the same schema.

| **Excel Column** | **From llama-bench** | **From llm_bench CSV** | **Unit** |
|---|---|---|---|
| Model | `model_filename` (basename) | `model` | — |
| Framework | See framework tag table below | `framework` | — |
| Device | `GGML_OPENVINO_DEVICE` env / build | `device` | — |
| Quantization | Model filename (Q4_0) | `precision` column | — |
| NUM_INPUT_TOKENS | `n_prompt` | `input_size` | tokens |
| NUM_OUTPUT_TOKENS | `n_gen` | `output_size` | tokens |
| LOAD_COMPILE_TIME (ms) | *(use `/usr/bin/time -v`)* | `pretrain_time(s)` × 1000 | ms |
| **TTFT (ms)** | `(n_prompt / avg_ts) × 1000` for pp row | `1st_latency(ms)` | ms |
| **PROMPT_EVAL / PP (tok/s)** | `avg_ts` for `pp32` row | `input_size / (1st_latency_ms / 1000)` | tok/s |
| **TOKEN_GEN (tok/s)** | `avg_ts` for `tg32` / `tg128` row | `1000 / 2nd_avg_latency(ms)` | tok/s |
| AVG_TOKEN_LATENCY (ms/tok) | `1000 / avg_ts` for TG row | `2nd_avg_latency(ms)` | ms/tok |
| GENERATION_TIME (s) | `avg_ns / 1e9` | `generation_time(s)` | s |
| MEMORY_RSS (GiB) | *(use OS tools / `/usr/bin/time`)* | `max_rss_mem(GiB)` | GiB |
| MEMORY_SYS (GiB) | *(use OS tools)* | `max_sys_mem(GiB)` | GiB |
| COMPILE_MEMORY (GiB) | N/A | `compile_max_rss_mem(GiB)` | GiB |
| TG_STDDEV (tok/s) | `stddev_ts` for TG row | Compute from `avg`/`mini`/`median` rows | tok/s |
| llama.cpp commit | `build_commit` | — | — |
| OV/GenAI version | — | From `openvino.__version__` at run time | — |

### Framework Tag Reference

| **Excel Framework Tag** | **llama-bench build** | **OV env vars** | **llm_bench** |
|---|---|---|---|
| `llamaCPP_ggml_CPU` | `build/Release` | — | — |
| `llamaCPP_Vulkan_GPU` | `build/ReleaseVulkan` | — | — |
| `llamaCPP_OV_CPU` | `build/ReleaseOV` | `DEVICE=CPU, STATEFUL=1` | — |
| `llamaCPP_OV_GPU` | `build/ReleaseOV` | `DEVICE=GPU, STATEFUL=1` | — |
| `llamaCPP_OV_NPU` | `build/ReleaseOV` | `DEVICE=NPU, unset STATEFUL` | — |
| `OV_GENAI_IR_CPU` | — | — | `-d CPU -m IR_dir` |
| `OV_GENAI_IR_GPU` | — | — | `-d GPU -m IR_dir` |
| `OV_GENAI_IR_NPU` | — | — | `-d NPU -m IR_dir` |
| `OV_GENAI_GGUF_CPU` | — | — | `-d CPU -m .gguf` |
| `OV_GENAI_GGUF_GPU` | — | — | `-d GPU -m .gguf` |
| `OV_GENAI_GGUF_NPU` | — | — | `-d NPU -m .gguf` |

---

## 10. Environment Capture Script

Run this **once per machine** before starting any benchmark session:

```bash name=capture_env.sh
#!/bin/bash
mkdir -p logs
OUT=logs/env_info_$(date +%Y%m%d_%H%M%S).txt

echo "=== DATE & HOSTNAME ===" | tee $OUT
date | tee -a $OUT
hostname | tee -a $OUT

echo -e "\n=== OS VERSION ===" | tee -a $OUT
cat /etc/os-release | tee -a $OUT
uname -a | tee -a $OUT

echo -e "\n=== CPU INFO ===" | tee -a $OUT
lscpu | tee -a $OUT

echo -e "\n=== MEMORY ===" | tee -a $OUT
free -h | tee -a $OUT

echo -e "\n=== GPU / NPU DEVICES ===" | tee -a $OUT
clinfo --list 2>/dev/null | tee -a $OUT || echo "clinfo not found" | tee -a $OUT
ls /dev/accel* 2>/dev/null | tee -a $OUT || echo "No /dev/accel (NPU) device found" | tee -a $OUT
ls /dev/dri/ 2>/dev/null | tee -a $OUT

echo -e "\n=== DRIVER VERSIONS ===" | tee -a $OUT
dpkg -l 2>/dev/null | grep -E "intel-opencl|level-zero|intel-npu|intel-gpu|intel-gsc" | tee -a $OUT

echo -e "\n=== OpenVINO VERSION ===" | tee -a $OUT
source /opt/intel/openvino/setupvars.sh 2>/dev/null
python3 -c "import openvino; print('openvino:', openvino.__version__)" 2>/dev/null | tee -a $OUT
python3 -c "import openvino_genai; print('openvino_genai:', openvino_genai.__version__)" 2>/dev/null | tee -a $OUT

echo -e "\n=== llama.cpp VERSION / COMMIT ===" | tee -a $OUT
[ -d ~/llama.cpp ] && git -C ~/llama.cpp log -1 --pretty='%H %s' | tee -a $OUT

echo -e "\n=== openvino.genai VERSION / COMMIT ===" | tee -a $OUT
[ -d ~/openvino.genai ] && git -C ~/openvino.genai log -1 --pretty='%H %s' | tee -a $OUT

echo -e "\n=== RELEVANT ENVIRONMENT VARIABLES ===" | tee -a $OUT
env | grep -E "GGML|OMP|MKL|OPENVINO|SYCL|HF_TOKEN|HUGGINGFACE" | grep -v TOKEN | tee -a $OUT

echo -e "\nEnvironment info saved to: $OUT"
```

---

## 11. Known Issues & Failure Documentation

Document all failures in a `logs/FAILURES.md` file using this template:

````markdown name=logs/FAILURES.md
# Benchmark Failure Log

## Template
| Date | Model | Framework | Device | Quant | Input | Output | Error Summary | Workaround/Status |
|------|-------|-----------|--------|-------|-------|--------|---------------|-------------------|

## Known Issues

### llama-bench + OpenVINO Backend
| Issue | Condition | Workaround |
|---|---|---|
| `-p 0` or `-n 0` is silently skipped | Any backend | Use `-p 32` minimum; use `-pg` for combined |
| GPU stateless execution failures | `llama-bench` OV GPU | Always set `GGML_OPENVINO_STATEFUL_EXECUTION=1` for GPU |
| NPU context too large → crash | Default ctx = model training ctx (e.g. 131072) | Always set `-c 256` (or `-c 512`) for NPU |
| NPU: `STATEFUL=0` env var doesn't disable stateful | NPU device | Must `unset GGML_OPENVINO_STATEFUL_EXECUTION` entirely |
| NPU: model caching not supported | `GGML_OPENVINO_CACHE_DIR` set + NPU | `unset GGML_OPENVINO_CACHE_DIR` for NPU runs |
| `-fa 1` required for llama-bench OV backend | OV backend, all devices | Always add `-fa 1` for OV backend |
| `--context-shift` not supported with OV backend | `llama-cli` | Known limitation; do not use |

### llm_bench
| Issue | Condition | Workaround |
|---|---|---|
| Prompt permutation changes token count | Default (permutation enabled) | Always use `--disable_prompt_permutation` |
| Actual output token count < `-ic` limit | Model generates EOS early | Check `output_size` column; document actual count |
| GenAI manages context internally | No `-c` param | Document if OOM occurs; reduce `-ic` |

### Model-Specific Issues
| Model | Framework | Device | Quant | Status | Notes |
|---|---|---|---|---|---|
| (fill as you go) | | | | | |
````

---

## ⚡ Quick Reference Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════════╗
║              BENCHMARK QUICK REFERENCE                               ║
╠══════════════════════════════════════════════════════════════════════╣
║ llama-bench standard run:                                            ║
║   -p 32 -n 32,128 -b 32 -r 3 -c 256 [-fa 1 for OV backend]         ║
║                                                                      ║
║ llm_bench standard run:                                              ║
║   -pf prompt_32tok.jsonl -ic 32  -n 3 -bs 1                         ║
║   -pf prompt_32tok.jsonl -ic 128 -n 3 -bs 1                         ║
║   --disable_prompt_permutation -s 42 --task text_gen                 ║
╠══════════════════════════════════════════════════════════════════════╣
║ KEY ENV VARS for llama-bench OV backend:                             ║
║   CPU/GPU:  GGML_OPENVINO_STATEFUL_EXECUTION=1                       ║
║   GPU:      GGML_OPENVINO_DEVICE=GPU                                 ║
║   NPU:      GGML_OPENVINO_DEVICE=NPU                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║ METRIC MAPPING:                                                      ║
║   PP tok/s:  llama-bench avg_ts (pp32)  ↔  llm_bench:              ║
║              input_size / (1st_latency_ms / 1000)                   ║
║   TG tok/s:  llama-bench avg_ts (tg128) ↔  llm_bench:              ║
║              1000 / 2nd_avg_latency(ms)                             ║
║   TTFT (ms): llama-bench: (32 / pp_tok_s) × 1000                   ║
║              llm_bench:   1st_latency(ms)  ← more accurate          ║
║   LOAD+COMPILE: /usr/bin/time -v (llama-bench) or                   ║
║                 pretrain_time(s) × 1000 (llm_bench)                 ║
╠══════════════════════════════════════════════════════════════════════╣
║ CONTEXT REQUIREMENTS:                                                ║
║   n_prompt + n_gen ≤ context:  32 + 128 = 160 → use -c 256 ✅       ║
║   NPU: always explicitly set -c 256 (default can be 131072!) ⚠️     ║
╚══════════════════════════════════════════════════════════════════════╝
```