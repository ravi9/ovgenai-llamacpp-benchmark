# LLM Benchmarking Suite

This repository contains a comprehensive benchmarking suite for evaluating Large Language Models (LLMs) across different frameworks (llama.cpp, OpenVINO GenAI) and hardware backends (CPU, GPU, NPU).

## Overview

The benchmarking workflow consists of:
1. **Setup**: Build frameworks and download models
2. **Conversion**: Convert models to required formats (OpenVINO IR)
3. **Execution**: Run benchmarks across all frameworks and devices
4. **Analysis**: Parse and consolidate benchmark results

---

## Scripts Documentation

### 1. `build_llama_cpp.sh`

**Purpose**: Clone and build llama.cpp with multiple backend support.

**Usage**:
```bash
./build_llama_cpp.sh
```

**What it does**:
- Clones or updates the llama.cpp repository
- Saves current git commit to `logs/llama_cpp_commit.txt` for reproducibility
- Builds three variants:
  - **CPU backend** (default): `build/Release/bin/`
  - **Vulkan backend** (GPU): `build/ReleaseVulkan/bin/` (if Vulkan SDK available)
  - **OpenVINO backend** (CPU/GPU/NPU): `build/ReleaseOV/bin/` (if OpenVINO installed)

**Prerequisites**:
```bash
# Build tools
sudo apt-get install -y build-essential cmake ninja-build

# Optional: Vulkan SDK for GPU support
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
  https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update
sudo apt install vulkan-sdk

# Optional: OpenVINO for multi-device support
# Install from: https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html
```

**Output**:
- Binary executables in `llama.cpp/build/*/bin/`
- Commit hash in `logs/llama_cpp_commit.txt`

---

### 2. `setup_llm_bench.sh`

**Purpose**: Set up OpenVINO GenAI environment and llm_bench tool.

**Usage**:
```bash
./setup_llm_bench.sh
```

**What it does**:
- Creates Python virtual environment in `env/`
- Activates the environment and upgrades pip
- Clones OpenVINO GenAI repository
- Installs llm_bench dependencies from `requirements.txt`
- Saves current git commit to `logs/openvino_genai_commit.txt` for reproducibility
- Prompts for Hugging Face login (optional, needed for gated models)

**Prerequisites**:
```bash
# Python 3.8 or higher
python3 --version

# Git
sudo apt-get install git
```

**Hugging Face authentication**:
For gated models (e.g., Llama models), you'll need a Hugging Face token:
1. Create account at https://huggingface.co
2. Generate access token at https://huggingface.co/settings/tokens
3. Accept model license agreements on model pages
4. Provide token when prompted by `huggingface-cli login`

**Output**:
- Python virtual environment in `env/`
- OpenVINO GenAI repository in `openvino.genai/`
- Commit hash in `logs/openvino_genai_commit.txt`
- Hugging Face credentials stored in `~/.huggingface/token`

**Note**: After running this script, always activate the environment before using OpenVINO GenAI tools:
```bash
source env/bin/activate
```

---

### 3. `download_gguf.sh`

**Purpose**: Download GGUF model files from URLs listed in a text file.

**Usage**:
```bash
./download_gguf.sh ggufs.txt
```

**Input format** (`ggufs.txt`):
```
# Comments and empty lines are ignored
https://huggingface.co/user/model/resolve/main/model-Q4_0.gguf
https://huggingface.co/user/model2/resolve/main/model2-Q4_0.gguf
```

**What it does**:
- Creates `models/gguf/` directory if it doesn't exist
- Downloads each GGUF file from the provided URLs
- Skips comments and empty lines
- Preserves original filenames

**Output**:
- Downloaded GGUF files in `models/gguf/`

---

### 4. `convert_models_to_irs.sh`

**Purpose**: Convert Hugging Face models to OpenVINO IR format with multiple quantization strategies.

**Usage**:
```bash
./convert_models_to_irs.sh models.txt
```

**Input format** (`models.txt`):
```
"username/model-name"
"meta-llama/Llama-3.1-8B-Instruct"
"Qwen/Qwen2.5-1.5B-Instruct"
```

**What it does**:
For each model, creates three INT4 quantized variants:

1. **INT4_SYM_CW** (Channel-wise): Per-channel quantization, no grouping
   ```bash
   --weight-format int4 --sym --group-size -1
   ```

2. **INT4_SYM_GS32** (Group-size 32): Grouped quantization with 32 weights per group
   ```bash
   --weight-format int4 --sym --group-size 32
   ```

3. **INT4_DEFAULT**: Default INT4 quantization
   ```bash
   --weight-format int4
   ```

**Prerequisites**:
```bash
# Install optimum-cli
pip install optimum[openvino]
```

**Output**:
```
models/ir/
└── model-name/
    ├── INT4_SYM_CW/
    ├── INT4_SYM_GS32/
    └── INT4_DEFAULT/
```

---

### 5. `verify_tocken_count.py`

**Purpose**: Verify tokenization consistency across different model tokenizers.

**Usage**:
```bash
python verify_tocken_count.py
```

**What it does**:
- Tests a predefined prompt against all model tokenizers in `models/ir/`
- Counts tokens for each model
- Saves results to timestamped file: `token_count_verification_YYYYMMDD_HHMMSS.txt`

**Default prompt**:
> "Describe the key innovations in artificial intelligence over the last decade, including deep learning, transformers, and large language models."

**Output example**:
```
models/ir/Llama-3.1-8B-Instruct/INT4_DEFAULT          : 32 tokens
models/ir/Qwen2.5-1.5B-Instruct/INT4_DEFAULT          : 35 tokens
```

**Use case**: Ensures prompt consistency when comparing models with different tokenizers.

---

### 6. `run_llama_bench.sh`

**Purpose**: Benchmark all GGUF models using llama.cpp across all backends.

**Usage**:
```bash
./run_llama_bench.sh
```

**What it does**:
For each GGUF model in `models/gguf/`, runs benchmarks on:

1. **llama_ggml_cpu**: Default CPU backend
2. **llama_vulkan_gpu**: Vulkan GPU backend
3. **llama_ov_cpu**: OpenVINO CPU backend (with flash attention)
4. **llama_ov_gpu**: OpenVINO GPU backend (stateful execution)
5. **llama_ov_npu**: OpenVINO NPU backend

**Benchmark parameters**:
```
-p 32           # Prompt size: 32 tokens
-n 32,128       # Generate: 32 and 128 tokens
-b 32           # Batch size: 32
-r 3            # Repetitions: 3
-fitc 256       # Fill input context up to 256 tokens
-o csv          # Output format: CSV
```

**OpenVINO configurations**:
- **CPU/GPU**: Stateful execution enabled, cache directory: `/tmp/ov_cache`
- **NPU**: Stateful execution DISABLED (not supported), no cache directory

**Output**:
```
logs/
└── ModelName/
    ├── llama_ggml_cpu.log
    ├── llama_vulkan_gpu.log
    ├── llama_ov_cpu.log
    ├── llama_ov_gpu.log
    └── llama_ov_npu.log
```

**Note**: NPU requires `unset GGML_OPENVINO_STATEFUL_EXECUTION` - setting it to `0` does NOT work.

---

### 7. `run_llm_bench.sh`

**Purpose**: Benchmark all models using OpenVINO GenAI's llm_bench tool.

**Usage**:
```bash
./run_llm_bench.sh
```

**What it does**:
For each model in `models/ir/`, benchmarks:

**A. IR Models** (three quantization variants):
- INT4_SYM_CW (Channel-wise)
- INT4_SYM_GS32 (Group-size 32)
- INT4_DEFAULT

**B. GGUF Models** (via OpenVINO GenAI's GGUF reader)

**Devices tested**: CPU, GPU, NPU

**Input context sizes**: 32, 128 tokens

**Benchmark parameters**:
```
-pf prompts/prompt_32tok.jsonl    # Prompt file
-n 3                                # Iterations
-bs 1                               # Batch size
--disable_prompt_permutation        # No prompt shuffling
-s 42                               # Random seed
--task text_gen                     # Text generation task
```

**Output**:
```
logs/
└── ModelName/
    ├── ov_genai_ir_cpu_cw_ic32.csv
    ├── ov_genai_ir_cpu_cw_ic128.csv
    ├── ov_genai_ir_gpu_cw_ic32.csv
    ├── ov_genai_ir_gpu_cw_ic128.csv
    ├── ov_genai_gguf_cpu_ic32.csv
    ├── ov_genai_gguf_gpu_ic128.csv
    ├── ...
    └── FAILED.log (if any benchmark fails)
```

**Prerequisites**:
```bash
# Activate virtual environment with OpenVINO GenAI
source env/bin/activate
```

---

### 8. `parse_benchmark_logs.py`

**Purpose**: Parse and consolidate all benchmark results into a single CSV file.

**Usage**:
```bash
python3 parse_benchmark_logs.py
```

**What it does**:

1. **Scans** `logs/` directory for all model subdirectories
2. **Parses** two types of benchmark files:
   - **llama-bench logs** (`.log` files): llama.cpp benchmarks
   - **llm_bench CSV files** (`.csv` files): OpenVINO GenAI benchmarks
3. **Extracts** key metrics from each benchmark
4. **Consolidates** results into a timestamped CSV file

**Metrics extracted**:

| Metric | Description | Source |
|--------|-------------|--------|
| Model | Model name | Directory name |
| Framework | Framework tag (e.g., `OV_GENAI_IR_GPU_CW_ic128`) | Filename pattern |
| Device | CPU, GPU, or NPU | Filename/CSV data |
| Quantization | Q4_0, INT4_SYM_CW, INT4_DEFAULT, etc. | Filename/CSV data |
| NUM_INPUT_TOKENS | Number of input tokens | Benchmark data |
| NUM_OUTPUT_TOKENS | Number of generated tokens | Benchmark data |
| LOAD_COMPILE_TIME (ms) | Model load and compile time | llm_bench only |
| TTFT (ms) | Time To First Token | Calculated |
| PROMPT_EVAL / PP (tok/s) | Prompt processing throughput | Calculated |
| TOKEN_GEN (tok/s) | Token generation throughput | Calculated |
| AVG_TOKEN_LATENCY (ms/tok) | Average per-token latency | Calculated |
| GENERATION_TIME (s) | Total generation time | Benchmark data |
| MEMORY_RSS (GiB) | Resident memory usage | llm_bench only |
| MEMORY_SYS (GiB) | System memory usage | llm_bench only |
| COMPILE_MEMORY (GiB) | Compilation memory peak | llm_bench only |
| TG_STDDEV (tok/s) | Token generation std deviation | Calculated |
| llama.cpp commit | Git commit hash | `logs/llama_cpp_commit.txt` |
| OV/GenAI version | OpenVINO GenAI commit | `logs/openvino_genai_commit.txt` |

**Framework tags** include:
- `llamaCPP_ggml_CPU` - llama.cpp default CPU
- `llamaCPP_Vulkan_GPU` - llama.cpp Vulkan
- `llamaCPP_OV_CPU` - llama.cpp with OpenVINO
- `OV_GENAI_GGUF_CPU_ic32` - OpenVINO GenAI GGUF reader, 32 input context
- `OV_GENAI_IR_GPU_CW_ic128` - OpenVINO GenAI IR, GPU, channel-wise, 128 input context
- `OV_GENAI_IR_NPU_GS32_ic32` - OpenVINO GenAI IR, NPU, group-size 32, 32 input context

**Failed benchmark handling**:
- Benchmarks that error out are marked with `FAILED` in all metric columns
- Quantization type is still extracted from error messages when possible

**Output**:
```
benchmarking_results_2026-04-27_14-15.csv
```

**Output format**: CSV with 18 columns, one row per benchmark configuration

---

## Complete Workflow

### Step 1: Setup Environment

```bash
# 1. Build llama.cpp
./build_llama_cpp.sh

# 2. Set up OpenVINO GenAI environment
./setup_llm_bench.sh

# 3. Activate virtual environment (for subsequent steps)
source env/bin/activate
```

### Step 2: Download and Convert Models

```bash
# 1. Create model lists
cat > models.txt << EOF
"meta-llama/Llama-3.1-8B-Instruct"
"Qwen/Qwen2.5-1.5B-Instruct"
"microsoft/Phi-3-mini-4k-instruct"
EOF

cat > ggufs.txt << EOF
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/Llama-3.1-8B-Instruct-Q4_0.gguf
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct-Q4_0.gguf
EOF

# 2. Download GGUF models
./download_gguf.sh ggufs.txt

# 3. Convert to OpenVINO IR
./convert_models_to_irs.sh models.txt

# 4. Optional: Verify token counts
python verify_tocken_count.py
```

### Step 3: Run Benchmarks

```bash
# 1. Run llama.cpp benchmarks (all backends)
./run_llama_bench.sh

# 2. Run OpenVINO GenAI benchmarks
./run_llm_bench.sh
```

**Expected runtime**: Several hours depending on number of models and hardware

### Step 4: Analyze Results

```bash
# Parse all logs and create consolidated CSV
python3 parse_benchmark_logs.py

# View results
head benchmarking_results_*.csv
```

---

## Directory Structure

```
bench/
├── build_llama_cpp.sh              # Build llama.cpp
├── setup_llm_bench.sh              # Set up OpenVINO GenAI
├── download_gguf.sh                # Download GGUF models
├── convert_models_to_irs.sh        # Convert to OpenVINO IR
├── verify_tocken_count.py          # Verify tokenization
├── run_llama_bench.sh              # llama.cpp benchmarks
├── run_llm_bench.sh                # OpenVINO GenAI benchmarks
├── parse_benchmark_logs.py         # Parse and consolidate results
├── models.txt                      # Model list for conversion
├── ggufs.txt                       # GGUF URLs for download
│
├── llama.cpp/                      # llama.cpp repository
│   └── build/
│       ├── Release/                # CPU backend
│       ├── ReleaseVulkan/          # Vulkan GPU backend
│       └── ReleaseOV/              # OpenVINO backend
│
├── openvino.genai/                 # OpenVINO GenAI repository
│   └── tools/
│       └── llm_bench/              # Benchmarking tool
│
├── models/
│   ├── gguf/                       # GGUF model files
│   └── ir/                         # OpenVINO IR models
│       └── ModelName/
│           ├── INT4_SYM_CW/
│           ├── INT4_SYM_GS32/
│           └── INT4_DEFAULT/
│
├── prompts/
│   └── prompt_32tok.jsonl          # Benchmark prompts
│
├── logs/                           # Benchmark logs
│   ├── llama_cpp_commit.txt        # llama.cpp version
│   ├── openvino_genai_commit.txt   # OpenVINO GenAI version
│   └── ModelName/
│       ├── llama_*.log             # llama.cpp results
│       ├── ov_genai_*.csv          # OpenVINO GenAI results
│       └── FAILED.log              # Failed benchmarks
│
├── env/                            # Python virtual environment
│
└── benchmarking_results_*.csv      # Consolidated results
```

---

## Troubleshooting

### Common Issues

**1. Hugging Face authentication fails**
- **Cause**: Invalid token or model access not granted
- **Solution**: Generate a new token at https://huggingface.co/settings/tokens and accept model licenses on model pages

**2. NPU benchmarks fail with llama.cpp**
- **Cause**: Stateful execution not properly disabled
- **Solution**: Use `unset GGML_OPENVINO_STATEFUL_EXECUTION` (not `=0`)

**3. GGUF download fails**
- **Cause**: Invalid URL or authentication required
- **Solution**: Check URL format, may need HuggingFace token for gated models

**4. IR conversion fails**
- **Cause**: Missing dependencies or insufficient disk space
- **Solution**: Ensure `optimum[openvino]` installed, check disk space

**5. Memory errors during benchmarks**
- **Cause**: Model too large for available RAM
- **Solution**: Close other applications, use smaller models, or upgrade RAM

**6. Parse script shows "FAILED" for all metrics**
- **Cause**: Benchmark crashed or model file not found
- **Solution**: Check logs in `logs/ModelName/` for error messages

---

## Performance Tips

1. **Use SSD**: Store models on SSD for faster loading
2. **Close background apps**: Free up RAM and GPU memory
3. **Enable caching**: OpenVINO cache (`/tmp/ov_cache`) speeds up subsequent runs
4. **Batch processing**: Run benchmarks overnight for large model sets
5. **Monitor resources**: Use `htop`, `nvidia-smi`, or similar tools

---

## Citation & License

If you use this benchmarking suite, please cite the frameworks:

- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **OpenVINO**: https://github.com/openvinotoolkit/openvino
- **OpenVINO GenAI**: https://github.com/openvinotoolkit/openvino.genai

---

## Version Information

Last updated: April 27, 2026

This documentation reflects the current state of the benchmarking suite. For updates, check the git log.
