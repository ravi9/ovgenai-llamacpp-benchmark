Design and execute a structured benchmarking study comparing multiple LLMs across different inference frameworks and hardware backends.


## 1. Objective
Benchmark the performance of the following models across multiple runtimes and configurations:

MODEL LIST:

Group - 1 

- GPT-OSS-20B
- Qwen/Qwen3-8B  
- Llama-3.2-1B-Instruct  
- bartowski/DeepSeek-R1-Distill-Llama-8B  
- microsoft/Phi-3.5-mini-instruct  
- microsoft/Phi-3-mini-4k-instruct ( only if Phi-3.5 does not work) 
- tencent/Hunyuan-7B-Instruct  
- LiquidAI/LFM2-2.6B  

Group - 2 

- Llama-3.1-8B-Instruct 
- Qwen/Qwen2.5-1.5B-Instruct 
- openbmb/MiniCPM-1B-sft-bf16 
- mistralai/Mistral-7B-Instruct-v0.3 
- Qwen3-VL-4B-Instruct 
- Qwen3.5-4B 
- ibm-granite/granite-4.0-micro 

---

## 2. Benchmarking Frameworks & Configurations

### A. OpenVINO GenAI (IR Models)
- Convert models to OpenVINO IR format or use existing IR models.
- Quantization settings:
  - INT4, Symmetric (SYM), Group size: -1 (Channel-wise (CW) - NPU friendly)
  - INT4, Symmetric (SYM), Group size: 32 (LlamaCPP Q4_0 equivalent)
  - INT4, Symmetric (SYM), Group size: default (Optimum-intel BKC)
- Use: `llm_bench` tool: https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench
- Devices: CPU, GPU, NPU
- Relevant Links:
  - [Default export parameters for reference](https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/configuration.py)
  - [Supported list/export reference](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/README.md)

### B. OpenVINO GenAI (GGUF Reader)

- Use GGUF models directly
- Use: [`llm_bench` tool](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench)
- Devices: CPU, GPU, NPU

### C. llama.cpp (GGUF Models Only)
- Download models from hugging face
- Use Q4_0 quantized models
- Use llama-bench tool. 
  - [See llamaCPP_OpenVINO backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)
  - To run llama-bench with OV backend, -fa 1 is needed. 
  - Use `set GGML_OPENVINO_STATEFUL_EXECUTION=1` for CPU, GPU.  
- Run with following:
    1. Default CPU backend
    2. Vulkan backend (GPU)
    3. OpenVINO backend with: CPU, GPU, NPU

---

## 3. Output Format

### A. Tabular Results
Provide a table with columns:
- Model
- Framework: 
   - llamaCPP_ggml_CPU,  
   - llamaCPP_Vulcan_GPU,  
   - llamaCPP_OV_CPU,  
   - llamaCPP_OV_GPU,  
   - llamaCPP_OV_NPU,  
   - OV_GENAI_CPU,  
   - OV_GENAI_GPU,  
   - OV_GENAI_NPU,  
   - OV_GENAI_GGUF_CPU,  
   - OV_GENAI_GGUF_GPU
- Device (CPU, GPU, NPU)
- Quantization Type (Q4_0, int4_gs_32, int4_gs_128, int4_gs_cw)
- LOAD_TIME (MS)	
- COMPILE_TIME (MS)	
- PROMPT_EVAL (TK/S)	
- TTFT (MS)	
- TOKEN_GEN (TK/S)	
- NUM_INPUT_TOKENS
- NUM_OUTPUT_TOKENS
- Memory Usage

---

## 4. Constraints
- Ensure reproducibility
- Document failures and unsupported configurations

---

## 5. Execution Approach
Follow these steps:
1. Prepare models (GGUF and IR)
2. Setup/Build/Install each framework/backend
3. Run benchmarks systematically with scripts. Use AI where ever possible.
4. Reproducibility instructions (scripts/commands)
5. Save raw benchmark logs. 
6. Gather results into a excel sheet.
7. Report with Observations/charts

Save:
- OS version
- System hardware details
- Driver versions
- OpenVINO version/commit
- llama.cpp version/commit
- Exact benchmark commands
- Environment variables used