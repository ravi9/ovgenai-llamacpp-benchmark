from transformers import AutoTokenizer
from datetime import datetime

PROMPT = "Describe the key innovations in artificial intelligence over the last decade, including deep learning, transformers, and large language models."

# Test against all model tokenizers from models/ir/
model_ids = [
    "models/ir/LiquidAI_LFM2-2.6B/INT4_DEFAULT",
    "models/ir/meta-llama_Llama-3.1-8B-Instruct/INT4_DEFAULT",
    "models/ir/meta-llama_Llama-3.2-1B-Instruct/INT4_DEFAULT",
    "models/ir/microsoft_Phi-3.5-mini-instruct/INT4_DEFAULT",
    "models/ir/microsoft_Phi-3-mini-4k-instruct/INT4_DEFAULT",
    "models/ir/mistralai_Mistral-7B-Instruct-v0.3/INT4_DEFAULT",
    "models/ir/openbmb_MiniCPM-1B-sft-bf16/INT4_DEFAULT",
    "models/ir/Qwen_Qwen2.5-1.5B-Instruct/INT4_DEFAULT",
    "models/ir/Qwen_Qwen3-8B/INT4_DEFAULT",
]

# Open output file
output_file = f"token_count_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_file, 'w') as f:
    # Write header
    header = f"Token Count Verification - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"Prompt: {PROMPT}\n"
    header += "=" * 80 + "\n\n"
    print(header)
    f.write(header)
    
    # Test each model
    for model_id in model_ids:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            count = len(tok.encode(PROMPT))
            output = f"{model_id:<60}: {count} tokens"
            print(output)
            f.write(output + "\n")
        except Exception as e:
            output = f"{model_id}: ERROR - {e}"
            print(output)
            f.write(output + "\n")
    
    # Write footer
    footer = "\n" + "=" * 80 + "\n"
    footer += f"Results saved to: {output_file}\n"
    print(footer)
    f.write(footer)

print(f"\nOutput saved to: {output_file}")