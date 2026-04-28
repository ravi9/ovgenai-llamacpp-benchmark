python3 -m venv env
source env/bin/activate
pip install --upgrade pip

git clone https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt

# Save commit for reproducibility
git -C ../.. log -1 --oneline > ../../../logs/openvino_genai_commit.txt

# Optional: Hugging Face login for gated models
huggingface-cli login