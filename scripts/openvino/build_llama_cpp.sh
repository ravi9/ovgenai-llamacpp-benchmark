#!/bin/bash

# Clone or update repository
if [ -d "llama.cpp" ]; then
    echo "Directory llama.cpp already exists, updating..."
    cd llama.cpp
    git pull || echo "Warning: git pull failed, continuing with existing code..."
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
fi

# Save commit for reproducibility
mkdir -p ../logs
git log -1 --oneline > ../logs/llama_cpp_commit.txt

# --- Prerequisites (Linux) ---
# Skip sudo commands if packages already installed, otherwise print instructions
echo "Checking prerequisites..."
if ! command -v cmake &> /dev/null || ! command -v ninja &> /dev/null; then
    echo "WARNING: Missing build tools. Please install:"
    echo "  sudo apt-get install -y build-essential cmake ninja-build"
fi

# --- Build 1: Default CPU backend ---
echo "Building CPU backend..."
cmake -B build/Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --parallel

# --- Build 2: Vulkan backend (GPU) ---
echo "Checking for Vulkan SDK..."
if command -v glslc &> /dev/null && pkg-config --exists vulkan 2>/dev/null; then
    echo "Building Vulkan backend..."
    cmake -B build/ReleaseVulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
    cmake --build build/ReleaseVulkan --parallel
else
    echo "SKIPPED: Vulkan SDK not found. Install with:"
    echo "  wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -"
    echo "  sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list"
    echo "  sudo apt update"
    echo "  sudo apt install vulkan-sdk"
fi

# --- Build 3: OpenVINO backend (CPU, GPU, NPU) ---
echo "Checking for OpenVINO..."
if [ -f "/opt/intel/openvino_2026.1.0/setupvars.sh" ] || [ -n "$INTEL_OPENVINO_DIR" ]; then
    echo "Building OpenVINO backend..."
    if [ -f "/opt/intel/openvino_2026.1.0/setupvars.sh" ]; then
        source /opt/intel/openvino_2026.1.0/setupvars.sh
    fi
    cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
    cmake --build build/ReleaseOV --parallel
else
    echo "SKIPPED: OpenVINO not found. Install from:"
    echo "  https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html"
fi

echo ""
echo "Build complete! CPU backend built successfully."
echo "  CPU binary: build/Release/bin/"
[ -d "build/ReleaseVulkan/bin" ] && echo "  Vulkan binary: build/ReleaseVulkan/bin/"
[ -d "build/ReleaseOV/bin" ] && echo "  OpenVINO binary: build/ReleaseOV/bin/"
exit 0