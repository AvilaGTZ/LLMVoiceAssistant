#!/bin/bash

# Define colors for messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
DEFAULT_WHISPER_MODEL="../models/whisper/ggml-large-v3.bin"
DEFAULT_LLAMA_MODEL="../models/llama/llama-2-7b-chat.Q4_K_M.gguf"
DEFAULT_AUDIO_FILE="../audioSamples/raccoons.wav"

# Get paths from command line arguments or use defaults
export WHISPER_MODEL_PATH=${1:-$DEFAULT_WHISPER_MODEL}
export LLAMA_MODEL_PATH=${2:-$DEFAULT_LLAMA_MODEL}
export AUDIO_FILE_PATH=${3:-$DEFAULT_AUDIO_FILE}

# Create build directory and compile
echo -e "${BLUE}Building project...${NC}"
mkdir -p build
cd build
cmake ..
cmake --build .

echo -e "${BLUE}Build completed!${NC}"
echo -e "${GREEN}Configuration used:${NC}"
echo "Whisper Model: $WHISPER_MODEL_PATH"
echo "Llama Model: $LLAMA_MODEL_PATH"
echo "Audio File: $AUDIO_FILE_PATH"