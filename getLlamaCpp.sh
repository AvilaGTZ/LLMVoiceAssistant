#!/bin/bash

# Define colors for messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model URL
DEFAULT_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

# Get the model URL from command line argument or use default
MODEL_URL=${1:-$DEFAULT_URL}

# Extract model name from URL (everything after the last /)
MODEL_NAME=$(basename "$MODEL_URL")

# Base directory where llama.cpp will be installed
BASE_DIR="$PWD"

echo -e "${BLUE}Starting llama.cpp installation...${NC}"

# Clone repository
echo -e "${GREEN}Cloning llama.cpp repository...${NC}"
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Create and enter build directory
echo -e "${GREEN}Setting up build...${NC}"
mkdir -p build
cd build

# Compile with CMAKE
echo -e "${GREEN}Compiling with CMAKE...${NC}"
cmake ..
cmake --build .

# Return to base directory
cd ../..

# Create models directory and llama subdirectory
echo -e "${GREEN}Creating models directory structure...${NC}"
mkdir -p models/llama

# Download the model
echo -e "${GREEN}Downloading model: ${MODEL_NAME}...${NC}"
echo -e "${GREEN}This might take a while depending on your internet connection...${NC}"
curl -L "$MODEL_URL" -o "models/llama/$MODEL_NAME"

# Configure environment variables
echo -e "${GREEN}Setting up environment variables...${NC}"
LLAMA_LIB_PATH="$BASE_DIR/llama.cpp/build"

# Add paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LLAMA_LIB_PATH:$LD_LIBRARY_PATH"

# Add configuration to .bashrc to make it permanent
echo -e "${GREEN}Adding configuration to .bashrc...${NC}"
echo "export LD_LIBRARY_PATH=$LLAMA_LIB_PATH:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo -e "${BLUE}llama.cpp installation completed!${NC}"
echo -e "${GREEN}Libraries have been installed in:${NC}"
echo "llama.cpp: $LLAMA_LIB_PATH"
echo -e "${GREEN}Model has been installed in:${NC}"
echo "models/llama/$MODEL_NAME"