#!/bin/bash

# Define colors for messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the model name from command line argument or use default
# Recommended to use 'large-v3'
MODEL_NAME=${1:-base.en}

# Base directory where whisper.cpp will be installed
BASE_DIR="$PWD"

echo -e "${BLUE}Starting whisper.cpp installation...${NC}"

# Clone repository
echo -e "${GREEN}Cloning whisper.cpp repository...${NC}"
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

# Create and enter build directory
echo -e "${GREEN}Setting up build...${NC}"
mkdir -p build
cd build

# Compile with CMAKE
echo -e "${GREEN}Compiling with CMAKE...${NC}"
cmake ..
cmake --build .

# Return to whisper.cpp main directory
cd ..

# Download the specified model
echo -e "${GREEN}Downloading model: ${MODEL_NAME}...${NC}"
bash models/download-ggml-model.sh ${MODEL_NAME}

# Create models directory one level up and whisper subdirectory
echo -e "${GREEN}Creating models directory structure...${NC}"
mkdir -p ../models/whisper

# Move the downloaded model to the new location
echo -e "${GREEN}Moving model to ../models/whisper/...${NC}"
mv models/ggml-${MODEL_NAME}.bin ../models/whisper/

# Configure environment variables
echo -e "${GREEN}Setting up environment variables...${NC}"
WHISPER_LIB_PATH="$BASE_DIR/whisper.cpp/build/src"
GGML_LIB_PATH="$BASE_DIR/whisper.cpp/build/ggml/src"

# Add paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$WHISPER_LIB_PATH:$GGML_LIB_PATH:$LD_LIBRARY_PATH"

# Add configuration to .bashrc to make it permanent
echo -e "${GREEN}Adding configuration to .bashrc...${NC}"
echo "export LD_LIBRARY_PATH=$WHISPER_LIB_PATH:$GGML_LIB_PATH:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo -e "${BLUE}whisper.cpp installation completed!${NC}"
echo -e "${GREEN}Libraries have been installed in:${NC}"
echo "whisper.cpp: $WHISPER_LIB_PATH"
echo "ggml: $GGML_LIB_PATH"
echo -e "${GREEN}Model has been installed in:${NC}"
echo "../models/whisper/ggml-${MODEL_NAME}.bin"