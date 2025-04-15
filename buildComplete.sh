#!/bin/bash

# Define colors for messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Edge AI Assistant Build Script${NC}"
echo -e "${BLUE}=================================${NC}\n"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "\n${RED}✗ Error: $1 failed${NC}"
        exit 1
    fi
}

# Get command line arguments
WHISPER_MODEL=${1:-"large-v3"}
LLAMA_MODEL_URL=${2:-"https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"}
AUDIO_FILE=${3:-"../audioSamples/raccoons.wav"}

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo -e "Whisper Model: ${WHISPER_MODEL}"
echo -e "Llama Model URL: ${LLAMA_MODEL_URL}"
echo -e "Audio File: ${AUDIO_FILE}\n"

# 1. Install Whisper.cpp
print_section "Installing Whisper.cpp"
./getWhisperCpp.sh "$WHISPER_MODEL"
check_status "Whisper.cpp installation"

# 2. Install Llama.cpp
print_section "Installing Llama.cpp"
./getLlamaCpp.sh "$LLAMA_MODEL_URL"
check_status "Llama.cpp installation"

# Get the model paths based on the installations
WHISPER_MODEL_PATH="../models/whisper/ggml-${WHISPER_MODEL}.bin"
LLAMA_MODEL_PATH="../models/llama/$(basename ${LLAMA_MODEL_URL})"

# 3. Build the assistant
print_section "Building Assistant"
./buildAssistant.sh "$WHISPER_MODEL_PATH" "$LLAMA_MODEL_PATH" "$AUDIO_FILE"
check_status "Assistant build"

# Print completion message
echo -e "\n${GREEN}=================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}=================================${NC}"
echo -e "\nTo run the assistant:"
echo -e "  cd build"
echo -e "  ./assistant\n"
