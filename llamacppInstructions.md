
# Cloning and compiling
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build .

# Download a ggml model
# E.g: llama-2-7b-chat.Q4_K_M.gguf from:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf

# Test llm model
./build/bin/llama-cli -m models/llama-2-7b-chat.Q4_K_M.gguf -p "¿Cómo reseteo un motor Siemens X200?"

```
