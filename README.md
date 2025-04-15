# LLMVoiceAssistant
Edge AI Voice Assistant

A powerful voice assistant that combines Whisper.cpp for speech-to-text and Llama.cpp for text processing, designed to run entirely on the edge without cloud dependencies.

🚀 Features

- Speech-to-text conversion using Whisper.cpp
- Text processing and response generation using Llama.cpp
- Fully offline operation
- Configurable model paths and audio input

🛠️ Prerequisites

- CMake (>= 3.10)
- C++17 compatible compiler
- Git
- Curl

📥 Installation

1. Clone the Repository
```
git clone https://github.com/AvilaGTZ/LLMVoiceAssistant.git
cd LLMVoiceAssistant
```

2. Install Whisper.cpp
```
./getWhisperCpp.sh [model_name]
```
#### Available models:
- tiny.en (default)
- base.en
- small.en
- medium.en
- large-v3

#### The script will:

- Clone and compile Whisper.cpp
- Download the specified model (or base.en by default)
- Place the model in ../models/whisper/
- Configure necessary environment variables

3. Install Llama.cpp
```
./getLlamaCpp.sh [model_url]
```

Default model: `llama-2-7b-chat.Q4_K_M.gguf`

#### The script will:

- Clone and compile Llama.cpp
- Download the specified model (or default Llama 2 7B Chat)
- Place the model in ../models/llama/
- Configure necessary environment variables

Alternative models can be downloaded from `TheBloke's Hugging Face repository`.

4. Build the Assistant
```
./buildAssistant.sh [whisper_model_path] [llama_model_path] [audio_file_path]
```

#### Parameters (all optional):

- whisper_model_path: Path to Whisper model (default: ../models/whisper/ggml-large-v3.bin)
- llama_model_path: Path to Llama model (default: ../models/llama/llama-2-7b-chat.Q4_K_M.gguf)
- audio_file_path: Path to input audio file (default: ../audioSamples/jfk_clean.wav)

🏃‍♂️ Running the Assistant
```
cd build
./assistant
```

#### The assistant will:

- Load the specified Whisper model
- Convert the input audio to text
- Process the text using the Llama model
- Generate and display the response

📁 Project Structure
```
.
├── models/
│   ├── whisper/
│   └── llama/
├── audioSamples/
├── src/
│   └── main.cpp
├── getWhisperCpp.sh
├── getLlamaCpp.sh
├── buildAssistant.sh
└── CMakeLists.txt
```
### Using Different Models
```
# Install smaller Whisper model
./getWhisperCpp.sh base.en

# Install different Llama model
./getLlamaCpp.sh "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf"

# Build with custom paths
./buildAssistant.sh "../models/whisper/ggml-base.en.bin" "../models/llama/llama-2-13b-chat.Q4_K_M.gguf" "../audioSamples/custom_audio.wav"
```

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

- Whisper.cpp
- Llama.cpp
- OpenAI Whisper
- Llama 2

