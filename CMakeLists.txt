cmake_minimum_required(VERSION 3.10)
project(VoiceAssistant)

set(CMAKE_CXX_STANDARD 17)

# Get environment variables and set them as compile definitions
set(WHISPER_MODEL_PATH $ENV{WHISPER_MODEL_PATH})
set(LLAMA_MODEL_PATH $ENV{LLAMA_MODEL_PATH})
set(AUDIO_FILE_PATH $ENV{AUDIO_FILE_PATH})

add_compile_definitions(
    WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH}"
    LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH}"
    AUDIO_FILE_PATH="${AUDIO_FILE_PATH}"
)

# Add Whisper and Llama directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/whisper.cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/llama.cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/llama.cpp/common)
include_directories(${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include)

link_directories(
    ${CMAKE_SOURCE_DIR}/whisper.cpp/build/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/build/bin
    ${CMAKE_SOURCE_DIR}/llama.cpp/build/common
)

# Create executable
add_executable(assistant src/main.cpp
                         src/assistantUtils.cpp)

# Link libraries
target_link_libraries(asistente whisper llama common ggml ggml-base common${CMAKE_THREAD_LIBS_INIT})
