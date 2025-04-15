cmake_minimum_required(VERSION 3.10)
project(VoiceAssistant)

set(CMAKE_CXX_STANDARD 17)

# Agregar directorios de Whisper y Llama
include_directories(${CMAKE_SOURCE_DIR}/../whisper.cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/../llama.cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/../llama.cpp/common)
include_directories(${CMAKE_SOURCE_DIR}/../llama.cpp/ggml/include)

link_directories(
    ${CMAKE_SOURCE_DIR}/../whisper.cpp/build/src
    ${CMAKE_SOURCE_DIR}/../llama.cpp/build/bin
    ${CMAKE_SOURCE_DIR}/../llama.cpp/build/common
)


# Crear ejecutable
add_executable(asistente src/main.cpp)

# Vincular las bibliotecas compiladas
target_link_libraries(asistente whisper llama common ggml ggml-base common${CMAKE_THREAD_LIBS_INIT})
