# Cloning and compiling
```
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
mkdir build
cd build
cmake ..
cmake --build .

# Download a audio to text model to test
bash models/download-ggml-model.sh base.en

# Export the whisper libraries and the ggml libraries
--------------------------------------------------------------------------
e.g : export LD_LIBRARY_PATH=/ruta/a/lib1:/ruta/a/lib2:$LD_LIBRARY_PATH
--------------------------------------------------------------------------
export LD_LIBRARY_PATH=/home/avilagtz/Documents/Development/EdgeAI/AsistenteVozEdgeAI/whisper.cpp/build/src:/home/avilagtz/Documents/Development/EdgeAI/AsistenteVozEdgeAI/whisper.cpp/build/ggml/src:$LD_LIBRARY_PATH

# Execute the test
./build/bin/whisper-cli -m /home/avilagtz/Documents/Development/EdgeAI/AsistenteVozEdgeAI/whisper.cpp/models/ggml-base.bin -f samples/jfk.wav
```
