
#include "assistantUtils.h"

namespace assistantUtils {

bool wav_pcmf32_load(const char *fname, std::vector<float> & pcmf32) {
    // WAV header
    struct WAVHeader {
        char     riff[4];
        int32_t  overall_size;
        char     wave[4];
        char     fmt_chunk_marker[4];
        int32_t  length_of_fmt;
        int16_t  format_type;
        int16_t  channels;
        int32_t  sample_rate;
        int32_t  byterate;
        int16_t  block_align;
        int16_t  bits_per_sample;
    };

    struct DataChunkHeader {
        char     data_chunk_header[4];
        int32_t  data_size;
    };

    std::ifstream file(fname, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "No se pudo abrir el archivo: %s\n", fname);
        return false;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char *>(&header), sizeof(WAVHeader));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "El archivo no es un WAV válido: %s\n", fname);
        return false;
    }

    if (header.format_type != 1 || (header.bits_per_sample != 16 && header.bits_per_sample != 0)) {
        fprintf(stderr, "Formato no soportado: solo PCM 16-bit.\n");
        return false;
    }

    // Saltar bytes extra del chunk fmt si los hay
    if (header.length_of_fmt > 16) {
        file.seekg(header.length_of_fmt - 16, std::ios::cur);
    }

    // Buscar el chunk "data" (podría no estar justo después del fmt)
    DataChunkHeader data_header;
    while (true) {
        file.read(reinterpret_cast<char *>(&data_header), sizeof(DataChunkHeader));
        if (!file) {
            fprintf(stderr, "No se encontró el chunk de datos en el archivo WAV.\n");
            return false;
        }

        if (std::strncmp(data_header.data_chunk_header, "data", 4) == 0) {
            break;
        }

        // Saltar chunk desconocido
        file.seekg(data_header.data_size, std::ios::cur);
    }

    // Leer los datos PCM
    const int n_samples = data_header.data_size / sizeof(int16_t);
    std::vector<int16_t> pcm16(n_samples);
    file.read(reinterpret_cast<char *>(pcm16.data()), data_header.data_size);

    if (!file) {
        fprintf(stderr, "No se pudieron leer los datos de audio.\n");
        return false;
    }

    pcmf32.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        pcmf32[i] = float(pcm16[i]) / 32768.0f;
    }

    return true;
}

} // namespace assistantUtils