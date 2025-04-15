#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <sstream> 
#include <vector>
#include "common.h"
#include "whisper.h"  // Librería de Whisper
#include "llama.h"    // Librería de LLaMA

bool whisper_pcmf32_load(const char *fname, std::vector<float> & pcmf32) {
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



int main() {
    std::cout << "Iniciando Asistente de Voz..." << std::endl;

    // 1️⃣ Cargar Whisper para convertir audio en texto
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    const char * model_path = "../models/whisper/ggml-large-v2.bin";
    //whisper_model_loader loader = whisper_model_loader_from_file(model_path);

    std::cout << "Inicializando whisper..." << std::endl;
    whisper_context *ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        std::cerr << "Error al inicializar Whisper." << std::endl;
        return 1;
    }

    std::cout << "Convirtiendo archivo de audio a bytes..." << std::endl;
    std::string audio_file = "jfk_clean.wav";
    std::vector<float> pcmf32;
    if (!whisper_pcmf32_load(audio_file.c_str(), pcmf32)) {
        std::cerr << "Error al cargar el archivo WAV." << std::endl;
        return 1;
    }

    std::cout << "Creando whisper default parameters..." << std::endl;
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;

    std::cout << "Invocando whisper_full..." << std::endl;
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        std::cerr << "Error en la transcripción.\n";
        return 1;
    }

    std::cout << "Obteniendo resultado con whisper_full_get_segment_text..." << std::endl;
    int n_segments = whisper_full_n_segments(ctx);
    std::ostringstream oss;
    for (int i = 0; i < n_segments; ++i) {
        oss << whisper_full_get_segment_text(ctx, i);
    }
    std::string transcription = oss.str();

    std::cout << "\nTexto transcrito:\n" << transcription << std::endl;

    std::cout << std::endl;

    // 2️⃣ Pasar el texto a LLaMA para generar respuesta
    std::cout << "\nInicializando LLaMA..." << std::endl;
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("../models/llama/llama-2-7b-chat.Q4_K_M.gguf", model_params);

    if (model == NULL) {
        std::cout << "Error al cargar el modelo de llama..." << std::endl;
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Crear el contexto para llama
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = 4;
    ctx_params.n_ctx = 2048;
    llama_context * llama_ctx = llama_init_from_model(model, ctx_params);

    // tokenize the prompt
    const bool is_first = llama_kv_self_used_cells(llama_ctx) == 0;

    const int n_prompt_tokens = -llama_tokenize(vocab, transcription.c_str(), transcription.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, transcription.c_str(), transcription.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        std::cout << "Error al tokenizar el prompt..." << std::endl;
    }

    int n_parallel = 8;

    llama_batch batch = llama_batch_init(std::max(prompt_tokens.size(), (size_t) n_parallel), 0, n_parallel);

    std::vector<llama_seq_id> seq_ids(n_parallel, 0);
    for (int32_t i = 0; i < n_parallel; ++i) {
        seq_ids[i] = i;
    }

    // evaluate the initial prompt
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], i, seq_ids, false);
    }

    GGML_ASSERT(batch.n_tokens == (int) prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(llama_ctx, batch)) {
            std::cout << "Error al evaluar el encoder de llama..." << std::endl;
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        common_batch_clear(batch);
        common_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(llama_ctx, batch) != 0) {
        std::cout << "Error al tratar de decodificar con llama decode..." << std::endl;
        return 1;
    }


    const int vocab_size = llama_vocab_n_tokens(vocab);
    const llama_token eos_token = llama_vocab_eos(vocab);

    std::string respuesta;
    llama_token token = 0;
    int n_predict = 100;

    std::cout << "\nRespuesta generada por LLaMA:\n" << std::endl;
    
    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits(llama_ctx);

        // Greedy decoding (puedes cambiar por sampling)
        float max_logit = logits[0];
        token = 0;
        for (int j = 1; j < vocab_size; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                token = j;
            }
        }

        if (token == eos_token) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        // Prepara batch con el nuevo token
        common_batch_clear(batch);
        common_batch_add(batch, token, i + 1, seq_ids, false);
        batch.logits[0] = true;

        if (llama_decode(llama_ctx, batch) != 0) {
            std::cerr << "Error al decodificar el siguiente token..." << std::endl;
            break;
        }

        std::string s(buf, n);
        printf("%s", s.c_str());
        fflush(stdout);
    }

    return 0;
}
