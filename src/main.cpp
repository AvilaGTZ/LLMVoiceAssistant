#include "assistantUtils.h"
#include "common.h"
#include "whisper.h"  // Whisper library
#include "llama.h"    // LLaMA library
#include "sampling.h"

using namespace assistantUtils;

int main() {
    std::cout << "Starting Voice Assistant..." << std::endl;

    // 📁 Paths are defined in buildAssistant.sh
    const char * whisper_model_path = WHISPER_MODEL_PATH;
    const char * llama_model_path = LLAMA_MODEL_PATH;
    std::string audio_file_path = AUDIO_FILE_PATH;

    // 1️⃣ Initialize Whisper to convert audio to text
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // ❌ Don't use GPU

    std::cout << "Initializing Whisper..." << std::endl;
    whisper_context *ctx = whisper_init_from_file_with_params(whisper_model_path, cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize Whisper." << std::endl;
        return 1;
    }

    std::cout << "Loading audio file into memory..." << std::endl;
    std::vector<float> pcmf32;
    if (!wav_pcmf32_load(audio_file_path.c_str(), pcmf32)) {
        std::cerr << "Failed to load WAV file." << std::endl;
        return 1;
    }

    std::cout << "Setting Whisper default parameters..." << std::endl;
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;

    std::cout << "Running Whisper transcription..." << std::endl;
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        std::cerr << "Transcription error." << std::endl;
        return 1;
    }

    std::cout << "Retrieving transcription result..." << std::endl;
    int n_segments = whisper_full_n_segments(ctx);
    std::ostringstream oss;
    for (int i = 0; i < n_segments; ++i) {
        oss << whisper_full_get_segment_text(ctx, i);
    }
    std::string transcription = oss.str();

    std::cout << "\nTranscribed Text:\n" << transcription << std::endl;

    // Transform the transcription into something more meaninfull for llama model
    std::string systemPrompt = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n";
    std::string llamaPrompt = systemPrompt + transcription + "\n[/INST]";

    std::cout << std::endl;

    // 2️⃣ Pass the text to LLaMA to generate a response
    std::cout << "\nInitializing LLaMA..." << std::endl;
    
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(llama_model_path, model_params);

    if (model == NULL) {
        std::cout << "Failed to load LLaMA model." << std::endl;
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 🧠 Create context for LLaMA
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = 4;
    ctx_params.n_ctx = 2048;
    llama_context * llama_ctx = llama_init_from_model(model, ctx_params);

    // 📝 Tokenize the prompt
    const bool is_first = llama_kv_self_used_cells(llama_ctx) == 0;

    // Transform the whisper llamaPrompt string into llama_tokens
    const int n_prompt_tokens = -llama_tokenize(vocab, llamaPrompt.c_str(), llamaPrompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, llamaPrompt.c_str(), llamaPrompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        std::cout << "Failed to tokenize the prompt." << std::endl;
    }

    // How many prompts will be evaluated parallel
    // Only a single propmt will be processed. 
    int n_parallel = 1;

    // Initialize the batch with the size of the calculated prompt tokens.
    llama_batch batch = llama_batch_init(std::max(prompt_tokens.size(), (size_t) n_parallel), 0, n_parallel);

    std::vector<llama_seq_id> seq_ids(n_parallel, 0);
    for (int32_t i = 0; i < n_parallel; ++i) {
        seq_ids[i] = i;
    }

    // 🧪 Add all the prompt tokens into the llama batch
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], i, seq_ids, false);
    }

    // Evaluate that the number of tokens in the batch is equal to the calculated prompt tokens.
    GGML_ASSERT(batch.n_tokens == (int) prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(llama_ctx, batch)) {
            std::cout << "Failed to evaluate LLaMA encoder." << std::endl;
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        common_batch_clear(batch);
        common_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
    } else {
        std::cout << "LLaMA model has no encoder..." << std::endl;
    }

    // 🧮 LLaMA decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    // llama_deocde produces logits for all its vocab tokens.
    if (llama_decode(llama_ctx, batch) != 0) {
        std::cout << "Failed to decode with LLaMA." << std::endl;
        return 1;
    }

    // How many tokens are available in the llama model
    const int vocab_size = llama_vocab_n_tokens(vocab);

    // Special token that indicates the end of the Llama answer.
    // Will help during the llama_decode process to know if the 'answer is complete'
    // If Llama generates this token means that 'llama considers that the answer is complete'
    const llama_token eos_token = llama_vocab_eos(vocab);

    // Init at token 0.
    llama_token token = 0;
    // Number of predictions to avoid infinite text generation.
    // This is being used in addition with eos_token
    int n_predict = 150;

    // String to store the llama answer
    std::string llamaResponse;

    std::cout << "\nWhisper transcription:\n" << transcription << std::endl;
    std::cout << "\nPrompt:\n" << llamaPrompt << std::endl;
    std::cout << "\nResponse generated by LLaMA:\n" << std::endl;

    // 🧠 LLaMA Decoding Process Explained:
    // LLaMA generates text one token at a time. For each step:
    // 1. It calculates the 'logits' (scores) for all possible next tokens.
    // 2. It selects the token with the highest score (greedy decoding).
    // 3. It adds this token to the sequence and repeats the process.
    // This loop continues until it generates the desired number of tokens or reaches the end-of-sequence token.
    
    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits(llama_ctx);

        // 🔍 Greedy decoding: pick the token with the highest logit
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

        // Get the string associated to the token and stored in buf
        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        // Prepare batch with the new token
        common_batch_clear(batch);
        // Add the previous token as a context for the new batch
        common_batch_add(batch, token, i + 1, seq_ids, false);
        batch.logits[0] = true;

        // Generate the new set of loggits with the new batch containing the new context.
        if (llama_decode(llama_ctx, batch) != 0) {
            std::cerr << "Error at decoding next token..." << std::endl;
            break;
        }

        std::string s(buf, n);
        llamaResponse += s;
        //printf("%s", s.c_str());
        //fflush(stdout);
    }
    std::cout << llamaResponse << std::endl;

    return 0;
}
