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

    // Get the model vocabulary
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 🧠 Create context for LLaMA
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = 4;
    ctx_params.n_ctx = 2048;
    llama_context * llama_ctx = llama_init_from_model(model, ctx_params);

    // initialize the sampler configurations
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

    /**
     * We are going to use SAMPLER inference
     * llama_sampler_init_temp -> Controls random responses: 0 = deterministic, >1 very random
     * llama_sampler_init_top_p -> Nucleus sampling: Maintains the tokens with accum prob <= defined number
     * llama_sampler_init_top_k -> Limit the tokens to the defined number e.g 40
     * llama_sampler_init_min_p -> Filter tokens with probablity less than the number defined e.g 0.05
     * llama_sampler_init_dist -> Init the generator with seed
     * 
     * Do not mix top_p and min_p
     * Do not mix top_k and min_p
    */


    // Llama example config
    /*llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));*/

    // Assistant recommended
    /*llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));*/

    // Radical and no so friendly
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 📝 Tokenize the prompt
    const bool is_first = llama_kv_self_used_cells(llama_ctx) == 0;

    // Transform the whisper llamaPrompt string into llama_tokens
    const int n_prompt_tokens = -llama_tokenize(vocab, llamaPrompt.c_str(), llamaPrompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, llamaPrompt.c_str(), llamaPrompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        std::cout << "Failed to tokenize the prompt." << std::endl;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;

    std::cout << "\nWhisper transcription:\n" << transcription << std::endl;
    std::cout << "\nPrompt:\n" << llamaPrompt << std::endl;
    std::cout << "\nResponse generated by LLaMA:\n" << std::endl;

    std::string llamaResponse;

    while (true) {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(llama_ctx);
        int n_ctx_used = llama_kv_self_used_cells(llama_ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            printf("\033[0m\n");
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        if (llama_decode(llama_ctx, batch)) {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        new_token_id = llama_sampler_sample(smpl, llama_ctx, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            GGML_ABORT("failed to convert token to piece\n");
        }
        std::string piece(buf, n);
        // Uncomment to get the 'llm real time result...' 
        //printf("%s", piece.c_str());
        //fflush(stdout);
        llamaResponse += piece;

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    std::cout << llamaResponse << std::endl;

    return 0;
}
