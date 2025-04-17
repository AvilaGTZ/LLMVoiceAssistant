#include "llmInference.h"

namespace assistantLlm {

LlmInference::LlmInference(const char * llamaModelPath) {
    // Init llama with default parameters
    std::cout << "\nInitializing LLaMA..." << std::endl;
    
    // Init llama model
    llama_model_params modelParams = llama_model_default_params();
    m_llamaModel = llama_model_load_from_file(llamaModelPath, modelParams);

    if (m_llamaModel == NULL) {
        std::cout << "Failed to load LLaMA model." << std::endl;
        throw std::runtime_error("Failed to load LLaMA model.");
    }

    // Init llama vocabulary
    m_llamaVocabulary = llama_model_get_vocab(m_llamaModel);

    // Init default context params
    std::cout << "Setting default context paramters." << std::endl;
    m_llamaContextParams = llama_context_default_params();
    m_llamaContextParams.n_threads = 4;
    m_llamaContextParams.n_ctx = 2048;

    // Init llama context
    m_llamaContext = llama_init_from_model(m_llamaModel, m_llamaContextParams);

    // initialize the sampler configurations
    m_llamaSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    /**
     * We are going to use SAMPLER inference
     * llama_sampler_init_temp -> Controls random responses: 0 = deterministic, >1 very random
     * llama_sampler_init_top_p -> Nucleus sampling: Maintains the tokens with accum prob <= defined number
     * llama_sampler_init_top_k -> Limit the tokens to the defined number e.g 40
     * llama_sampler_init_min_p -> Filter tokens with probablity less than the number defined e.g 0.05. Razonable probbaility.
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
    llama_sampler_chain_add(m_llamaSampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_llamaSampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_llamaSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

void LlmInference::setSystemPrompt(std::string &newSystemPrompt) {
    std::cout << "\nChanging system prompt. This will modify model inference behavior" << std::endl;
    m_systemPrompt = "[INST] <<SYS>>\n" + newSystemPrompt + "\n<</SYS>>\n\n";
}

std::string LlmInference::convertToLlamaPrompt(std::string &normalPrompt) {
    // Transform the transcription into something more meaninfull for llama model
    std::cout << "\nTransforming prompt into something more meaninfull for llama model." << std::endl;
    std::string llamaPrompt = m_systemPrompt + normalPrompt + "\n[/INST]";
    return llamaPrompt;
}

std::vector<llama_token> LlmInference::tokenizePrompt(std::string &prompt) {
    std::cout << "Converting text prompt into llama tokens." << std::endl;

    // 📝 Tokenize the prompt
    const bool is_first = llama_kv_self_used_cells(m_llamaContext) == 0;

    // Transform the whisper llamaPrompt string into llama_tokens
    const int n_prompt_tokens = -llama_tokenize(m_llamaVocabulary, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> promptTokens(n_prompt_tokens);
    if (llama_tokenize(m_llamaVocabulary, prompt.c_str(), prompt.size(), promptTokens.data(), promptTokens.size(), is_first, true) < 0) {
        std::cout << "Failed to tokenize the prompt." << std::endl;
        throw std::runtime_error("Failed to tokenize the prompt.");
    }
    return promptTokens;
}

std::string LlmInference::generateInference_SAMPLER(std::string &prompt) {
    std::cout << "Generating llm inference with prompt:" << std::endl;
    std::cout << prompt << std::endl;

    std::vector<llama_token> tokenizedPrompt = tokenizePrompt(prompt);

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(tokenizedPrompt.data(), tokenizedPrompt.size());
    llama_token new_token_id;
    std::string llamaResponse;
    while (true) {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(m_llamaContext);
        int n_ctx_used = llama_kv_self_used_cells(m_llamaContext);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            printf("\033[0m\n");
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        if (llama_decode(m_llamaContext, batch)) {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        new_token_id = llama_sampler_sample(m_llamaSampler, m_llamaContext, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(m_llamaVocabulary, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(m_llamaVocabulary, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            GGML_ABORT("failed to convert token to piece\n");
        }
        std::string piece(buf, n);
        // Uncomment to get the 'llm real time result...' 
        //printf("%s", piece.c_str());
        printf("%s", ".");
        fflush(stdout);
        llamaResponse += piece;

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    return llamaResponse;
}

LlmInference::~LlmInference () {
    std::cout << "Terminating assistant llm process." << std::endl;
    if (m_llamaModel) {
        llama_model_free(m_llamaModel);
        m_llamaModel = nullptr;
    }

    if (m_llamaContext) {
        llama_free(m_llamaContext);
        m_llamaContext = nullptr;
    }

    if (m_llamaSamplerConfig) {
        llama_sampler_free(m_llamaSamplerConfig);
        m_llamaSamplerConfig = nullptr;
    }

    if (m_llamaSampler) {
        llama_sampler_free(m_llamaSampler);
        m_llamaSampler = nullptr;
    }    
    
}

} // namespace assistantLlm