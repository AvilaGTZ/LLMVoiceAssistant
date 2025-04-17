#pragma once

#include <stdexcept>

#include "assistantUtils.h"
#include "common.h"
#include "llama.h"

namespace assistantLlm {

class LlmInference {
public:
    LlmInference(const char * llamaModelPath);
    std::string convertToLlamaPrompt(std::string &normalPrompt);
    void setSystemPrompt(std::string &newSystemPrompt);
    std::vector<llama_token> tokenizePrompt(std::string &prompt);
    std::string generateInference_SAMPLER(std::string &prompt);
    //std::string generateInference_GREEDY(std::string &prompt);

    ~LlmInference();

private:
    llama_model * m_llamaModel = nullptr;
    llama_context * m_llamaContext = nullptr;
    llama_sampler * m_llamaSamplerConfig = nullptr;
    const llama_vocab * m_llamaVocabulary = nullptr;

    llama_sampler * m_llamaSampler = nullptr;
    llama_context_params m_llamaContextParams;

    std::string m_systemPrompt = 
        "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n";
};

} // namespace assistantLlm