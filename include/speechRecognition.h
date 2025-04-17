#pragma once

#include <stdexcept>

#include "common.h"
#include "whisper.h"
#include "assistantUtils.h"

namespace assistantAsr {

class SpeechRecongnition {
public:
    SpeechRecongnition(const char * whisperModelPath);
    void getAudioDataFromWav(std::string &wavPath, std::vector<float> &audioDataOutput);
    std::string generateTranscript(std::vector<float> &audioData);

    ~SpeechRecongnition();

private:
    whisper_context * m_whisperContext = nullptr;
    whisper_full_params m_whisperDefaultParams;
};

} // namespace assistantAsr