
#include "speechRecognition.h" 

namespace assistantAsr {

SpeechRecongnition::SpeechRecongnition(const char * whisperModelPath) {
    // In this constructior default parameters are going to be used.
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // ❌ Don't use GPU

    std::cout << "Initializing SpeechRecongnition with Whisper." << std::endl;
    m_whisperContext = whisper_init_from_file_with_params(whisperModelPath, cparams);
    if (!m_whisperContext) {
        std::cerr << "Failed to initialize Whisper." << std::endl;
        throw std::runtime_error("Failed to initialize SpeechRecongnition.");
    }

    std::cout << "Setting Whisper default parameters..." << std::endl;
    m_whisperDefaultParams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    m_whisperDefaultParams.print_progress = false;

}

SpeechRecongnition::~SpeechRecongnition() {
    std::cout << "Terminating SpeechRecongnition." << std::endl;
    if (m_whisperContext) {
        whisper_free(m_whisperContext);
    }
}

void SpeechRecongnition::getAudioDataFromWav(std::string &wavPath, std::vector<float> &audioDataOutput) {
    std::cout << "Loading audio file into memory..." << std::endl;
    std::vector<float> pcmf32;
    if (!assistantUtils::wav_pcmf32_load(wavPath.c_str(), audioDataOutput)) {
        std::cerr << "Failed to load WAV file." << std::endl;
        throw std::runtime_error("Failed to load WAV file.");
    }
}

std::string SpeechRecongnition::generateTranscript(std::vector<float> &audioData) {
    std::cout << "Running Whisper transcription..." << std::endl;
    if (whisper_full(m_whisperContext, m_whisperDefaultParams, audioData.data(), audioData.size()) != 0) {
        std::cerr << "Transcription error." << std::endl;
        throw std::runtime_error("Transcription error.");
    }

    std::cout << "Retrieving transcription result..." << std::endl;
    int n_segments = whisper_full_n_segments(m_whisperContext);
    std::ostringstream oss;
    for (int i = 0; i < n_segments; ++i) {
        oss << whisper_full_get_segment_text(m_whisperContext, i);
    }
    std::string transcription = oss.str();

    std::cout << "\nTranscribed Text:\n" << transcription << std::endl;
    return transcription;
}

} // namespace assistantAsr