#pragma once

#include <fstream>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <sstream> 
#include <vector>

namespace assistantUtils {

    bool wav_pcmf32_load(const char *fname, std::vector<float> & pcmf32);

} // namespace assistantUtils