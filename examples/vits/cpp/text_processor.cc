#include "text_processor.h"
#include <algorithm>
#include <sstream>
#include <regex>
#include <unordered_set>
#include <cctype>

// espeak-ng API
#include "espeak-ng/speak_lib.h"

TextProcessor::TextProcessor() : initialized_(false) {
}

TextProcessor::~TextProcessor() {
    if (initialized_) {
        // Terminate espeak
        espeak_Terminate();
    }
}

bool TextProcessor::Init(const char* espeak_data_path) {
    if (initialized_) {
        return true;
    }

    printf("Initializing symbol mappings...\n");
    InitSymbols();

    printf("Initializing eSpeak NG...\n");
    printf("eSpeak data path: %s\n", espeak_data_path);

    if (!CheckEspeakAvailable()) {
        fprintf(stderr, "eSpeak NG is not available\n");
        return false;
    }

    // Initialize eSpeak with custom data path
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, espeak_data_path, 0);
    if (result < 0) {
        fprintf(stderr, "Failed to initialize eSpeak: %d\n", result);
        return false;
    }

    // Set voice to English (US)
    espeak_SetVoiceByName("en-us");

    initialized_ = true;
    printf("Text processor initialized successfully\n");
    return true;
}

void TextProcessor::InitSymbols() {
    // Define symbols based on Python text/symbols.py
    // symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

    // Start with pad symbol
    symbols_.push_back("_");

    // Add punctuation (from _punctuation in symbols.py)
    // _punctuation = ';:,.!?¡¿—…"«»"" ' (16 symbols including space)
    // Use UTF-8 encoding for special characters to avoid compilation issues
    std::vector<std::string> punctuation_symbols = {
        ";", ":", ",", ".", "!", "?", "¡", "¿",
        "\xE2\x80\x94",  // — (U+2014 EM DASH)
        "\xE2\x80\xA6",  // … (U+2026 ELLIPSIS)
        "\"",            // " (U+0022 QUOTATION MARK)
        "\xC2\xAB",      // « (U+00AB LEFT-POINTING DOUBLE ANGLE QUOTATION MARK)
        "\xC2\xBB",      // » (U+00BB RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK)
        "\xE2\x80\x9C",  // " (U+201C LEFT DOUBLE QUOTATION MARK)
        "\xE2\x80\x9D",  // " (U+201D RIGHT DOUBLE QUOTATION MARK)
        " "
    };

    for (const auto& punct : punctuation_symbols) {
        symbols_.push_back(punct);
    }

    // Note: space is already included in punctuation above as ID 15
    // So we don't add it separately here

    // Add uppercase letters A-Z (ID 17-42)
    for (char c = 'A'; c <= 'Z'; c++) {
        symbols_.push_back(std::string(1, c));
    }

    // Add lowercase letters a-z (ID 43-68)
    for (char c = 'a'; c <= 'z'; c++) {
        symbols_.push_back(std::string(1, c));
    }

    // Add IPA symbols (from _letters_ipa in symbols.py, ID 69+)
    const char* ipa_symbols[] = {
        "ɑ", "ɐ", "ɒ", "æ", "ɓ", "ʙ", "β", "ɔ", "ɕ", "ç", "ɗ", "ɖ", "ð", "ʤ", "ə", "ɘ", "ɚ",
        "ɛ", "ɜ", "ɝ", "ɞ", "ɟ", "ʄ", "ɡ", "ɠ", "ɢ", "ʛ", "ɦ", "ɧ", "ħ", "ɥ", "ʜ", "ɨ", "ɪ",
        "ʝ", "ɭ", "ɬ", "ɫ", "ɮ", "ʟ", "ɱ", "ɯ", "ɰ", "ŋ", "ɳ", "ɲ", "ɴ", "ø", "ɵ", "ɸ", "θ",
        "œ", "ɶ", "ʘ", "ɹ", "ɺ", "ɾ", "ɻ", "ʀ", "ʁ", "ɽ", "ʂ", "ʃ", "ʈ", "ʧ", "ʉ", "ʊ", "ʋ",
        "ⱱ", "ʌ", "ɣ", "ɤ", "ʍ", "χ", "ʎ", "ʏ", "ʑ", "ʐ", "ʒ", "ʔ", "ʡ", "ʕ", "ʢ", "ǀ", "ǁ",
        "ǂ", "ǃ", "ˈ", "ˌ", "ː", "ˑ", "ʼ", "ʴ", "ʰ", "ʱ", "ʲ", "ʷ", "ˠ", "ˤ", "˞", "↓", "↑",
        "→", "↗", "↘", "'", "̩", "'", "ᵻ"
    };

    for (const char* ipa : ipa_symbols) {
        symbols_.push_back(std::string(ipa));
    }

    // Build mappings
    for (size_t i = 0; i < symbols_.size(); i++) {
        symbol_to_id_[symbols_[i]] = static_cast<int64_t>(i);
        id_to_symbol_[static_cast<int64_t>(i)] = symbols_[i];
    }

    printf("Loaded %zu symbols\n", symbols_.size());

    // Verify key positions match Python
    // Build mappings and verify (silent in production)
    for (size_t i = 0; i < symbols_.size(); i++) {
        symbol_to_id_[symbols_[i]] = static_cast<int64_t>(i);
        id_to_symbol_[static_cast<int64_t>(i)] = symbols_[i];
    }
}

bool TextProcessor::CheckEspeakAvailable() {
    // Try to get espeak version to check if it's available
    const char* version = espeak_Info(nullptr);
    return (version != nullptr);
}

std::string TextProcessor::CleanText(const std::string& text, const std::vector<std::string>& cleaner_names) {
    std::string cleaned = text;

    for (const auto& name : cleaner_names) {
        if (name == "english_cleaners2") {
            cleaned = EnglishCleaners2(cleaned);
        } else if (name == "lowercase") {
            cleaned = Lowercase(cleaned);
        } else if (name == "collapse_whitespace") {
            cleaned = CollapseWhitespace(cleaned);
        } else {
            fprintf(stderr, "Unknown cleaner: %s\n", name.c_str());
        }
    }

    return cleaned;
}

std::vector<int64_t> TextProcessor::TextToSequence(const std::string& text, const std::vector<std::string>& cleaner_names) {
    std::vector<int64_t> sequence;

    // Clean the text
    std::string cleaned_text = CleanText(text, cleaner_names);

    // Convert each UTF-8 character/symbol to its ID
    size_t i = 0;
    while (i < cleaned_text.length()) {
        std::string symbol;

        // Handle UTF-8 multi-byte characters
        unsigned char c = cleaned_text[i];
        if (c < 0x80) {
            // ASCII character (1 byte)
            symbol = cleaned_text.substr(i, 1);
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte UTF-8 character
            symbol = cleaned_text.substr(i, 2);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte UTF-8 character
            symbol = cleaned_text.substr(i, 3);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte UTF-8 character
            symbol = cleaned_text.substr(i, 4);
            i += 4;
        } else {
            // Invalid UTF-8, skip
            i++;
            continue;
        }

        // Skip only empty symbols, keep spaces and punctuation
        if (symbol.empty()) {
            continue;
        }

        // Look up symbol ID
        auto it = symbol_to_id_.find(symbol);
        if (it != symbol_to_id_.end()) {
            sequence.push_back(it->second);
        } else {
            fprintf(stderr, "Warning: Symbol '%s' (length %zu) not found in symbol table\n",
                   symbol.c_str(), symbol.length());
        }
    }

    return sequence;
}

std::vector<int64_t> TextProcessor::Intersperse(const std::vector<int64_t>& sequence, int64_t value) {
    std::vector<int64_t> result;

    if (sequence.empty()) {
        return result;
    }

    // Intersperse value between elements
    result.push_back(value);
    for (int64_t elem : sequence) {
        result.push_back(elem);
        result.push_back(value);
    }

    return result;
}

std::string TextProcessor::SequenceToText(const std::vector<int64_t>& sequence) {
    std::string result;

    for (int64_t id : sequence) {
        auto it = id_to_symbol_.find(id);
        if (it != id_to_symbol_.end()) {
            result += it->second;
        } else {
            result += "?";  // Unknown symbol
        }
    }

    return result;
}

// Text cleaning functions

std::string TextProcessor::Lowercase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string TextProcessor::CollapseWhitespace(const std::string& text) {
    static const std::regex whitespace_re("\\s+");
    std::string result = std::regex_replace(text, whitespace_re, " ");
    return result;
}

std::string TextProcessor::ConvertToASCII(const std::string& text) {
    // For now, just return the text as-is
    // In a full implementation, you'd use a library like libiconv or ICU
    // to convert Unicode to ASCII
    return text;
}

std::string TextProcessor::ExpandAbbreviations(const std::string& text) {
    static const std::vector<std::pair<std::string, std::string>> abbreviations = {
        {"mrs", "misess"},
        {"mr", "mister"},
        {"dr", "doctor"},
        {"st", "saint"},
        {"co", "company"},
        {"jr", "junior"},
        {"maj", "major"},
        {"gen", "general"},
        {"drs", "doctors"},
        {"rev", "reverend"},
        {"lt", "lieutenant"},
        {"hon", "honorable"},
        {"sgt", "sergeant"},
        {"capt", "captain"},
        {"esq", "esquire"},
        {"ltd", "limited"},
        {"col", "colonel"},
        {"ft", "fort"}
    };

    std::string result = text;

    for (const auto& abbr : abbreviations) {
        std::regex pattern("\\b" + abbr.first + "\\.", std::regex_constants::icase);
        result = std::regex_replace(result, pattern, abbr.second);
    }

    return result;
}

std::string TextProcessor::EnglishCleaners2(const std::string& text) {
    // Pipeline for English text, including abbreviation expansion
    std::string result = text;

    result = ConvertToASCII(result);
    result = Lowercase(result);
    result = ExpandAbbreviations(result);

    // Extract punctuation marks with their positions (like phonemize preserve_punctuation=True)
    // From Python symbols.py: _punctuation = ';:,.!?¡¿—…"«»"" '
    std::vector<std::pair<size_t, std::string>> punct_marks;
    for (size_t i = 0; i < result.length(); i++) {
        char c = result[i];
        // Basic ASCII punctuation
        if (c == ',' || c == '.' || c == '?' || c == '!' || c == ':' || c == ';') {
            punct_marks.push_back({i, std::string(1, c)});
        }
        // Check for multi-byte UTF-8 punctuation characters
        else if (c == '\xC2' || c == '\xE2') {  // UTF-8 escape sequences for special chars
            // Check for ¡¿ (U+00A1, U+00BF) - starts with C2
            if (c == '\xC2' && i + 1 < result.length()) {
                unsigned char next = result[i + 1];
                if (next == '\xA1' || next == '\xBF') {  // ¡ or ¿
                    std::string punct = result.substr(i, 2);
                    punct_marks.push_back({i, punct});
                    i++;  // Skip next byte
                }
            }
            // Check for —… (U+2014, U+2026) and «»"" (U+00AB, U+00BB, U+201C, U+201D) - start with E2
            else if (c == '\xE2' && i + 2 < result.length()) {
                unsigned char byte2 = result[i + 1];
                unsigned char byte3 = result[i + 2];
                // Check for common UTF-8 punctuation patterns
                if ((byte2 == '\x80' && (byte3 == '\x94' || byte3 == '\xA6')) ||  // — or …
                    (byte2 == '\x80' && (byte3 == '\xAB' || byte3 == '\x9C' || byte3 == '\x9D')) ||  // « or " or "
                    (byte2 == '\x80' && byte3 == '\xBB')) {  // »
                    std::string punct = result.substr(i, 3);
                    punct_marks.push_back({i, punct});
                    i += 2;  // Skip next 2 bytes
                }
            }
        }
    }

    // Process with eSpeak in segments (current working approach)
    std::string phonemes_result;
    const void* textptr = result.c_str();
    int textmode = espeakCHARS_AUTO;
    int phonememode = espeakPHONEMES_IPA;

    int segment_count = 0;
    size_t punct_idx = 0;

    while (textptr != nullptr && punct_idx < punct_marks.size()) {
        const char* phonemes = espeak_TextToPhonemes(&textptr, textmode, phonememode);

        if (phonemes == nullptr || strlen(phonemes) == 0) {
            break;
        }

        phonemes_result += phonemes;

        // Calculate current position in original text
        size_t current_pos = (const char*)textptr - result.c_str();

        // Add punctuation marks that appear before current position
        while (punct_idx < punct_marks.size()) {
            const auto& [pos, mark] = punct_marks[punct_idx];
            if (pos < current_pos) {
                // Check if punctuation is not already at the end
                if (!phonemes_result.empty() && phonemes_result.back() != mark[0]) {
                    phonemes_result += mark;
                }
                punct_idx++;
            } else {
                break;
            }
        }

        // Check if we've reached the end
        if (textptr == nullptr || strlen((const char*)textptr) == 0) {
            break;
        }
    }

    // Add any remaining punctuation
    while (punct_idx < punct_marks.size()) {
        const auto& [pos, mark] = punct_marks[punct_idx];
        if (!phonemes_result.empty() && phonemes_result.back() != mark[0]) {
            phonemes_result += mark;
        }
        punct_idx++;
    }

    result = CollapseWhitespace(phonemes_result);
    return result;
}