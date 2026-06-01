#ifndef TEXT_PROCESSOR_H_
#define TEXT_PROCESSOR_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Text processor for VITS text-to-speech system
// Handles text cleaning, phoneme conversion, and symbol-to-ID mapping
class TextProcessor {
public:
    TextProcessor();
    ~TextProcessor();

    // Initialize the text processor (load espeak, build symbol mappings)
    bool Init(const char* espeak_data_path = "./model/espeak-ng-data");

    // Clean text using specified cleaners
    std::string CleanText(const std::string& text, const std::vector<std::string>& cleaner_names);

    // Convert text to sequence of symbol IDs
    std::vector<int64_t> TextToSequence(const std::string& text, const std::vector<std::string>& cleaner_names);

    // Intersperse a value between elements of a sequence
    std::vector<int64_t> Intersperse(const std::vector<int64_t>& sequence, int64_t value);

    // Convert sequence back to text (for debugging)
    std::string SequenceToText(const std::vector<int64_t>& sequence);

private:
    bool initialized_;
    std::unordered_map<std::string, int64_t> symbol_to_id_;
    std::unordered_map<int64_t, std::string> id_to_symbol_;
    std::vector<std::string> symbols_;

    // Initialize symbol mappings
    void InitSymbols();

    // Text cleaning functions
    std::string Lowercase(const std::string& text);
    std::string CollapseWhitespace(const std::string& text);
    std::string ConvertToASCII(const std::string& text);
    std::string ExpandAbbreviations(const std::string& text);
    std::string EnglishCleaners2(const std::string& text);

    // Helper function to check if espeak is available
    bool CheckEspeakAvailable();
};

#endif  // TEXT_PROCESSOR_H_