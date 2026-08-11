#pragma once
#include <string>
#include <memory>
#include <cstdint>

struct VocabInfo {
    int vocab_size;
    int special_bos_id[64];
    int special_eos_id[64];
    int n_special_bos_id;
    int n_special_eos_id;
    int linefeed_id;
};

class TokenizerImpl;

class Tokenizer {
public:
    // 从 HF 模型目录加载（读取 tokenizer.json + tokenizer_config.json）
    Tokenizer(const char* model_dir);

    // 从内存加载 tokenizer.json
    Tokenizer(const void* json_data, size_t json_size);

    ~Tokenizer();

    bool GetVocabInfo(VocabInfo* info);

    int Tokenize(const char* text, int32_t text_len,
                 int32_t* tokens, int32_t n_tokens_max);

    std::string TokenToPiece(int32_t token);
    std::string Decode(int32_t* tokens, int32_t n_tokens);

    bool IsLoaded() const;

private:
    std::unique_ptr<TokenizerImpl> impl_;
};

