#pragma once
#include "llama.h"
#include "Tokenizer.h"
#include "TokenizerBase.h"

class LlamaTokenizer : public TokenizerBase {
public:
    LlamaTokenizer(const char* tokenizer_path);

    LlamaTokenizer(const void * tokenizer_buffer, size_t buffer_size);

    ~LlamaTokenizer() override;

    bool GetVocabInfo(VocabInfo* info) override;

    int Tokenize(const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max) override;

    std::string TokenToPiece(int32_t token) override;

    std::string Decode(int32_t* tokens, int32_t n_tokens) override;

private:
    bool Init(const char* tokenizer_path);
    bool Init(const void * tokenizer_buffer, size_t buffer_size);
    void Deinit();

    llama_vocab* mVocab = NULL;
};