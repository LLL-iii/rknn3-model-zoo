#pragma once
#include <string>
#include <vector>

#include "Tokenizer.h"

class TokenizerBase {
public:
    TokenizerBase() = default;

    virtual ~TokenizerBase() = default;

    virtual bool GetVocabInfo(VocabInfo* info) = 0;

    virtual int Tokenize(const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max) = 0;

    virtual std::string TokenToPiece(int32_t token) = 0;

    virtual std::string Decode(int32_t* tokens, int32_t n_tokens) = 0;
};