#include "Tokenizer.h"
#include "LlamaTokenizer.h"
#include "Log.h"

#define LOG_TAG "Tokenizer"

Tokenizer::Tokenizer(TokenizerBackendType type, const char* tokenizer_path) {
    switch (type)
    {
    case TOKENIZER_BACKEND_LLAMA:
        tokenizerBase = new LlamaTokenizer(tokenizer_path);
        backendType = type;
        break;

    default:
        LOG_E("Error: Invalid TokenizerBackendType=%d.\n", type);
        break;
    }
}

Tokenizer::Tokenizer(TokenizerBackendType type, const void * tokenizer_buffer, size_t buffer_size) {
    switch (type)
    {
    case TOKENIZER_BACKEND_LLAMA:
        tokenizerBase = new LlamaTokenizer(tokenizer_buffer, buffer_size);
        backendType = type;
        break;

    default:
        LOG_E("Error: Invalid TokenizerBackendType=%d.\n", type);
        break;
    }
}

Tokenizer::~Tokenizer() {
    if (tokenizerBase) {
        switch (backendType)
        {
        case TOKENIZER_BACKEND_LLAMA:
            delete (LlamaTokenizer*)tokenizerBase;
            break;

        default:
            break;
        }
    }
}

bool Tokenizer::GetVocabInfo(VocabInfo* info) {
    if (!tokenizerBase) {
        LOG_E("Error: TokenizerBase is not initailized.\n");
        return false;
    }
    return tokenizerBase->GetVocabInfo(info);
}

int Tokenizer::Tokenize(const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max) {
    if (!tokenizerBase) {
        LOG_E("Error: TokenizerBase is not initailized.\n");
        return -1;
    }
    return tokenizerBase->Tokenize(text, text_len, tokens, n_tokens_max);
}

std::string Tokenizer::TokenToPiece(int32_t token) {
    if (!tokenizerBase) {
        LOG_E("Error: TokenizerBase is not initailized.\n");
        return "";
    }
    return tokenizerBase->TokenToPiece(token);
}

std::string Tokenizer::Decode(int32_t* tokens, int32_t n_tokens) {
    if (!tokenizerBase) {
        LOG_E("Error: TokenizerBase is not initailized.\n");
        return "";
    }
    return tokenizerBase->Decode(tokens, n_tokens);
}
