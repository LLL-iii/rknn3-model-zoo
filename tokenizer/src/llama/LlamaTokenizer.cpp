#include "common.h"
#include "LlamaTokenizer.h"
#include "Log.h"

#define LOG_TAG "LlamaTokenizer"

LlamaTokenizer::LlamaTokenizer(const char* tokenizer_path) {
    Init(tokenizer_path);
}

LlamaTokenizer::LlamaTokenizer(const void * tokenizer_buffer, size_t buffer_size) {
    Init(tokenizer_buffer, buffer_size);
}

LlamaTokenizer::~LlamaTokenizer() {
    Deinit();
}

bool LlamaTokenizer::Init(const char* tokenizer_path) {
    llama_backend_init();

    mVocab = llama_load_vocab_from_file(tokenizer_path, true, false, NULL);
    if (!mVocab) {
        LOG_E("Error: could not load the vocab from file: %s", tokenizer_path);
        return false;
    }
    return true;
}

bool LlamaTokenizer::Init(const void * tokenizer_buffer, size_t buffer_size) {
    if (!tokenizer_buffer) {
        LOG_E("Error: invalid model_file=%p buffer_size=%zu", tokenizer_buffer, buffer_size);
        return false;
    }

    llama_backend_init();
    mVocab = llama_load_vocab_from_buffer(tokenizer_buffer, buffer_size, true, false, NULL);
    if (!mVocab) {
        LOG_E("Error: could not load the vocab from tokenizer_buffer=%p buffer_size=%zu", tokenizer_buffer, buffer_size);
        return false;
    }
    return true;
}

void LlamaTokenizer::Deinit() {
    if (mVocab) {
        llama_vocab_free(mVocab);
        mVocab = nullptr;
    }
    llama_backend_free();
}

bool LlamaTokenizer::GetVocabInfo(VocabInfo* vocab_info) {
    if (!mVocab) {
        LOG_E("Error: TokenizerBase is not initailized.\n");
        return false;
    }
    vocab_info->vocab_size = llama_vocab_size(mVocab);

    vocab_info->linefeed_id = llama_vocab_nl(mVocab);

    vocab_info->n_special_bos_id = 1;
    vocab_info->special_bos_id[0] = llama_vocab_bos(mVocab);

    llama_token special_eos[] = {
        llama_vocab_eos(mVocab),
        llama_vocab_eot(mVocab),
        llama_vocab_eom(mVocab)
    };

    vocab_info->n_special_eos_id = 0;
    for (int i = 0; i < 3; ++i) {
        if (special_eos[i] != LLAMA_TOKEN_NULL) {
            vocab_info->special_eos_id[vocab_info->n_special_eos_id] = special_eos[i];
            ++(vocab_info->n_special_eos_id);
        }
    }

    return true;
}

int LlamaTokenizer::Tokenize(const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max) {
    if (!mVocab) {
        LOG_E("Error: llama_vocab is not initailized.\n");
        return -1;
    }
    if (!text || !tokens || text_len <= 0 || n_tokens_max <= 0) {
        LOG_E("Invalid params: text=%p text_len=%d tokens=%p n_tokens_max=%d!\n",
                                                    text, text_len, tokens, n_tokens_max);
        return -1;
    }

    // Convert input text to string
    std::string input(text, text_len);

    // Get tokenization settings from vocab
    // const bool add_bos = llama_vocab_get_add_bos(mVocab);
    // const bool add_eos = llama_vocab_get_add_eos(mVocab);
    const bool add_special = true;
    const bool parse_special = true;

    // Tokenize the input text
    std::vector<llama_token> token_vec = common_tokenize(mVocab, input, add_special, parse_special);

    // Copy tokens to output buffer up to max size
    const int n_tokens = std::min((int32_t)token_vec.size(), n_tokens_max);
    if (n_tokens_max < n_tokens) {
        LOG_E("Warning: n_tokens_max is smaller than output token size.\n");
    }
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = token_vec[i];
    }

    return n_tokens;
}

std::string LlamaTokenizer::TokenToPiece(int32_t token) {
    if (!mVocab) {
        LOG_E("Error: llama_vocab is not initailized.\n");
        return "";
    }
    bool invalid_utf8 = false;
    return common_token_to_piece(mVocab, token, invalid_utf8);
}

std::string LlamaTokenizer::Decode(int32_t* tokens, int32_t n_tokens) {
    if (!mVocab) {
        LOG_E("Error: llama_vocab is not initailized.\n");
        return "";
    }
    bool invalid_utf8 = false;
    std::vector<llama_token> token_vec(tokens, tokens + n_tokens);
    return common_detokenize(mVocab, token_vec, invalid_utf8);
}