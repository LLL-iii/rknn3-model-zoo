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
            printf("%s: special_eos_id[%d] = %d\n", __func__, vocab_info->n_special_eos_id, special_eos[i]);
        }
    }

    // 补充 eog 的 token
    std::set<llama_token> special_eog = llama_vocab_eog(mVocab);
    for (auto token : special_eog)
    {
        // 只有当 token 不在 special_eos_id 中时才添加
        bool exists = false;
        for (int j = 0; j < vocab_info->n_special_eos_id; ++j)
        {
            if (vocab_info->special_eos_id[j] == token)
            {
                exists = true;
                break;
            }
        }
        if (!exists)
        {
            vocab_info->special_eos_id[vocab_info->n_special_eos_id] = token;
            ++(vocab_info->n_special_eos_id);
            printf("%s: special_eog_id[%d] = %d\n", __func__, vocab_info->n_special_eos_id, token);
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

    // Get tokenization settings from vocab
    // const bool add_bos = llama_vocab_get_add_bos(mVocab);
    // const bool add_eos = llama_vocab_get_add_eos(mVocab);
    const bool add_special = true;
    const bool parse_special = true;

    // Avoid building a temporary std::string and a temporary std::vector in
    // common_tokenize(). llama_tokenize() writes directly to the caller buffer
    // and the llama_vocab char* path reuses thread-local scratch buffers.
    const int n_tokens = llama_tokenize(
        mVocab,
        text,
        text_len,
        reinterpret_cast<llama_token *>(tokens),
        n_tokens_max,
        add_special,
        parse_special);

    if (n_tokens < 0) {
        LOG_E("Warning: n_tokens_max is smaller than output token size, required=%d.\n", -n_tokens);
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