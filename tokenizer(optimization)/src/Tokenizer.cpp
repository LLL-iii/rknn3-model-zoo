#include "Tokenizer.h"
#include "pytorch/tokenizers/hf_tokenizer.h"
#include "pytorch/tokenizers/sentencepiece.h"
#include <cstdio>
#include <algorithm>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

static inline bool file_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}
static inline bool is_directory(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

class TokenizerImpl {
public:
    tokenizers::HFTokenizer  hfTok;
    tokenizers::SPTokenizer  spTok;
    bool useSPM = false;
    VocabInfo vocabInfo;
    bool loaded = false;
};

static VocabInfo makeSPMInfo(tokenizers::SPTokenizer& sp) {
    VocabInfo v;
    v.vocab_size        = sp.vocab_size();
    v.special_bos_id[0] = static_cast<int>(sp.bos_tok());
    v.special_eos_id[0] = static_cast<int>(sp.eos_tok());
    v.n_special_bos_id  = 1;
    v.n_special_eos_id  = 1;
    v.linefeed_id       = 13;
    return v;
}

static VocabInfo makeBPEInfo(tokenizers::HFTokenizer& hf) {
    VocabInfo v;
    v.vocab_size        = hf.vocab_size();
    v.special_bos_id[0] = static_cast<int>(hf.bos_tok());
    v.special_eos_id[0] = static_cast<int>(hf.eos_tok());
    v.n_special_bos_id  = 1;
    v.n_special_eos_id  = 1;
    v.linefeed_id       = 13;
    return v;
}

Tokenizer::Tokenizer(const char* model_dir)
    : impl_(std::make_unique<TokenizerImpl>()) {

    std::string dir(model_dir);

    std::string spModel = dir + "/tokenizer.model";
    if (file_exists(spModel)) {
        auto err = impl_->spTok.load(spModel);
        if (err == tokenizers::Error::Ok) {
            impl_->useSPM    = true;
            impl_->loaded    = true;
            impl_->vocabInfo = makeSPMInfo(impl_->spTok);
            return;
        }
    }

    auto err = impl_->hfTok.load(dir);
    if (err != tokenizers::Error::Ok) {
        fprintf(stderr, "[Tokenizer] load failed: %s (error=%d)\n",
                dir.c_str(), static_cast<int>(err));
        return;
    }

    int vs = impl_->hfTok.vocab_size();
    if (vs <= 0 || vs > 500000) {
        fprintf(stderr, "[Tokenizer] load succeeded but vocab_size=%d is invalid for %s\n",
                vs, dir.c_str());
        return;
    }

    impl_->loaded    = true;
    impl_->vocabInfo = makeBPEInfo(impl_->hfTok);
}

Tokenizer::Tokenizer(const void* data, size_t len)
    : impl_(std::make_unique<TokenizerImpl>()) {

    std::string tmp = "/tmp/tokenizer_rknn3_tmp.json";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "[Tokenizer] Cannot create temp file\n"); return; }
    fwrite(data, 1, len, f);
    fclose(f);
    auto err = impl_->hfTok.load(tmp);
    remove(tmp.c_str());
    if (err != tokenizers::Error::Ok) {
        fprintf(stderr, "[Tokenizer] load from buffer failed (error=%d)\n", static_cast<int>(err));
        return;
    }
    int vs = impl_->hfTok.vocab_size();
    if (vs <= 0 || vs > 500000) {
        fprintf(stderr, "[Tokenizer] load from buffer: invalid vocab_size=%d\n", vs);
        return;
    }
    impl_->loaded    = true;
    impl_->vocabInfo = makeBPEInfo(impl_->hfTok);
}

Tokenizer::~Tokenizer() = default;

bool Tokenizer::GetVocabInfo(VocabInfo* info) {
    if (!impl_ || !impl_->loaded) return false;
    *info = impl_->vocabInfo;
    return true;
}

const char* Tokenizer::GetBackendType() const {
    return (impl_ && impl_->useSPM) ? "SPM" : "BPE";
}

int Tokenizer::Tokenize(const char* text, int32_t text_len,
                         int32_t* tokens, int32_t max) {
    if (!impl_ || !impl_->loaded) return -1;

    if (impl_->useSPM) {
        auto res = impl_->spTok.encode(std::string(text, text_len), 0, 0);
        if (!res.ok()) return -1;
        const auto& ids = res.get();
        int n = std::min(max, static_cast<int32_t>(ids.size()));
        for (int i = 0; i < n; i++) tokens[i] = static_cast<int32_t>(ids[i]);
        return n;
    }

    auto res = impl_->hfTok.encode(std::string(text, text_len), 0, 0);
    if (!res.ok()) return -1;
    const auto& ids = res.get();
    int n = std::min(max, static_cast<int32_t>(ids.size()));
    for (int i = 0; i < n; i++) tokens[i] = static_cast<int32_t>(ids[i]);
    return n;
}

std::string Tokenizer::TokenToPiece(int32_t token) {
    if (!impl_ || !impl_->loaded) return "";
    try {
        if (impl_->useSPM) {
            auto r = impl_->spTok.id_to_piece(static_cast<uint64_t>(token));
            return r.ok() ? r.get() : "";
        }
        auto r = impl_->hfTok.id_to_piece(static_cast<uint64_t>(token));
        return r.ok() ? r.get() : "";
    } catch (...) { return ""; }
}

std::string Tokenizer::Decode(int32_t* tokens, int32_t count) {
    if (!impl_ || !impl_->loaded || count <= 0) return "";
    try {
        std::string ret;
        if (impl_->useSPM) {
            uint64_t prev = impl_->spTok.bos_tok();
            for (int32_t i = 0; i < count; i++) {
                auto d = impl_->spTok.decode(prev, static_cast<uint64_t>(tokens[i]));
                if (d.ok()) ret += d.get();
                prev = static_cast<uint64_t>(tokens[i]);
            }
        } else {
            uint64_t prev = impl_->hfTok.bos_tok();
            for (int32_t i = 0; i < count; i++) {
                auto d = impl_->hfTok.decode(prev, static_cast<uint64_t>(tokens[i]));
                if (d.ok()) ret += d.get();
                prev = static_cast<uint64_t>(tokens[i]);
            }
            for (size_t p = 0; (p = ret.find("\xE2\x96\x81", p))
                 != std::string::npos; p += 1)
                ret.replace(p, 3, " ");
        }
        return ret;
    } catch (...) { return ""; }
}

bool Tokenizer::IsLoaded() const {
    return impl_ && impl_->loaded;
}
