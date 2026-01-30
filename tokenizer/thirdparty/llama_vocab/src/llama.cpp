#include "llama-impl.h"

// #include "llama-chat.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
// #include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    // return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
    //        llama_supports_rpc();
    return false;
}

bool llama_supports_rpc(void) {
    // return ggml_backend_reg_by_name("RPC") != nullptr;
    return false;
}

void llama_backend_init(void) {
    // ggml_time_init();

    // needed to initialize f16 tables
    // {
    //     struct ggml_init_params params = { 0, NULL, false };
    //     struct ggml_context * ctx = ggml_init(params);
    //     ggml_free(ctx);
    // }
}

void llama_backend_free(void) {
    // ggml_quantize_free();
}

// int64_t llama_time_us(void) {
//     return ggml_time_us();
// }

llama_vocab * llama_load_vocab_from_file(const char * fname,
                                            bool use_mmap,
                                            bool check_tensors,
                                            const llama_model_kv_override * param_overrides_p) {
    std::vector<std::string> splits = {};
    llama_model_loader ml(fname, splits, use_mmap, check_tensors, param_overrides_p);
    llm_arch arch = LLM_ARCH_UNKNOWN;
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw_runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
    const auto kv = LLM_KV(arch);

    llama_vocab * vocab = new llama_vocab();
    if (vocab) {
        vocab->load(ml, kv);
        vocab->print_info();
    } else {
        LLAMA_LOG_ERROR("%s: Invaild llama_vocab!\n", __func__);
        return NULL;
    }

    return vocab;
}

llama_vocab * llama_load_vocab_from_buffer(const void * model_buffer,
                                            size_t buffer_size,
                                            bool use_mmap,
                                            bool check_tensors,
                                            const llama_model_kv_override * param_overrides_p) {
    llama_model_loader ml(model_buffer, buffer_size, use_mmap, check_tensors, param_overrides_p);
    llm_arch arch = LLM_ARCH_UNKNOWN;
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw_runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
    const auto kv = LLM_KV(arch);

    llama_vocab * vocab = new llama_vocab();
    if (vocab) {
        vocab->load(ml, kv);
        vocab->print_info();
    } else {
        LLAMA_LOG_ERROR("%s: Invaild llama_vocab!\n", __func__);
        return NULL;
    }

    return vocab;
}

void llama_vocab_free(llama_vocab * vocab) {
    delete vocab;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

// const char * llama_print_system_info(void) {
//     static std::string s;
//     s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

//     for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
//         auto * reg = ggml_backend_reg_get(i);
//         auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
//         if (get_features_fn) {
//             ggml_backend_feature * features = get_features_fn(reg);
//             s += ggml_backend_reg_name(reg);
//             s += " : ";
//             for (; features->name; features++) {
//                 s += features->name;
//                 s += " = ";
//                 s += features->value;
//                 s += " | ";
//             }
//         }
//     }

//     return s.c_str();
// }
