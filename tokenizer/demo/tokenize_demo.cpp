#include "Tokenizer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream> // TODO: remove me

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>   // For CommandLineToArgvW
#endif

#define USE_FILE_PATH_DIRECTLY 0

static void print_usage_information(const char * argv0) {
    printf("usage: %s [options]\n\n", argv0);
    printf("The tokenize program tokenizes a prompt using a given model,\n");
    printf("and prints the resulting tokens to standard output.\n\n");
    printf("It needs a model file, a prompt, and optionally other flags\n");
    printf("to control the behavior of the tokenizer.\n\n");
    printf("    The possible options are:\n");
    printf("\n");
    printf("    -h, --help                                       print this help and exit\n");
    printf("    -t TOKENIZER_PATH, --tokenizer TOKENIZER_PATH    path to tokenizer file.\n");
    printf("    -p PROMPT, --prompt PROMPT                       read prompt from the argument.\n");
    printf("    --show-count                                     print the total number of tokens.\n");
}

static std::vector<std::string> ingest_args(int raw_argc, char ** raw_argv) {
    std::vector<std::string> argv;

    int argc = raw_argc;
    for (int i = 0; i < argc; ++i) {
        argv.push_back(raw_argv[i]);
    }

    return argv;
}

void* load_file_to_memory(const char* filename, size_t* outSize) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("fopen failed");
        return nullptr;
    }

    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    rewind(fp);  // 或 fseek(fp, 0, SEEK_SET);

    if (fileSize <= 0) {
        fclose(fp);
        return nullptr;
    }

    char* buffer = (char*)malloc(fileSize);
    if (!buffer) {
        fclose(fp);
        return nullptr;
    }

    size_t readSize = fread(buffer, 1, fileSize, fp);
    fclose(fp);

    if (readSize != (size_t)fileSize) {
        free(buffer);
        return nullptr;
    }

    *outSize = fileSize;
    return buffer;
}

int main(int raw_argc, char ** raw_argv) {
    const std::vector<std::string> argv = ingest_args(raw_argc, raw_argv);
    const int argc = argv.size();

    if (argc <= 1) {
        print_usage_information(argv[0].c_str());
        return 1;
    }

    //////
    // Read out all the command line arguments.
    //////

    // variables where to put any arguments we see.
    bool show_token_count = false;
    const char * tokenizer_path = NULL;
    const char * prompt_path = NULL;
    const char * prompt_arg = NULL;

    // track which arguments were explicitly given
    // used for sanity checking down the line
    bool tokenizer_path_set = false;
    bool prompt_set = false;
    bool stdin_set = false;

    int iarg = 1;
    for (; iarg < argc; ++iarg) {
        std::string arg{argv[iarg]};
        if (arg == "-h" || arg == "--help") {
            print_usage_information(argv[0].c_str());
            return 0;
        }
        else if (arg == "-t" || arg == "--tokenizer") {
            if (tokenizer_path_set) {
                printf("Error: -t or --tokenizer specified multiple times.\n");
                return 1;
            }
            tokenizer_path = argv[++iarg].c_str();
            tokenizer_path_set = true;
        }
        else if (arg == "-p" || arg == "--prompt") {
            if (prompt_set) {
                printf("Error: -p or --prompt specified multiple times.\n");
                return 1;
            }
            prompt_arg = argv[++iarg].c_str();
            prompt_set = true;
        }
        else if (arg == "--show-count") {
            show_token_count = true;
        }
        else {
            printf("Error: unknown option '%s'\n", argv[iarg].c_str());
            return 1;
        }
    }

    //////
    // Sanity check the command line arguments.
    //////

    // Check that we have the required stuff set.
    if (tokenizer_path_set && tokenizer_path == NULL) {
        printf("Error: --tokenizer requires an argument.\n");
        return 1;
    }
    if (!tokenizer_path_set) {
        printf("Error: must specify --tokenizer.\n");
        return 1;
    }
    if (prompt_set && prompt_arg == NULL) {
        printf("Error: --prompt requires an argument.\n");
        return 1;
    }
    std::string prompt = prompt_arg;

#if USE_FILE_PATH_DIRECTLY
    Tokenizer* tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
#else
    size_t buffer_size = 0;
    void* model_buffer = load_file_to_memory(tokenizer_path, &buffer_size);
    if (!model_buffer) {
        printf("Error: failed to open GGUF file '%s'\n", tokenizer_path);
        return 1;
    }
    printf("GGUF file '%s', buffer=%p, size=%zu\n", tokenizer_path, model_buffer, buffer_size);
    Tokenizer* tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, model_buffer, buffer_size);
    free(model_buffer);
#endif
    if (!tokenizer) {
        printf("Error: could not create Tokenizer.\n");
        return 1;
    }

    VocabInfo vocab_info;
    tokenizer->GetVocabInfo(&vocab_info);
    printf("vocab_info: vocab_size=%d n_special_bos_id=%d n_special_eos_id=%d\n",
           vocab_info.vocab_size, vocab_info.n_special_bos_id, vocab_info.n_special_eos_id);
    for (int i = 0; i < vocab_info.n_special_bos_id; i++) {
        printf("special_bos_id[%d] = %d\n", i, vocab_info.special_bos_id[i]);
    }
    for (int i = 0; i < vocab_info.n_special_eos_id; i++) {
        printf("special_eos_id[%d] = %d\n", i, vocab_info.special_eos_id[i]);
    }

    const int n_tokens_max = prompt.size() + 3;
    int32_t* tokens = (int32_t*)malloc(n_tokens_max * sizeof(int32_t));
    int token_size = tokenizer->Tokenize(prompt.c_str(), prompt.size(), tokens, n_tokens_max);
    if(token_size <= 0) {
        printf("Error: tokenize failed.\n");
        return 1;
    }

    for (int i = 0; i < token_size; i++) {
        printf("%6d -> '", tokens[i]);
        printf("%s", (tokenizer->TokenToPiece(tokens[i])).c_str());
        printf("'\n");
    }

    printf("Decode: %s\n", (tokenizer->Decode(tokens, token_size)).c_str());

    if (show_token_count) {
        printf("Total number of tokens: %d\n", token_size);
    }

    free(tokens);
    delete tokenizer;

    return 0;
}
