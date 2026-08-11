#include "Tokenizer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream> // TODO: remove me

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>   // For CommandLineToArgvW
#endif

static void print_usage_information(const char * argv0) {
    printf("usage: %s [options]\n\n", argv0);
    printf("The tokenize program tokenizes a prompt using a given model,\n");
    printf("and prints the resulting tokens to standard output.\n\n");
    printf("It needs a model file, a prompt, and optionally other flags\n");
    printf("to control the behavior of the tokenizer.\n\n");
    printf("    The possible options are:\n");
    printf("\n");
    printf("    -h, --help                                       print this help and exit\n");
    // printf("    -t TOKENIZER_PATH, --tokenizer TOKENIZER_PATH    path to tokenizer file.\n");
    printf("    -t MODEL_DIR, --tokenizer MODEL_DIR              path to HuggingFace model directory.\n");
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

int main(int raw_argc, char ** raw_argv) {
    const std::vector<std::string> argv = ingest_args(raw_argc, raw_argv);
    const int argc = argv.size();

    if (argc <= 1) {
        print_usage_information(argv[0].c_str());
        return 1;
    }

    bool show_token_count = false;
    const char * tokenizer_path = NULL;
    const char * prompt_arg = NULL;
    bool tokenizer_path_set = false;
    bool prompt_set = false;
    bool stdin_batch = false;
    bool stdin_decode = false;

    int iarg = 1;
    for (; iarg < argc; ++iarg) {
        std::string arg{argv[iarg]};
        if (arg == "-h" || arg == "--help") {
            print_usage_information(argv[0].c_str());
            return 0;
        }
        else if (arg == "--stdin-batch") {
            stdin_batch = true;
        }
        else if (arg == "--stdin-decode") {
            stdin_decode = true;
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

    if (!tokenizer_path_set) {
        printf("Error: must specify --tokenizer.\n");
        return 1;
    }
    if (prompt_set && prompt_arg == NULL) {
        printf("Error: --prompt requires an argument.\n");
        return 1;
    }
    if (!prompt_set && !stdin_batch && !stdin_decode) {
        printf("Error: must specify --prompt, --stdin-batch, or --stdin-decode.\n");
        return 1;
    }

    Tokenizer* tokenizer = new Tokenizer(tokenizer_path);
    if (!tokenizer || !tokenizer->IsLoaded()) {
        printf("Error: could not create Tokenizer or load failed.\n");
        delete tokenizer;
        return 1;
    }

    // ── Batch mode: length-prefixed, one text per line ──────────────
    if (stdin_batch) {
        // Protocol: first line = total count N, then N lines of:
        // "BYTELEN BYTES..."  (byte length, a space, then the raw bytes)
        std::string line;
        std::vector<int32_t> buf;
        int total = 0;
        if (std::getline(std::cin, line)) {
            total = static_cast<int>(strtol(line.c_str(), nullptr, 10));
        }
        for (int k = 0; k < total; ++k) {
            if (!std::getline(std::cin, line)) {
                printf("ERR\n");
                continue;
            }
            // Parse "BYTELEN BYTES..."
            size_t sp = line.find(' ');
            if (sp == std::string::npos) {
                printf("ERR\n");
                continue;
            }
            int blen = static_cast<int>(strtol(line.c_str(), nullptr, 10));
            // Text starts at sp+1; read exactly blen bytes (handle embedded
            // \r\n etc. by reading more if text was split by getline)
            std::string text = line.substr(sp + 1);
            while (static_cast<int>(text.size()) < blen) {
                if (!std::getline(std::cin, line)) break;
                text += "\n";
                text += line;
            }
            text.resize(blen); // trim any trailing extra
            buf.resize(text.size() + 3);
            int n = tokenizer->Tokenize(
                text.c_str(), static_cast<int32_t>(text.size()),
                buf.data(), static_cast<int32_t>(buf.size()));
            if (n < 0) {
                printf("ERR\n");
                continue;
            }
            printf("%d", n);
            for (int i = 0; i < n; i++)
                printf(" %d", buf[i]);
            printf("\n");
        }
        fflush(stdout);
        delete tokenizer;
        return 0;
    }

    // ── Decode-only batch mode: read "N id1 ... idN" per line ─────
    if (stdin_decode) {
        std::string line;
        std::vector<int32_t> buf;
        int total = 0;
        if (std::getline(std::cin, line)) {
            total = static_cast<int>(strtol(line.c_str(), nullptr, 10));
        }
        for (int k = 0; k < total; ++k) {
            if (!std::getline(std::cin, line)) {
                printf("0\n");  // empty
                continue;
            }
            if (line.back() == '\r') line.pop_back();
            std::istringstream iss(line);
            int id;
            buf.clear();
            while (iss >> id) buf.push_back(id);
            if (buf.empty()) {
                printf("0\n");
                continue;
            }
            std::string dec = tokenizer->Decode(buf.data(),
                static_cast<int32_t>(buf.size()));
            // Length-prefixed output: "BYTELEN BYTES\n" so embedded \n
            // in decoded text doesn't break line-based parsing.
            printf("%zu %s\n", dec.size(), dec.c_str());
        }
        fflush(stdout);
        delete tokenizer;
        return 0;
    }

    // ── Single-prompt mode (original behavior) ────────────────────────
    std::string prompt = prompt_arg;

    VocabInfo vocab_info;
    tokenizer->GetVocabInfo(&vocab_info);
    printf("vocab_info: vocab_size=%d n_special_bos_id=%d n_special_eos_id=%d\n",
           vocab_info.vocab_size, vocab_info.n_special_bos_id, vocab_info.n_special_eos_id);
    for (int i = 0; i < vocab_info.n_special_bos_id; i++)
        printf("special_bos_id[%d] = %d\n", i, vocab_info.special_bos_id[i]);
    for (int i = 0; i < vocab_info.n_special_eos_id; i++)
        printf("special_eos_id[%d] = %d\n", i, vocab_info.special_eos_id[i]);

    const int n_tokens_max = prompt.size() + 3;
    int32_t* tokens = (int32_t*)malloc(n_tokens_max * sizeof(int32_t));
    int token_size = tokenizer->Tokenize(
        prompt.c_str(), prompt.size(), tokens, n_tokens_max);
    if (token_size < 0) {
        printf("Error: tokenize failed.\n");
        return 1;
    }
    if (token_size == 0)
        printf("(no tokens)\n");

    for (int i = 0; i < token_size; i++)
        printf("%6d -> '%s'\n", tokens[i],
               (tokenizer->TokenToPiece(tokens[i])).c_str());

    printf("Decode: %s\n", (tokenizer->Decode(tokens, token_size)).c_str());
    if (show_token_count)
        printf("Total number of tokens: %d\n", token_size);

    free(tokens);
    delete tokenizer;
    return 0;
}
