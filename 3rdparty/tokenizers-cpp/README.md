# tokenizers-cpp

This directory vendors the prebuilt headers and static libraries used by
`examples/Qwen3_TTS/cpp`.

Upstream project:

- https://github.com/mlc-ai/tokenizers-cpp

This local README is kept together with the upstream Apache-2.0 `LICENSE` so
that the dependency layout under the top-level `3rdparty/` directory matches
the shared third-party dependency style used elsewhere in this repo.

Upstream summary:

- Cross-platform C++ tokenizer binding library
- Wraps HuggingFace tokenizers and sentencepiece
- Common deployment targets include Linux and Android

In this repository, the following prebuilt artifacts are used:

- `include/tokenizers_cpp.h`
- `include/tokenizers_c.h`
- `Linux/aarch64/libtokenizers_cpp.a`
- `Linux/aarch64/libtokenizers_c.a`
- `Android/arm64-v8a/libtokenizers_cpp.a`
- `Android/arm64-v8a/libtokenizers_c.a`

If these binaries need to be refreshed, please refer to the upstream project
for the original source layout and build instructions.
