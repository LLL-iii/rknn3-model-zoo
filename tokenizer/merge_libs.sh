#!/usr/bin/env bash
set -e

BUILD_DIR="$1"
HF_BUILD_DIR="$2"
OUTPUT="$3"

MERGED_DIR="${BUILD_DIR}/merged"
mkdir -p "${MERGED_DIR}"
cd "${MERGED_DIR}"

echo "[merge] unpacking tokenizer..."
ar -x "${BUILD_DIR}/libtokenizer.a"
ar -x "${HF_BUILD_DIR}/libtokenizers.a"
ar -x "${HF_BUILD_DIR}/sp-build/src/libsentencepiece.a"

# PCRE2 is the single regex engine (RE2 + abseil were removed to enable C++11
# builds).  Its symbols live in regex_lookahead (Pcre2Regex/StdRegex adapters)
# and pcre2-8-static (the PCRE2 engine itself); both must be merged in.
echo "[merge] unpacking regex engine (PCRE2)..."
ar -x "${HF_BUILD_DIR}/libregex_lookahead.a"
ar -x "${HF_BUILD_DIR}/third-party/pcre2/libpcre2-8.a"

echo "[merge] stripping debug sections..."
for f in *.o; do
    strip -S "$f" 2>/dev/null || true
done

echo "[merge] creating ${OUTPUT}..."
ar -qcs "${OUTPUT}" *.o
# strip -s 会删光全局符号导致外部无法链接；改用 --strip-unneeded 仅删调试/local 符号
strip --strip-unneeded "${OUTPUT}"
rm -f *.o

echo "[merge] done: $(du -sh "${OUTPUT}" | cut -f1)"
