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

echo "[merge] unpacking re2..."
if [ -f "${HF_BUILD_DIR}/third-party/re2/libre2.a" ]; then
    ar -x "${HF_BUILD_DIR}/third-party/re2/libre2.a"
fi

# Abseil is NOT unpacked here because sentencepiece uses its internal
# SPM_ABSL_PROVIDER=internal stubs. The 91 libabsl_*.a in the build tree
# are compiled by CMake but never actually needed at link time — unpacking
# them would inject ~11 MB of dead code into the merged archive.

echo "[merge] stripping debug sections..."
for f in *.o; do
    strip -S "$f" 2>/dev/null || true
done

echo "[merge] creating ${OUTPUT}..."
ar -qcs "${OUTPUT}" *.o
strip -s "${OUTPUT}"
rm -f *.o

echo "[merge] done: $(du -sh "${OUTPUT}" | cut -f1)"