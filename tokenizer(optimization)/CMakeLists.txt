cmake_minimum_required(VERSION 3.18)
project("tokenizer" C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options ----
option(FATAL_WARNINGS   "Enable -Werror" OFF)
option(SANITIZE_ADDRESS "Enable ASAN"    OFF)

# ---- Size optimisation ----
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -Os -ffunction-sections -fdata-sections")
set(CMAKE_C_FLAGS_RELEASE
    "${CMAKE_C_FLAGS_RELEASE}   -Os -ffunction-sections -fdata-sections")
# LTO causes .o files to store GIMPLE IR instead of machine code, 4-5x bloat.
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# ---- Embedded compat ---- static-link runtime libs to avoid GLIBCXX mismatch
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")

# ---- Global PIC (abseil requires it) ----
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Options forwarded to meta-pytorch CMakeLists.txt ----
set(TOKENIZERS_BUILD_TEST   OFF CACHE BOOL "")
set(TOKENIZERS_BUILD_TOOLS  OFF CACHE BOOL "")
set(TOKENIZERS_BUILD_PYTHON OFF CACHE BOOL "")
set(SUPPORT_REGEX_LOOKAHEAD ON CACHE BOOL "")
set(SPM_BUILD_TEST OFF CACHE BOOL "")
set(SPM_ENABLE_SHARED OFF CACHE BOOL "")
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "")

# ---- Build meta-pytorch/tokenizers ----
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tokenizers
    ${CMAKE_CURRENT_BINARY_DIR}/hf-build
)

# ---- RKNN3 tokenizer main library (single .cpp) ----
add_library(tokenizer STATIC src/Tokenizer.cpp)

target_include_directories(tokenizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tokenizers/include
)

target_link_libraries(tokenizer tokenizers)

# ---- Merge all .a into single libtokenizer_merged.a ----
add_custom_command(
    TARGET tokenizer POST_BUILD
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/merge_libs.sh
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/hf-build
            ${CMAKE_CURRENT_BINARY_DIR}/libtokenizer_merged.a
    COMMENT "Merging all deps into libtokenizer_merged.a"
)

# ---- Demo ----
add_subdirectory(demo)

# ---- Install ----
install(FILES  ${CMAKE_CURRENT_SOURCE_DIR}/include/Tokenizer.h
               DESTINATION include)
install(FILES  ${CMAKE_CURRENT_BINARY_DIR}/libtokenizer_merged.a
               DESTINATION lib RENAME libtokenizer.a)
install(TARGETS tokenize_demo
        RUNTIME DESTINATION demo)
