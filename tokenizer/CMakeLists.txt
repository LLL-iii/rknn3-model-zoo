cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
# set(CMAKE_BUILD_TYPE Debug)
project("tokenizer" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
add_compile_options(-D_ISOC11_SOURCE -fPIC)

# build
option(FATAL_WARNINGS "tokenizer: enable -Werror flag" OFF)

# sanitizers
option(SANITIZE_THREAD    "tokenizer: enable thread sanitizer"    OFF)
option(SANITIZE_ADDRESS   "tokenizer: enable address sanitizer"   OFF)
option(SANITIZE_UNDEFINED "tokenizer: enable undefined sanitizer" OFF)

# override ggml options
set(GGML_ALL_WARNINGS   ${FATAL_WARNINGS})
set(GGML_FATAL_WARNINGS ${FATAL_WARNINGS})

if (SANITIZE_THREAD)
    message(STATUS "Using -fsanitize=thread")

    add_compile_options(-fsanitize=thread)
    link_libraries     (-fsanitize=thread)
endif()

if (SANITIZE_ADDRESS)
    message(STATUS "Using -fsanitize=address")

    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    link_libraries     (-fsanitize=address)
endif()

if (SANITIZE_UNDEFINED)
    message(STATUS "Using -fsanitize=undefined")

    add_compile_options(-fsanitize=undefined)
    link_libraries     (-fsanitize=undefined)
endif()


set(CMAKE_C_FLAGS_DEBUG            "-O0 -g3"          CACHE STRING "Flags used by the C compiler during DEBUG builds.")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3"          CACHE STRING "Flags used by the CXX compiler during DEBUG builds.")
set(CMAKE_C_FLAGS_RELEASE          "-O2 -s"           CACHE STRING "Flags used by the C compiler during RELEASE builds." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -s"           CACHE STRING "Flags used by the CXX compiler during RELEASE builds." FORCE)

add_subdirectory(thirdparty/llama_vocab)

include_directories(${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_SOURCE_DIR}/src
                    ${CMAKE_SOURCE_DIR}/src/llama
                    ${CMAKE_SOURCE_DIR}/src/utils
                    ${CMAKE_SOURCE_DIR}/thirdparty/llama_vocab/include
                    ${CMAKE_SOURCE_DIR}/thirdparty/llama_vocab/ggml/include
                    ${CMAKE_SOURCE_DIR}/thirdparty/llama_vocab/common)

add_library(tokenizer STATIC
            src/Tokenizer.cpp
            src/TokenizerBase.h
            src/llama/LlamaTokenizer.cpp)

add_dependencies(tokenizer llama_merge_static_libs)

target_link_libraries(tokenizer  ${LLAMA_LIBS})


target_compile_features(tokenizer PUBLIC cxx_std_11) # don't bump

add_subdirectory(demo)

# INSTALL(TARGETS tokenizer DESTINATION lib)
INSTALL(FILES ${CMAKE_SOURCE_DIR}/include/Tokenizer.h DESTINATION include)

# 静态库路径
set(TOKENIZER_LIB ${CMAKE_CURRENT_BINARY_DIR}/libtokenizer.a)
set(TOKENIZER_LIBS ${CMAKE_CURRENT_BINARY_DIR}/libtokenizer_merged.a)


# 合并命令
add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E echo "TOKENIZER_LIB: ${TOKENIZER_LIB}"
    COMMAND ${CMAKE_COMMAND} -E echo "LLAMA_LIBS: ${LLAMA_LIBS}"
    OUTPUT ${TOKENIZER_LIBS}
    COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_LIB} ${TOKENIZER_LIBS}
    COMMAND ${CMAKE_AR} -x ${LLAMA_LIBS}
    COMMAND ${CMAKE_AR} -q ${TOKENIZER_LIBS} *.o*
    COMMAND ${CMAKE_COMMAND} -E remove *.o*
    DEPENDS tokenizer ${LLAMA_LIBS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}  # 必须工作在能写文件的地方
    COMMENT "Merging libllama_vocab.a libtokenizer.a into one static library"
)

# 添加合并目标
add_custom_target(tokenizer_static_libs ALL DEPENDS ${TOKENIZER_LIBS})

# 安装时替换成合并后的版本
install(FILES ${TOKENIZER_LIBS} DESTINATION lib RENAME libtokenizer.a)
