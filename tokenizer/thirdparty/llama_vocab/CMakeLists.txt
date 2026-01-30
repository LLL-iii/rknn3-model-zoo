cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("llama_vocab" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
add_compile_options(-D_ISOC11_SOURCE -fPIC)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

if (LLAMA_SANITIZE_THREAD)
    message(STATUS "Using -fsanitize=thread")

    add_compile_options(-fsanitize=thread)
    link_libraries     (-fsanitize=thread)
endif()

if (LLAMA_SANITIZE_ADDRESS)
    message(STATUS "Using -fsanitize=address")

    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    link_libraries     (-fsanitize=address)
endif()

if (LLAMA_SANITIZE_UNDEFINED)
    message(STATUS "Using -fsanitize=undefined")

    add_compile_options(-fsanitize=undefined)
    link_libraries     (-fsanitize=undefined)
endif()

set(CMAKE_C_FLAGS_DEBUG            "-O0 -g3"          CACHE STRING "Flags used by the C compiler during DEBUG builds.")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3"          CACHE STRING "Flags used by the CXX compiler during DEBUG builds.")
set(CMAKE_C_FLAGS_RELEASE          "-O2 -s"           CACHE STRING "Flags used by the C compiler during RELEASE builds." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -s"           CACHE STRING "Flags used by the CXX compiler during RELEASE builds." FORCE)

add_subdirectory(ggml)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_library(llama_vocab STATIC
            src/llama.cpp
            src/llama-arch.cpp
            src/llama-impl.cpp
            src/llama-mmap.cpp
            src/llama-model-loader.cpp
            src/llama-vocab.cpp
            src/unicode-data.cpp
            src/unicode.cpp
            common/common.cpp
            common/log.cpp
            )

target_link_libraries(llama_vocab PUBLIC ggml)

target_compile_features(llama_vocab PUBLIC cxx_std_11) # don't bump

add_subdirectory(demo)

# 判断是否为主项目
# INSTALL(TARGETS llama_vocab DESTINATION lib)
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h DESTINATION include)
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/common/common.h DESTINATION include)

# 静态库路径
set(LLAMA_LIB ${CMAKE_CURRENT_BINARY_DIR}/libllama_vocab.a)
set(GGML_LIB ${CMAKE_CURRENT_BINARY_DIR}/ggml/src/libggml.a)
set(GGML_BASE_LIB ${CMAKE_CURRENT_BINARY_DIR}/ggml/src/libggml-base.a)
set(LLAMA_MERGED_LIB ${CMAKE_CURRENT_BINARY_DIR}/libllama_vocab_merged.a)

# 合并命令
add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E echo "GGML_LIB: ${GGML_LIB}"
    COMMAND ${CMAKE_COMMAND} -E echo "GGML_BASE_LIB: ${GGML_BASE_LIB}"
    COMMAND ${CMAKE_COMMAND} -E echo "MERGED_LIB: ${LLAMA_MERGED_LIB}"
    OUTPUT ${LLAMA_MERGED_LIB}
    COMMAND ${CMAKE_COMMAND} -E copy ${LLAMA_LIB} ${LLAMA_MERGED_LIB}
    COMMAND ${CMAKE_AR} -x ${GGML_LIB}
    COMMAND ${CMAKE_AR} -x ${GGML_BASE_LIB}
    COMMAND ${CMAKE_AR} -q ${LLAMA_MERGED_LIB} *.o*
    COMMAND ${CMAKE_COMMAND} -E remove *.o*
    DEPENDS llama_vocab ggml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}  # 必须工作在能写文件的地方
    COMMENT "Merging libllama_vocab.a libggml.a libggml-base.a into one static library"
)

# 添加合并目标
add_custom_target(llama_merge_static_libs ALL DEPENDS ${LLAMA_MERGED_LIB})

# 安装时替换成合并后的版本
install(FILES ${LLAMA_MERGED_LIB} DESTINATION lib/ RENAME libllama_vocab.a)

set(LLAMA_LIBS ${LLAMA_MERGED_LIB} PARENT_SCOPE)


