cmake_minimum_required(VERSION 3.10)

project(rknn_qwen3_tts_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/tokenizers-cpp)
set(TOKENIZERS_CPP_INCLUDE_DIR ${TOKENIZERS_CPP_ROOT}/include)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(TOKENIZERS_CPP_LIB_DIR ${TOKENIZERS_CPP_ROOT}/Android/arm64-v8a)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TOKENIZERS_CPP_LIB_DIR ${TOKENIZERS_CPP_ROOT}/Linux/aarch64)
else()
    message(FATAL_ERROR "Unsupported system for local tokenizers-cpp: ${CMAKE_SYSTEM_NAME}")
endif()

set(LIBTOKENIZER_CPP ${TOKENIZERS_CPP_LIB_DIR}/libtokenizers_cpp.a)
set(LIBTOKENIZER_C ${TOKENIZERS_CPP_LIB_DIR}/libtokenizers_c.a)

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "TOKENIZERS_CPP_INCLUDE_DIR = ${TOKENIZERS_CPP_INCLUDE_DIR}")
message(STATUS "LIBTOKENIZER_CPP = ${LIBTOKENIZER_CPP}")
message(STATUS "LIBTOKENIZER_C = ${LIBTOKENIZER_C}")

add_executable(${PROJECT_NAME}
    main.cc
    talker.cc
    speech_decoder.cc
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
    ${TOKENIZERS_CPP_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    ${LIBRKNNAPI}
    ${LIBTOKENIZER_CPP}
    ${LIBTOKENIZER_C}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION .)

install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/model\")")

file(GLOB CODE_PREDICTOR_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/code_predictor/*.rknn")
file(GLOB CODE_PREDICTOR_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/code_predictor/*.weight")
file(GLOB SPEECH_DECODER_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/speech_decoder/*.rknn")
file(GLOB SPEECH_DECODER_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/speech_decoder/*.weight")
file(GLOB TALKER_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/talker/*.rknn")
file(GLOB TALKER_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/talker/*.weight")
file(GLOB TEXT_PROJECTOR_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/text_projector/*.rknn")
file(GLOB TEXT_PROJECTOR_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/text_projector/*.weight")
file(GLOB EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../models/embeds/*.bin" "${CMAKE_CURRENT_SOURCE_DIR}/../models/embeds/tokenizer.json")

install(
    FILES
        ${CODE_PREDICTOR_RKNN_FILES}
        ${CODE_PREDICTOR_WEIGHT_FILES}
        ${SPEECH_DECODER_RKNN_FILES}
        ${SPEECH_DECODER_WEIGHT_FILES}
        ${TALKER_RKNN_FILES}
        ${TALKER_WEIGHT_FILES}
        ${TEXT_PROJECTOR_RKNN_FILES}
        ${TEXT_PROJECTOR_WEIGHT_FILES}
        ${EMBED_FILES}
    DESTINATION ./model
)
