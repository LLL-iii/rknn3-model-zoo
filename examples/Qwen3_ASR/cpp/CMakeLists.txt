cmake_minimum_required(VERSION 3.10)

project(rknn_qwen3_asr_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

include_directories(${CMAKE_SOURCE_DIR}/llm)
include_directories(${CMAKE_SOURCE_DIR}/audio/)

set(llm_file llm/rknn_qwen3_asr_llm.cc)
set(audio_file audio/rknn_qwen3_asr_audio.cc)

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "TARGET_LIB_ARCH = ${TARGET_LIB_ARCH}")

add_executable(${PROJECT_NAME}
    main.cc
    qwen3_asr.cc
    ${llm_file}
    ${audio_file}
)

add_executable(${PROJECT_NAME}_online
    online_main.cc
    qwen3_asr.cc
    ${llm_file}
    ${audio_file}
)

target_link_libraries(${PROJECT_NAME}
    timeutils
    audioutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

target_link_libraries(${PROJECT_NAME}_online
    timeutils
    audioutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)

    target_link_libraries(${PROJECT_NAME}_online
    log
)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME}
        pthread
    )

    target_link_libraries(${PROJECT_NAME}_online
        pthread
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

target_include_directories(${PROJECT_NAME}_online PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(TARGETS ${PROJECT_NAME}_online DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../data/audio/asr_en.wav DESTINATION ./)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../data/audio/mel_128_filters.txt DESTINATION ./)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*/*.rknn")
set(WEIGHT_FILES "")
foreach(rknn_file ${RKNN_FILES})
    string(REPLACE ".rknn" ".weight" weight_file ${rknn_file})
    if(EXISTS ${weight_file})
        list(APPEND WEIGHT_FILES ${weight_file})
    endif()
endforeach()
file(GLOB TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.tokenizer.gguf")
file(GLOB EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.embed.bin")
install(FILES ${RKNN_FILES} DESTINATION ./model)
install(FILES ${WEIGHT_FILES} DESTINATION ./model)
install(FILES ${TOKENIZER_FILES} DESTINATION ./model)
install(FILES ${EMBED_FILES} DESTINATION ./model)
