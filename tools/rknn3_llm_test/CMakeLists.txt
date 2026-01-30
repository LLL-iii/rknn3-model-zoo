cmake_minimum_required(VERSION 3.10)

project(rknn3_llm_test)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()


set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/rknpu3)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBRKNNAPI ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknn3_api.so)
else()
    set(LIBRKNNAPI ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_ANDROID_ARCH_ABI}/librknn3_api.so) 
endif()
set(LIBRKNNAPI_INCLUDES ${RKNN_PATH}/include)

set(JSON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/json)
set(LIBJSON_INCLUDES ${JSON_PATH})


set(TOKENIZER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/tokenizer)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTOKENIZER ${TOKENIZER_PATH}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/libtokenizer.a)
else()
    set(LIBTOKENIZER ${TOKENIZER_PATH}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_ANDROID_ARCH_ABI}/libtokenizer.a)
endif()
set(LIBTOKENIZER_INCLUDES ${TOKENIZER_PATH}/include)


set(LLM_DATASET_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../datasets/llm_dataset)

include_directories(${LIBRKNNAPI_INCLUDES})
include_directories(${LIBTOKENIZER_INCLUDES})
include_directories(${LIBJSON_INCLUDES})

add_executable(rknn3_session_test_eval_perf src/rknn3_session_test_eval_perf.cpp)
add_executable(rknn3_session_test_cmmlu src/rknn3_session_test_cmmlu.cpp)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_link_libraries(rknn3_session_test_eval_perf ${LIBRKNNAPI} ${LIBTOKENIZER} pthread dl)
    target_link_libraries(rknn3_session_test_cmmlu ${LIBRKNNAPI} ${LIBTOKENIZER} pthread dl)
else()
    target_link_libraries(rknn3_session_test_eval_perf ${LIBRKNNAPI} ${LIBTOKENIZER} log)
    target_link_libraries(rknn3_session_test_cmmlu ${LIBRKNNAPI} ${LIBTOKENIZER} log)
endif()

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install/rknn3_llm_test_${TARGET_SOC}_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn3_session_test_eval_perf DESTINATION ./)
install(TARGETS rknn3_session_test_cmmlu DESTINATION ./)
install(DIRECTORY ${LLM_DATASET_PATH} DESTINATION ./)
install(PROGRAMS ${LIBRKNNAPI} DESTINATION lib)
