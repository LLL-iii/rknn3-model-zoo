cmake_minimum_required(VERSION 3.10)

project(rknn_mobilenet_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(mobilenet_file mobilenet.cc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
    ${mobilenet_file}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    ${LIBRKNNAPI}
    dl
)

add_executable(dataset_eval
    dataset_eval.cc
    ${mobilenet_file}
)

target_link_libraries(dataset_eval
    imageutils
    fileutils
    ${LIBRKNNAPI}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)
    target_link_libraries(dataset_eval
    log
)
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
    target_link_libraries(dataset_eval Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)

target_include_directories(dataset_eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} dataset_eval DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/dog_224x224.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/labels.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
file(GLOB WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight")
install(FILES ${RKNN_FILES} DESTINATION model)
install(FILES ${WEIGHT_FILES} DESTINATION model)
