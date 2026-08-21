cmake_minimum_required(VERSION 3.18)
project("chat_template" C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 体积优化：libchat_template.a  ----
# 库用 -Os 且不用 -ffunction-sections：否则每函数一个 section，section 头表/符号表/字符串表
# 膨胀 .a 体积（机器码仅 ~1MB，元数据却占数 MB）。该 set 必须在 add_subdirectory 之前，
# 否则 jinja2cpp 子目录不继承；FORCE 覆盖 CMakeCache 中的旧 Release flags。
set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG" CACHE STRING "Release CXX flags" FORCE)
set(CMAKE_C_FLAGS_RELEASE   "-Os -DNDEBUG" CACHE STRING "Release C flags" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(JINJA_THIRDPARTY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

# ---- nlohmann_json::nlohmann_json（Jinja2Cpp nlohmann binding 依赖）----
# GLOBAL 确保对 add_subdirectory 的 jinja2cpp 子目录及 render_driver 都可见
if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JINJA_THIRDPARTY}/json")
endif()

# ---- Jinja2Cpp 构建选项（external 依赖 / nlohmann binding / std regex）----
set(JINJA2CPP_DEPS_MODE           external CACHE STRING "")
set(JINJA2CPP_WITH_JSON_BINDINGS  nlohmann CACHE STRING "")
set(JINJA2CPP_USE_REGEX           std      CACHE STRING "")
set(JINJA2CPP_CXX_STANDARD        11       CACHE STRING "")
set(JINJA2CPP_BUILD_TESTS         OFF      CACHE BOOL "")
set(JINJA2CPP_BUILD_SHARED        OFF      CACHE BOOL "")
set(JINJA2CPP_INSTALL             OFF      CACHE BOOL "")
set(JINJA2CPP_STRICT_WARNINGS     OFF      CACHE BOOL "")

# Boost filesystem/regex：Jinja2Cpp 在子目录创建的 imported target 对顶层不可见，
# 顶层预建（仅链接库，头文件由 jinja2cpp 目标自带 include 路径）
find_library(BOOST_FILESYSTEM_LIB NAMES boost_filesystem
             HINTS ${BOOST_ROOT}/lib PATHS /lib /usr/lib)
find_library(BOOST_REGEX_LIB NAMES boost_regex
             HINTS ${BOOST_ROOT}/lib PATHS /lib /usr/lib)
if(NOT TARGET Boost::filesystem AND BOOST_FILESYSTEM_LIB)
    add_library(Boost::filesystem INTERFACE IMPORTED)
    set_target_properties(Boost::filesystem PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        INTERFACE_LINK_LIBRARIES "${BOOST_FILESYSTEM_LIB}")
endif()
if(NOT TARGET Boost::regex AND BOOST_REGEX_LIB)
    add_library(Boost::regex INTERFACE IMPORTED)
    set_target_properties(Boost::regex PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        INTERFACE_LINK_LIBRARIES "${BOOST_REGEX_LIB}")
endif()

add_subdirectory(${JINJA_THIRDPARTY}/jinja2cpp ${CMAKE_CURRENT_BINARY_DIR}/jinja2cpp-build)

# ---- RKNN3 封装层 ChatTemplate（demo / 板端程序通过此接口使用）----
add_library(chat_template_wrap STATIC src/ChatTemplate.cpp src/Log.cpp)
# header-only 依赖显式加入（与 render_driver 同因：不依赖 jinja2cpp target 传递）
target_include_directories(chat_template_wrap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JINJA_THIRDPARTY}/jinja2cpp/include
    ${JINJA_THIRDPARTY}/json
    ${JINJA_THIRDPARTY}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# ---- 合并静态库：libchat_template.a（jinja2cpp + chat_template_wrap）----
# 交付物：所有需要推送到板端的库 .o 合并成一个，供板端用户直接链接。
# jinja2cpp 用 std::regex（JINJA2CPP_USE_REGEX=std），不依赖 boost_regex；
# filesystem_handler.cpp 已从 jinja2cpp 源码裁剪（chat_template 渲染不需要文件系统），故不依赖 boost_filesystem；
# fmt / nonstd / nlohmann 均为 header-only。故仅需合并 jinja2cpp + chat_template_wrap。
set(MERGED_DIR ${CMAKE_CURRENT_BINARY_DIR}/merged)
set(MERGED_LIB ${CMAKE_CURRENT_BINARY_DIR}/libchat_template.a)
add_custom_command(
    OUTPUT ${MERGED_LIB}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${MERGED_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MERGED_DIR}
    COMMAND ${CMAKE_COMMAND} -E chdir ${MERGED_DIR} ${CMAKE_AR} -x $<TARGET_FILE:jinja2cpp>
    COMMAND ${CMAKE_COMMAND} -E chdir ${MERGED_DIR} ${CMAKE_AR} -x $<TARGET_FILE:chat_template_wrap>
    COMMAND ${CMAKE_COMMAND} -E chdir ${MERGED_DIR}
            sh -c "${CMAKE_STRIP} --strip-unneeded *.o; ${CMAKE_AR} -qcs ${MERGED_LIB} *.o; rm -f *.o"
    DEPENDS jinja2cpp chat_template_wrap
    COMMENT "Merge static libraries -> ${MERGED_LIB}"
    VERBATIM
)
add_custom_target(chat_template_merge DEPENDS ${MERGED_LIB})

# ---- render_driver（chat_template 渲染驱动，板端测试用；链接合并库）----
add_executable(render_driver demo/render_driver.cpp)
# header-only 依赖必须显式加入：jinja2cpp target 的 INTERFACE include 在部分平台
# （如 Linux x86 上 nonstd::expected-lite 别名未创建）不会传递，否则编译报缺头。
# include 写法是 <nonstd/*.hpp> / <nlohmann/json.hpp>，故加 thirdparty 根（覆盖 nonstd）
# 与 thirdparty/json（覆盖 nlohmann）；CMAKE_INCLUDE_PATH 只影响 find_path 不影响编译器。
target_include_directories(render_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JINJA_THIRDPARTY}/jinja2cpp/include
    ${JINJA_THIRDPARTY}/json
    ${JINJA_THIRDPARTY}
    ${CMAKE_CURRENT_BINARY_DIR}
)
# Threads::Threads（-lpthread）：Jinja2Cpp 用 std::shared_timed_mutex（pthread rwlock），
# Linux 必须显式链接（Cygwin 内置 pthread ）。
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(render_driver PRIVATE
    ${MERGED_LIB}
    Threads::Threads
)
add_dependencies(render_driver chat_template_merge)
# demo 可执行文件单独保持小：function-sections + gc-sections 裁剪
target_compile_options(render_driver PRIVATE -ffunction-sections -fdata-sections)
target_link_options(render_driver PRIVATE -Wl,--gc-sections)
if(CYGWIN)
    # Cygwin 下静态链接 libgcc/libstdc++，否则板端运行时缺少 libgcc_s_seh-1.dll / libstdc++-6.dll
    target_link_options(render_driver PRIVATE -static-libgcc -static-libstdc++)
endif()