#!/bin/bash
# build.sh — chat_template 引擎（Jinja2Cpp）统一构建入口
#
# 依赖源码在 thirdparty/（jinja2cpp / json / fmt / nonstd），Boost 见下方说明。
# 构建产物 + 板端测试数据整合到 install/<sdk>_<platform>/，可直接推送到板端。
#
# 用法：
#   bash build.sh [-s <system>] [-a <arch>] [-b <build_type>] [-n <sdk>]
#     默认构建 linux_x86（bash build.sh 即等价于 -s linux -a x86 -b Release）
#     -s system : linux / android / cygwin
#     -a arch   : linux: aarch64/x86 ; android: arm64-v8a/armeabi-v7a ; cygwin: x86
#     -b type   : Debug / Release / RelWithDebInfo
#     -n sdk    : 默认 chat_template
#
# 示例：
#   Linux x86 服务器（默认）:      bash build.sh
#   Linux 板端（RK3588）:         bash build.sh -s linux -a aarch64
#   Android 板端（RK3576）:      bash build.sh -s android -a arm64-v8a
#   Windows 本机（Cygwin）:      bash build.sh -s cygwin -a x86
#
# Boost 说明：Jinja2Cpp 依赖 boost（filesystem/regex/system 编译库）。
#   - cygwin：使用 Cygwin 系统 Boost（无需 BOOST_ROOT）
#   - linux/android：需要 aarch64/Android 交叉编译的 Boost，通过 BOOST_ROOT 指定
#     （若无 BOOST_ROOT 指定, build.sh 默认从 thirdparty/boost 编译，产物在 build/boost_<platform>）。
set -e

TARGET_SDK=chat_template
ENABLE_ASAN=OFF
TARGET_SYSTEM=linux
TARGET_ARCH=x86
BUILD_TYPE=Release
echo "$0 $@"
while getopts "s:a:b:n:" opt; do
  case $opt in
    n) TARGET_SDK=$OPTARG ;;
    s) TARGET_SYSTEM=$OPTARG ;;
    a) TARGET_ARCH=$OPTARG ;;
    b) BUILD_TYPE=$OPTARG ;;
    ?)
      echo "Usage: $0 -s <system> -a <arch> -b <build_type> -n <sdk>"
      echo "  system: linux / android / cygwin"
      echo "  arch  : linux: aarch64/x86 ; android: arm64-v8a/armeabi-v7a ; cygwin: x86"
      exit -1 ;;
  esac
done

if [ x"${TARGET_SYSTEM}" != x"linux" ] && [ x"${TARGET_SYSTEM}" != x"android" ] && [ x"${TARGET_SYSTEM}" != x"cygwin" ]; then
  echo "ERROR: -s 必须是 linux/android/cygwin"; exit 1
fi

# 平台环境
if [ "${TARGET_SYSTEM}" == "linux" ]; then     source env_linux.sh; fi
if [ "${TARGET_SYSTEM}" == "android" ]; then   source env_android.sh; fi
if [ "${TARGET_SYSTEM}" == "cygwin" ]; then    source env_cygwin.sh; fi

# Android 默认 arm64-v8a
if [ "${TARGET_SYSTEM}" == "android" ] && [ -z "${TARGET_ARCH}" ]; then TARGET_ARCH=arm64-v8a; fi

TARGET_PLATFORM=${TARGET_SYSTEM}
if [[ -n ${TARGET_ARCH} ]]; then TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}; fi

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_SDK}_${TARGET_PLATFORM}
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SDK}_${TARGET_PLATFORM}_${BUILD_TYPE}

echo "==================================="
echo "TARGET_SYSTEM=${TARGET_SYSTEM}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "TARGET_SDK=${TARGET_SDK}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "BOOST_ROOT=${BOOST_ROOT:-<system>}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "==================================="

# ── Boost（优先 thirdparty/boost 源码，其次 BOOST_ROOT，cygwin 兜底系统）─────
BOOST_TP=$ROOT_PWD/thirdparty/boost
BOOST_PREFIX=$ROOT_PWD/build/boost_${TARGET_PLATFORM}

build_boost_thirdparty() {
  # 已构建则跳过（缓存）
  if [ -f "$BOOST_PREFIX/lib/libboost_filesystem.a" ] || [ -f "$BOOST_PREFIX/lib/libboost_filesystem.dll.a" ] || [ -f "$BOOST_PREFIX/lib/libboost_filesystem.so" ]; then
    echo "  [Boost] 已构建: $BOOST_PREFIX"
    return 0
  fi
  echo "==> [Boost] 从 thirdparty/boost 编译 -> $BOOST_PREFIX"
  mkdir -p "$ROOT_PWD/build"
  cd "$BOOST_TP"
  chmod +x bootstrap.sh 2>/dev/null || true
  chmod +x tools/build/src/engine/build.sh 2>/dev/null || true
  find tools -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
  if [ ! -x b2 ]; then
    ./bootstrap.sh --prefix="$BOOST_PREFIX" >"$ROOT_PWD/build/boost_bootstrap.log" 2>&1 \
      || { echo "ERROR: boost bootstrap 失败，见 build/boost_bootstrap.log"; return 1; }
  fi
  local b2_args=(toolset=gcc cxxflags="-Os -std=c++11")
  local b2_extra=()
  if [ "${TARGET_SYSTEM}" == "linux" ] && [ "${TARGET_ARCH}" == "aarch64" ]; then
    # aarch64 交叉编译：注册交叉编译器（user-config.jam），architecture 用 arm + address-model=64
    mkdir -p "$ROOT_PWD/build"
    cat > "$ROOT_PWD/build/user-config-aarch64.jam" <<EOF
using gcc : arm : ${CXX_COMPILER_AARCH64} ;
EOF
    b2_extra=(--user-config="$ROOT_PWD/build/user-config-aarch64.jam")
    b2_args=(toolset=gcc-arm cxxflags="-Os -std=c++11" architecture=arm address-model=64 \
             binary-format=elf abi=aapcs target-os=linux)
  elif [ "${TARGET_SYSTEM}" == "android" ]; then
    # Android 交叉编译：NDK clang（toolset=clang-android），architecture=arm + address-model
    case "$TARGET_ARCH" in
      arm64-v8a)   TRIPLE=aarch64-linux-android;     ADDR=64 ;;
      armeabi-v7a) TRIPLE=armv7a-linux-androideabi; ADDR=32 ;;
      *) echo "ERROR: 不支持 Android ABI=$TARGET_ARCH"; return 1 ;;
    esac
    local clang_cxx="$ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/${TRIPLE}26-clang++"
    if [ ! -x "$clang_cxx" ]; then
      echo "ERROR: 找不到 NDK clang: $clang_cxx（检查 ANDROID_NDK_PATH）"; return 1
    fi
    mkdir -p "$ROOT_PWD/build"
    cat > "$ROOT_PWD/build/user-config-android.jam" <<EOF
using clang : android : ${clang_cxx} ;
EOF
    b2_extra=(--user-config="$ROOT_PWD/build/user-config-android.jam")
    b2_args=(toolset=clang-android cxxflags="-Os -std=c++11 -DANDROID -fPIC" \
             architecture=arm address-model=${ADDR} target-os=android)
  fi
  ./b2 "${b2_extra[@]}" "${b2_args[@]}" variant=release \
      --with-filesystem --with-regex --with-system \
      link=static runtime-link=static threading=multi \
      -j"$(nproc 2>/dev/null || echo 4)" install --prefix="$BOOST_PREFIX" \
      >"$ROOT_PWD/build/boost_build.log" 2>&1 \
      || { echo "ERROR: boost 编译失败，见 build/boost_build.log"; return 1; }
  echo "  [Boost] 完成: $BOOST_PREFIX/lib"
  return 0
}

CMAKE_BOOST_FLAGS=()
if [ -d "$BOOST_TP" ]; then
  build_boost_thirdparty || exit 1
  CMAKE_BOOST_FLAGS=(-DBOOST_ROOT=$BOOST_PREFIX)
elif [ -n "${BOOST_ROOT}" ]; then
  CMAKE_BOOST_FLAGS=(-DBOOST_ROOT=${BOOST_ROOT})
  echo "  [Boost] 使用 BOOST_ROOT=$BOOST_ROOT"
else
  if [ "${TARGET_SYSTEM}" != "cygwin" ]; then
    echo "ERROR: 无 thirdparty/boost 且未设置 BOOST_ROOT，linux/android 无法构建"
    exit 1
  fi
  echo "  [Boost] 使用 Cygwin 系统 Boost"
fi

# ── CMake 配置 ─────────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"
if [[ -d "${INSTALL_DIR}" ]]; then rm -rf "${INSTALL_DIR}"; fi

# 依赖头文件优先从 thirdparty 解析。
# 关键：find_hdr_package 的 NAMES 形如 nonstd/expected.hpp、fmt/format.h，
#       故 CMAKE_INCLUDE_PATH 必须指向 thirdparty 根（使 thirdparty/nonstd/expected.hpp 成立），
#       不能指向 nonstd 子目录本身。CMake 路径列表用 ';' 分隔。
CMAKE_INCLUDE_PATH="${ROOT_PWD}/thirdparty"

COMMON_FLAGS=(
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
  -DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}
  # Android toolchain 默认 FIND_ROOT_PATH_MODE_INCLUDE/LIBRARY=ONLY（find_path/find_library
  # 只在 sysroot 搜），需允许搜索 CMAKE_INCLUDE_PATH（thirdparty 头文件）与 BOOST_ROOT/lib
  # （交叉编译的 boost 静态库），对其它平台无副作用
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH
  "${CMAKE_BOOST_FLAGS[@]}"
)

cd "${BUILD_DIR}"

if [ "${TARGET_SYSTEM}" == "linux" ]; then
  if [ "${TARGET_ARCH}" == "aarch64" ]; then
    cmake ../.. \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_C_COMPILER=${C_COMPILER_AARCH64} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER_AARCH64} \
      "${COMMON_FLAGS[@]}"
  else  # x86
    cmake ../.. \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_C_COMPILER=${C_COMPILER_X86_64} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER_X86_64} \
      "${COMMON_FLAGS[@]}"
  fi
elif [ "${TARGET_SYSTEM}" == "android" ]; then
  cmake ../.. \
    -DCMAKE_SYSTEM_NAME=Android \
    -DANDROID_PLATFORM=android-26 \
    -DANDROID_ABI=${TARGET_ARCH} \
    -DANDROID_STL=c++_static \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
    "${COMMON_FLAGS[@]}"
else  # cygwin
  cmake ../.. \
    -DCMAKE_SYSTEM_NAME=CYGWIN \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    "${COMMON_FLAGS[@]}"
fi

# ── 构建 ───────────────────────────────────────────────────────────────────
make -j"$(nproc 2>/dev/null || echo 4)" render_driver

# ── install 整合（可推送板端的完整包）────
echo ""
echo "==> 整合 install 推送包 -> ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/demo" "${INSTALL_DIR}/include" "${INSTALL_DIR}/lib"
cp "${BUILD_DIR}/render_driver" "${INSTALL_DIR}/demo/"
cp "${BUILD_DIR}/libchat_template.a" "${INSTALL_DIR}/lib/"
strip "${INSTALL_DIR}/demo/render_driver" 2>/dev/null || true

# 头文件：仅 RKNN3 封装层 ChatTemplate.h（pimpl 自包含，用户程序只需它 + lib 即可）
cp "${ROOT_PWD}/include/ChatTemplate.h" "${INSTALL_DIR}/include/"

echo ""
echo "==> 完成！安装产物："
find "${INSTALL_DIR}" -type f | sed "s|${INSTALL_DIR}/||"
