# env_linux.sh — Linux 服务器（Linaro aarch64 交叉编译 / 本机 x86_64）

# aarch64（RK3588 板端）
RK_AARCH64_TOOLCHAIN=/opt/toolchains/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/
export C_COMPILER_AARCH64=${RK_AARCH64_TOOLCHAIN}/aarch64-linux-gnu-gcc
export CXX_COMPILER_AARCH64=${RK_AARCH64_TOOLCHAIN}/aarch64-linux-gnu-g++

# x86_64（服务器本机 / Linux x86 测试）
export C_COMPILER_X86_64=/usr/bin/gcc
export CXX_COMPILER_X86_64=/usr/bin/g++

# armhf（ARM 32 位硬浮点，armv7 板端）
ARMHF_TOOLCHAIN=/opt/toolchains/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/
export C_COMPILER_ARMHF=${ARMHF_TOOLCHAIN}/arm-linux-gnueabihf-gcc
export CXX_COMPILER_ARMHF=${ARMHF_TOOLCHAIN}/arm-linux-gnueabihf-g++
