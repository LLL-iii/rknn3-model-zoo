# aarch64
RK_AARCH64_TOOLCHAIN=/opt/toolchains/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/
export C_COMPILER_AARCH64=${RK_AARCH64_TOOLCHAIN}/aarch64-linux-gnu-gcc
export CXX_COMPILER_AARCH64=${RK_AARCH64_TOOLCHAIN}/aarch64-linux-gnu-g++


# armhf
RK_ARM32_TOOLCHAIN=~/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/
export C_COMPILER_ARM32=${RK_ARM32_TOOLCHAIN}/arm-linux-gnueabihf-gcc
export CXX_COMPILER_ARM32=${RK_ARM32_TOOLCHAIN}/arm-linux-gnueabihf-g++


# x86_64
RK_X86_64_TOOLCHAIN=/usr/bin/
export C_COMPILER_X86_64=${RK_X86_64_TOOLCHAIN}/gcc
export CXX_COMPILER_X86_64=${RK_X86_64_TOOLCHAIN}/g++