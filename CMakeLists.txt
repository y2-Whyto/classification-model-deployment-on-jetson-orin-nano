cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER /usr/bin/clang++)

set(CMAKE_CXX_STANDARD 17)  # for <[experimental/]filesystem> include

project(test CXX)

set(CMAKE_CXX_FLAGS "-fstandalone-debug -fuse-ld=lld")  # for debug, suppressing optimization

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})  # generate executable at project dir

# NVIDIA, CUDA, TensorRT, OpenCV...
set(CUDA_HOME /usr/local/cuda)
if (DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
endif ()
find_package(OpenCV REQUIRED)
include_directories(
        ${CUDA_HOME}/include
        ${CUDA_HOME}/targets/aarch64-linux/include
#        /usr/src/tensorrt/samples/common  # 已弃用
        ${OpenCV_INCLUDE_DIRS}
)
link_directories(
        ${CUDA_HOME}/lib64
        ${CUDA_HOME}/targets/aarch64-linux/lib
        ${OpenCV_LIB_DIR}
)
link_libraries(
        stdc++fs  # for <[experimental/]filesystem> include
        cudart
        nvinfer
        ${OpenCV_LIBS}
)

add_executable(test test.cc)
