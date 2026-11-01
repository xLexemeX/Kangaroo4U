cmake_minimum_required(VERSION 3.18)

# Builds the solver into ../bin on both Linux (gcc + nvcc) and Windows
# (Visual Studio Build Tools + nvcc). Needs the CUDA Toolkit, 12.x recommended.
# If CMake cannot find nvcc, pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc.

project(rckangaroo LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_executable(rckangaroo
    RCKangaroo.cpp
    GpuKang.cpp
    Ec.cpp
    utils.cpp
    ecfast.cpp
    CallCubin.cpp
    RCGpuCore.cu
)

target_include_directories(rckangaroo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${CUDAToolkit_INCLUDE_DIRS}
)

# The arch list is spelled out below, so CMake must not add its own.
set_target_properties(rckangaroo PROPERTIES CUDA_ARCHITECTURES OFF)

# PTX for each family; the driver JITs it for the card it finds, including newer
# ones. 40xx and 50xx cards run the hand-written turbo cubins instead.
target_compile_options(rckangaroo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_89,code=compute_89>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=compute_86>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=compute_75>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=compute_61>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

if(MSVC)
    target_compile_definitions(rckangaroo PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

target_link_libraries(rckangaroo PRIVATE
    CUDA::cudart_static
    CUDA::cuda_driver
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
if(UNIX AND NOT APPLE)
    target_link_libraries(rckangaroo PRIVATE rt)
endif()

# One output folder for every generator. The empty generator expression stops
# Visual Studio from adding a Release/ or Debug/ subfolder.
set(K4U_BIN "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
set_target_properties(rckangaroo PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    RUNTIME_OUTPUT_DIRECTORY "${K4U_BIN}$<0:>"
)

# The turbo cubins are loaded from the binary's own folder.
add_custom_command(TARGET rckangaroo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel_sm89.cubin"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel_sm120.cubin"
        "$<TARGET_FILE_DIR:rckangaroo>"
)
