cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
include(cmake/set_ifndef.cmake)

project(TensorRT
        LANGUAGES CXX CUDA
        VERSION 8.2
        DESCRIPTION "TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators."
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT")

# C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")

find_package(Threads REQUIRED)

## find_package(CUDA) is broken for cross-compilation. Enable CUDA language instead.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_package(CUDA ${CUDA_VERSION} REQUIRED)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)
find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)
set(CUDA_LIBRARIES ${CUDART_LIB})


message(STATUS "CUBLAS_LIB: ${CUBLAS_LIB}")
message(STATUS "CUBLASLT_LIB: ${CUBLASLT_LIB}")
message(STATUS "CUDART_LIB: ${CUDART_LIB}")
message(STATUS "CUDNN_LIB: ${CUDNN_LIB}")

file(GLOB SRCS *.cpp)
set(PLUGIN_SOURCES ${PLUGIN_SOURCES} ${SRCS})
file(GLOB CU_SRCS *.cu)
set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} ${CU_SRCS})
file(GLOB COMMON_SRCS common/*.cpp)
set(COMMON_SOURCES ${COMMON_SOURCES} ${COMMON_SRCS})
file(GLOB COMMON_CU_SRCS common/kernels/*.cu)
set(COMMON_CU_SOURCES ${COMMON_CU_SOURCES} ${COMMON_CU_SRCS})

# Generate Gencode
if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      53
      60
      61
      70
      75
    )

  string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
  if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.1)
    list(APPEND GPU_ARCHS 86)
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")


include_directories(common common/kernels)
list(APPEND PLUGIN_CU_SOURCES "${COMMON_CU_SOURCES}")
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${GENCODES})
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")
list(APPEND PLUGIN_SOURCES "${COMMON_SOURCES}")

message(STATUS "PLUGIN_SOURCES: ${PLUGIN_SOURCES}")
message(STATUS "GENCODES: ${GENCODES}")

add_library(my_plugin SHARED
    ${PLUGIN_SOURCES}
)

target_include_directories(my_plugin
    PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes
)
target_include_directories(my_plugin
    PUBLIC /usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

target_link_libraries(my_plugin
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    nvinfer
)