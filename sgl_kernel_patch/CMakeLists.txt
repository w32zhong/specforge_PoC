cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sgl-kernel LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
clear_cuda_arches(CMAKE_FLAG)

# Enables colored output for compiler diagnostics
set(CMAKE_COLOR_DIAGNOSTICS ON)
# the build system will print every command executed
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
# add -fPIC flag, for creating .so to be loaded at any memory address
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Default: `libmylibrary.so`, with this setting: `mylibrary.so`
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# set libc C++ ABI according to TORCH C++ ABI
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
# switching from whole-program compilation to separate compilation
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Torch
find_package(Torch REQUIRED)

# Cutlass (feteched from GitHub)
include(FetchContent)
FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        57e3cfb47a2d9e0d46eb6335c3dc411498efa198
    GIT_SHALLOW    ON
)
# this will download and set `repo-cutlass_SOURCE_DIR`.
FetchContent_Populate(repo-cutlass)

# FlashInfer (FLASHINFER_DIR needs to be passed in!)
if(NOT EXISTS "${FLASHINFER_DIR}/csrc")
    message(FATAL_ERROR "FlashInfer sources missing at ${FLASHINFER_DIR}")
endif()

# apache-tvm-ffi headers (needed here to build FlashInfer bindings).
set(TVM_INCLUDE_DIRS "")
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import json; import tvm_ffi.libinfo as L; print(json.dumps([L.find_include_path(), L.find_dlpack_include_path()]))"
    OUTPUT_VARIABLE TVM_INCLUDE_JSON
    RESULT_VARIABLE TVM_INCLUDE_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (TVM_INCLUDE_RESULT EQUAL 0 AND NOT TVM_INCLUDE_JSON STREQUAL "")
    string(JSON TVM_INCLUDE_DIR_0 GET "${TVM_INCLUDE_JSON}" 0)
    string(JSON TVM_INCLUDE_DIR_1 GET "${TVM_INCLUDE_JSON}" 1)
    list(APPEND TVM_INCLUDE_DIRS "${TVM_INCLUDE_DIR_0}" "${TVM_INCLUDE_DIR_1}")
    list(REMOVE_DUPLICATES TVM_INCLUDE_DIRS)
else()
    message(FATAL_ERROR "Failed to locate apache-tvm-ffi headers. Ensure it is installed.")
endif()

# C header files
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${FLASHINFER_DIR}/include
    ${FLASHINFER_DIR}/csrc
    ${TVM_INCLUDE_DIRS}
)

# CFLAGS
set(SGL_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=sgl-kernel"
    "-O3"
    "-Xcompiler" # It tells nvcc to pass the following options to CXX:
    "-fPIC"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16" # enable half precision
    "-DCUTE_USE_PACKED_TUPLE=1" # store tuples within CUTE in a packed format
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1" # enable tensor core in CUTLASS
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0" # no testing in CUTLASS
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0" # disable debug tracing for CUTLASS
    "--expt-relaxed-constexpr" # allow a more relaxed C++ constexpr rules in nvcc
    "--expt-extended-lambda" # allow C++20 lambda syntax in nvcc

    "-Xcompiler=-Wno-clang-format-violations"
    "-Xcompiler=-Wno-conversion"
    "-Xcompiler=-Wno-deprecated-declarations"
    "-Xcompiler=-Wno-terminate"
    "-Xcompiler=-Wfatal-errors"
    "-Xcompiler=-ftemplate-backtrace-limit=1"
    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
    "-Xcudafe=--diag_suppress=2361"  # invalid narrowing conversion from "char" to "signed char"
    # uncomment to debug
    # "--ptxas-options=-v"
    # "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
)
set(SGL_KERNEL_COMPILE_THREADS 8 CACHE STRING "Set compilation threads.")
list(APPEND SGL_KERNEL_CUDA_FLAGS
    "--threads=${SGL_KERNEL_COMPILE_THREADS}"
)

# enable BF16
list(APPEND SGL_KERNEL_CUDA_FLAGS
    "-DFLASHINFER_ENABLE_BF16"
)

# C source files
set(SOURCES
    "csrc/common_extension.cc"

    "csrc/attention/cascade.cu"
    "csrc/attention/lightning_attention_decode_kernel.cu"
    "csrc/attention/cutlass_mla_kernel.cu"
    "csrc/attention/merge_attn_states.cu"

    "csrc/elementwise/activation.cu"
    "csrc/elementwise/cast.cu"
    "csrc/elementwise/copy.cu"
    "csrc/elementwise/concat_mla.cu"
    "csrc/elementwise/fused_add_rms_norm_kernel.cu"
    "csrc/elementwise/rope.cu"
    "csrc/elementwise/topk.cu"

    "csrc/moe/moe_align_kernel.cu"
    "csrc/moe/moe_fused_gate.cu"
    "csrc/moe/moe_sum_reduce.cu"
    "csrc/moe/moe_topk_softmax_kernels.cu"
    "csrc/moe/prepare_moe_input.cu"

    "csrc/memory/store.cu"
    "csrc/kvcacheio/transfer.cu"

    "csrc/speculative/eagle_utils.cu"
    "csrc/speculative/ngram_utils.cu"
    "csrc/speculative/packbit.cu"
    "csrc/speculative/speculative_sampling.cu"

    "${FLASHINFER_DIR}/csrc/norm.cu"
    "${FLASHINFER_DIR}/csrc/renorm.cu"
    "${FLASHINFER_DIR}/csrc/sampling.cu"
)

# GPU arch-dependent configurations (determined by TORCH_CUDA_ARCH_LIST)
if(TORCH_CUDA_ARCH_LIST)
    foreach(_dot_arch IN LISTS TORCH_CUDA_ARCH_LIST)
        # Remove the dot from X.Y to get the compute_XY / sm_XY format
        string(REPLACE "." "" _int_arch "${_dot_arch}")

        # name our build target for this arch
        set(TARGET_NAME "common_ops_sm${_int_arch}_build")

        # arch-dependent configurations
        set(_cuda_flags ${SGL_KERNEL_CUDA_FLAGS})

        # Add 'a' suffix for specific architectures (e.g., "120a")
        if(_dot_arch STREQUAL "9.0")
            set(_specific_arch "${_int_arch}a")
            set(_use_fast_math 1)

        elseif(_dot_arch STREQUAL "12.0")
            set(_specific_arch "${_int_arch}a")
            set(_use_fast_math 0)

        elseif(_dot_arch STREQUAL "8.9")
            set(_specific_arch "${_int_arch}")
            set(_use_fast_math 0)

        else()
            message(FATAL_ERROR "Unsupported TORCH_CUDA_ARCH_LIST entry: ${_dot_arch}")
        endif()

        # Construct the -gencode flag just for this target
        list(APPEND _cuda_flags
            "-gencode=arch=compute_${_specific_arch},code=sm_${_specific_arch}"
        )

        # call Python_add_library provided by `scikit-build` to build dynamic lib with
        # Stable Python ABI from SOURCES
        Python_add_library(${TARGET_NAME} MODULE USE_SABI
            ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

        # enable/disable fast math (must follow Python_add_library)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            USE_FAST_MATH=${_use_fast_math}
        )

        # additional C headers
        target_include_directories(${TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc
            ${repo-cutlass_SOURCE_DIR}/examples/common
            ${repo-cutlass_SOURCE_DIR}/examples/77_blackwell_fmha
        )

        # add a CMake generator expression to add our CUDA CFLAGS only to .cu files
        message(STATUS "SGL CUDA flags for ${TARGET_NAME}: ${_cuda_flags}")
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${_cuda_flags}>
        )

        set_target_properties(${TARGET_NAME} PROPERTIES
            OUTPUT_NAME "common_ops"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm${_int_arch}"
        )

        # link libraries
        target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBRARIES}
                              c10 cuda cublas cublasLt)

        # specify install directory
        install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION sgl_kernel/sm${_int_arch})

    endforeach()

else()
    message(FATAL_ERROR "TORCH_CUDA_ARCH_LIST is not set.")
endif()
