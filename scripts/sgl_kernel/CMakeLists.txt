cmake_minimum_required(VERSION 3.26)
project(sgl_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(Python  REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch   REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(FLASHINFER_DIR "${CMAKE_SOURCE_DIR}/flashinfer" CACHE PATH "FlashInfer checkout")
if(NOT EXISTS "${FLASHINFER_DIR}/csrc")
  message(FATAL_ERROR "FlashInfer sources missing at ${FLASHINFER_DIR}")
endif()

set(CUTLASS_DIR "${FLASHINFER_DIR}/3rdparty/cutlass")
if(NOT EXISTS "${CUTLASS_DIR}/include")
  message(FATAL_ERROR "Cutlass headers missing inside FlashInfer (${CUTLASS_DIR})")
endif()

set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/csrc
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/common
    ${FLASHINFER_DIR}/include
    ${FLASHINFER_DIR}/csrc
)

set(COMMON_SOURCES
    csrc/common_extension.cc
    csrc/attention/cascade.cu
    csrc/attention/lightning_attention_decode_kernel.cu
    csrc/attention/merge_attn_states.cu
    csrc/elementwise/activation.cu
    csrc/elementwise/copy.cu
    csrc/elementwise/fused_add_rms_norm_kernel.cu
    csrc/elementwise/rope.cu
    csrc/elementwise/topk.cu
    csrc/kvcacheio/transfer.cu
    csrc/memory/store.cu
    csrc/speculative/packbit.cu
    csrc/speculative/speculative_sampling.cu
)

Python_add_library(common_ops MODULE WITH_SOABI ${COMMON_SOURCES})

set(CUDA_FLAGS
    -gencode=arch=compute_89,code=sm_89
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -use_fast_math
    -Xcompiler=-fPIC
    -Xcompiler=-Wno-terminate
    -Xcompiler=-Wno-deprecated-declarations
    -Xcompiler=-Wno-conversion
    -Xcompiler=-Wfatal-errors
    -Xcompiler=-ftemplate-backtrace-limit=1
    -Xcudafe=--diag_suppress=177
    -Xcudafe=--diag_suppress=2361
)

target_include_directories(common_ops PRIVATE ${COMMON_INCLUDE_DIRS})
target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

target_compile_definitions(common_ops PRIVATE
    NDEBUG
    OPERATOR_NAMESPACE=sgl-kernel
    FLASHINFER_ENABLE_F16
    CUTE_USE_PACKED_TUPLE=1
    CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    CUTLASS_VERSIONS_GENERATED
    CUTLASS_TEST_LEVEL=0
    CUTLASS_TEST_ENABLE_CACHED_RESULTS=1
    CUTLASS_DEBUG_TRACE_LEVEL=0
    SGL_BUILD_ALLREDUCE=0
    SGL_BUILD_MSCCLPP=0
    SGL_BUILD_MOE=0
)

target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

set_target_properties(common_ops PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sm89"
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel/sm89)
