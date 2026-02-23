cmake_minimum_required(VERSION 3.20)
project(gpu_microkernel LANGUAGES CXX CUDA)

# ── CUDA Configuration ────────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Detect GPU architecture automatically
include(FindCUDA)
set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")  # Volta, Ampere, Ada, Hopper

# ── Required CUDA features ────────────────────────────────────────────────────
# Dynamic Parallelism (CDP) requires:
#   1. Separate compilation (-rdc=true)
#   2. Linking against cudadevrt
#   3. sm >= 3.5

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ── Source files ──────────────────────────────────────────────────────────────
set(CUDA_SOURCES
    cuda/src/microkernel.cu
)

set(HOST_SOURCES
    cuda/src/host_launcher.cpp
)

# ── Include directories ───────────────────────────────────────────────────────
include_directories(
    cuda/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ── Executable target ─────────────────────────────────────────────────────────
add_executable(microkernel ${CUDA_SOURCES} ${HOST_SOURCES})

target_compile_options(microkernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        # Dynamic Parallelism
        --device-c
        # Fast math (unsafe optimizations for better perf)
        --use_fast_math
        # Verbose PTX for analysis
        -Xptxas=-v
        # Register limit: leave headroom for OS thread cooperation
        -maxrregcount=128
        # Line info for cuda-gdb
        -lineinfo
        # Warnings as errors (catch warp divergence issues early)
        -Werror cross-execution-space-call
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -Wall
        -Wextra
    >
)

# ── Link cudadevrt for Dynamic Parallelism ────────────────────────────────────
target_link_libraries(microkernel
    PRIVATE
    cudadevrt          # Device-side CUDA runtime (required for CDP)
    cuda               # CUDA driver API
)

set_target_properties(microkernel PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Final link step for CDP
)

# ── Profiling / Analysis Targets ──────────────────────────────────────────────
# Run with: cmake --build . --target profile
add_custom_target(profile
    COMMAND ncu
        --set full
        --target-processes all
        --import-source yes
        --export ${CMAKE_BINARY_DIR}/profile_report
        $<TARGET_FILE:microkernel>
    DEPENDS microkernel
    COMMENT "Running NVIDIA Nsight Compute profiler"
)

add_custom_target(memcheck
    COMMAND compute-sanitizer
        --tool memcheck
        --check-device-heap yes
        $<TARGET_FILE:microkernel>
    DEPENDS microkernel
    COMMENT "Running CUDA memory sanitizer"
)

add_custom_target(racecheck
    COMMAND compute-sanitizer
        --tool racecheck
        $<TARGET_FILE:microkernel>
    DEPENDS microkernel
    COMMENT "Running CUDA race condition checker"
)

# ── PTX inspection ────────────────────────────────────────────────────────────
add_custom_target(ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
        -arch=sm_80
        -ptx
        ${CMAKE_SOURCE_DIR}/cuda/src/microkernel.cu
        -I ${CMAKE_SOURCE_DIR}/cuda/include
        -o ${CMAKE_BINARY_DIR}/microkernel.ptx
        --device-c
        --use_fast_math
    COMMENT "Generating PTX for SM 8.0"
)
