cmake_minimum_required(VERSION 3.18)
project(native-lib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer static libraries for bundled third-party deps (llama, whisper, ggml)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Platform specific settings for Android
if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    
    # RELEASE BUILD OPTIMIZATIONS for mobile performance
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -fno-finite-math-only -funroll-loops -DNDEBUG -march=armv8.2-a+dotprod")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -fno-finite-math-only -funroll-loops -DNDEBUG -march=armv8.2-a+dotprod")
    
    # Enable NEON for ARM optimization
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_CPU_HAS_ARM=ON")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGGML_CPU_HAS_ARM=ON")
endif()

# Disable GGML Metal and other backends not needed for Android
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENGL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
# Ensure CPU backend is linked statically into ggml (avoid runtime dlopen of ggml-cpu)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)

# === BUILD REAL AI LIBRARIES ===

# Enable llama.cpp first so its ggml is used by the whole project
add_subdirectory(${CMAKE_SOURCE_DIR}/main/cpp/llama llama_build)

# Reuse the existing ggml from llama for whisper
set(WHISPER_USE_SYSTEM_GGML ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/main/cpp/whisper whisper_build)

# === YOUR NATIVE LIB ===
add_library( native-lib SHARED
        ${CMAKE_SOURCE_DIR}/main/cpp/native-lib.cpp
)

# Include directories
target_include_directories(native-lib PRIVATE
        ${CMAKE_SOURCE_DIR}/main/cpp/include
        ${CMAKE_SOURCE_DIR}/main/cpp/whisper/include
        ${CMAKE_SOURCE_DIR}/main/cpp/llama/include
)

# Android log
find_library( log-lib log )

# Add OpenMP definitions (clean, no duplicates)
target_compile_definitions(native-lib PRIVATE 
    GGML_USE_OPENMP=1 
    GGML_CPU_HAS_ARM=1
)

# Link REAL AI libraries
target_link_libraries(
        native-lib
        whisper
        llama
        ggml
        ${log-lib}
)

# Ensure 16 KiB page size compatibility for Android 15+ (align PT_LOAD segments)
if(ANDROID)
    # lld honors these -z flags to control segment/page alignment
    target_link_options(native-lib PRIVATE
        -Wl,-z,max-page-size=16384
        -Wl,-z,common-page-size=16384
    )
endif()
