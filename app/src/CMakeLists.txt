cmake_minimum_required(VERSION 3.10)
project(native-lib)

# === IMPORTED SHARED LIBRARIES ===
# whisper.so
add_library( whisper SHARED IMPORTED )
set_target_properties( whisper PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libwhisper.so
)

# llama.so
add_library( llama SHARED IMPORTED )
set_target_properties( llama PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libllama.so
)

# ← NEW: the CPU backend that libggml.so expects
add_library( ggml_cpu  SHARED IMPORTED )
set_target_properties( ggml_cpu  PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libggml_cpu.so)

# ← NEW: the CPU backend that libggml.so expects
add_library( ggml_base  SHARED IMPORTED )
set_target_properties( ggml_base  PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libggml-base.so)


# ggml.so  ← new!
add_library( ggml SHARED IMPORTED )
set_target_properties( ggml PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libggml.so
)

# === YOUR NATIVE LIB ===
add_library( native-lib SHARED
        ${CMAKE_SOURCE_DIR}/main/cpp/native-lib.cpp
)

# headers
include_directories(
        ${CMAKE_SOURCE_DIR}/main/cpp/include
)

# Android log
find_library( log-lib log )

# link against whisper, llama, ggml and log
target_link_libraries(
        native-lib
        ggml_cpu
        ggml_base
        ggml
        whisper
        llama
        ${log-lib}
)
