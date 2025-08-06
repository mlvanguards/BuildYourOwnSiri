package com.example.byos

object NativeBridge {
    init {
        System.loadLibrary("ggml_cpu")
        System.loadLibrary("ggml_base")
        System.loadLibrary("ggml")
        System.loadLibrary("whisper")
        System.loadLibrary("llama")
        System.loadLibrary("native-lib")
    }
    fun load() = Unit  // call this in onCreate() to trigger the init block

    external fun initWhisper(modelPath: String): Boolean
    external fun transcribe(audioPath: String): String
    external fun initLlama(modelPath: String): Boolean
    external fun runLlama(prompt: String): String
}