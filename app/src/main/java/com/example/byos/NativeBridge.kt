package com.example.byos

object NativeBridge {
    init {
        // Load only our native library (stub implementation)
        System.loadLibrary("native-lib")
    }
    fun load() = Unit  // call this in onCreate() to trigger the init block

    external fun initWhisper(modelPath: String): Boolean
    external fun transcribe(audioPath: String): String
    external fun initLlama(modelPath: String): Boolean
    
    // Async methods
    external fun initAsyncCallback(callback: LlamaCallback): Boolean
    external fun runLlamaAsync(prompt: String): Boolean
    external fun cleanupCallback()
    
    // Cleanup method
    external fun cleanup()
}

interface LlamaCallback {
    fun onLlamaResult(result: String)
}