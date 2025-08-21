/**
 * MainActivity.kt - Build Your Own Siri Android App
 * 
 * This is the main activity that handles:
 * - Voice recording and playback
 * - Speech-to-text using Whisper
 * - AI function calling using LLaMA 
 * - Function execution and result display
 * - Model initialization and lifecycle management
 */
package com.example.byos

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Represents a function call parsed from the LLM's JSON response
 * @param name The function name to execute (e.g., "toggle_flashlight")
 * @param arguments Key-value pairs of function parameters
 */
data class FunctionCall(
    val name: String,
    val arguments: Map<String, String>
)

/**
 * Main activity implementing the voice assistant interface
 * Implements LlamaCallback to receive async AI responses
 */
class MainActivity : AppCompatActivity(), LlamaCallback {
    // Audio recording components
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null
    
    // Model initialization status flags
    private var isWhisperLoaded = false  // Speech-to-text model ready
    private var isLlamaLoaded = false    // Language model ready

    // UI components
    private lateinit var btnRecord: Button  // Record/Stop voice input
    private lateinit var btnAsk: Button     // Process recorded audio
    private lateinit var tvPrompt: TextView // Shows transcribed text
    private lateinit var tvReply: TextView  // Shows function execution results

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Initialize native library bridge
        NativeBridge.load()
        setContentView(R.layout.activity_main)

        // Bind UI components
        btnRecord = findViewById(R.id.btnRecord)
        btnAsk    = findViewById(R.id.btnAsk)
        tvPrompt  = findViewById(R.id.tvPrompt)
        tvReply   = findViewById(R.id.tvReply)

        // Setup app functionality
        ensureMicPermission()
        initializeModels()
        setupButtons()
    }
    
    /**
     * Configure button click handlers
     */
    private fun setupButtons() {
        btnRecord.setOnClickListener { toggleRecording() }
        btnAsk.setOnClickListener { askModel() }
    }
    
    /**
     * Initialize AI models on background thread
     * - Copies models from assets if needed
     * - Initializes Whisper for speech-to-text
     * - Initializes LLaMA for function calling
     * - Sets up async callbacks for LLaMA results
     */
    private fun initializeModels() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Initialize Whisper speech-to-text model
                val whisperModel = setupWhisperModel()
                isWhisperLoaded = whisperModel?.let { 
                    NativeBridge.initWhisper(it.absolutePath) 
                } ?: false
                
                // Initialize LLaMA function calling model
                val llamaModel = setupLlamaModel()
                isLlamaLoaded = llamaModel?.let { model ->
                    // Force garbage collection before loading large model
                    System.gc()
                    Thread.sleep(500)
                    System.gc()
                    
                    try {
                        val success = NativeBridge.initLlama(model.absolutePath)
                        if (success) {
                            // Set up callback to receive async AI responses
                            NativeBridge.initAsyncCallback(this@MainActivity)
                        }
                        success
                    } catch (e: OutOfMemoryError) {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "❌ Out of memory loading model", Toast.LENGTH_LONG).show()
                        }
                        false
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "❌ LLaMA init error: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                        false
                    }
                } ?: false
                
                // Update UI with initialization results
                withContext(Dispatchers.Main) {
                    when {
                        isWhisperLoaded && isLlamaLoaded -> Toast.makeText(this@MainActivity, "✅ Models ready", Toast.LENGTH_SHORT).show()
                        isLlamaLoaded -> Toast.makeText(this@MainActivity, "⚠️ LLaMA ready, Whisper failed", Toast.LENGTH_LONG).show()
                        isWhisperLoaded -> Toast.makeText(this@MainActivity, "⚠️ Whisper ready, LLaMA failed", Toast.LENGTH_LONG).show()
                        else -> Toast.makeText(this@MainActivity, "❌ Models failed to load", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "❌ Model initialization error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    
    /**
     * Clean up resources when activity is destroyed
     * - Stop any ongoing recording
     * - Release audio resources
     * - Clean up native model resources
     */
    override fun onDestroy() {
        super.onDestroy()
        // Stop recording and release audio resources
        try {
            isRecording = false
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null
        
        // Clean up native callbacks and models
        NativeBridge.cleanupCallback()
        
        if (isLlamaLoaded || isWhisperLoaded) {
            NativeBridge.cleanup()
            System.gc()
        }
    }

    /**
     * Request microphone permission if not already granted
     */
    private fun ensureMicPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                0
            )
        }
    }

    /**
     * Write standard WAV file header for recorded audio
     * Required for Whisper to properly decode the audio format
     */
    private fun writeWavHeader(file: File, dataBytes: Int, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val totalDataLen = dataBytes + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val header = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(totalDataLen)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort((channels * bitsPerSample / 8).toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray())
        header.putInt(dataBytes)
        val raf = java.io.RandomAccessFile(file, "rw")
        raf.seek(0)
        raf.write(header.array())
        raf.close()
    }

    /**
     * Start/stop audio recording
     * Records in 16kHz mono 16-bit PCM format required by Whisper
     */
    private fun toggleRecording() {
        val out = File(cacheDir, "audio.wav")
        if (isRecording) {
            // Stop current recording
            try {
                isRecording = false
                try { audioRecord?.stop() } catch (_: Exception) {}
                try { audioRecord?.release() } catch (_: Exception) {}
                audioRecord = null
                recordingJob = null
            } catch (e: Exception) {
                Toast.makeText(this, "Stop failed: ${e.message}", Toast.LENGTH_SHORT).show()
            } finally {
                btnRecord.text = "Record"
            }
        } else {
            // Start new recording
            // Configure audio recording parameters (Whisper requirements)
            val sampleRate = 16000  // 16kHz sample rate
            val channelConfig = AudioFormat.CHANNEL_IN_MONO  // Mono channel
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT  // 16-bit samples
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                minBuf * 2
            )
            try {
                // Create output stream and write placeholder WAV header
                val fos = out.outputStream()
                val header = ByteArray(44)  // Standard WAV header size
                fos.write(header)

                // Start recording loop
                val buffer = ByteArray(minBuf)
                var totalBytes = 0
                audioRecord?.startRecording()
                isRecording = true
                btnRecord.text = "Stop"
                Toast.makeText(this@MainActivity, "Recording...", Toast.LENGTH_SHORT).show()

                // Background recording loop
                recordingJob = lifecycleScope.launch(Dispatchers.IO) {
                    while (isRecording) {
                        val n = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                        if (n > 0) {
                            fos.write(buffer, 0, n)
                            totalBytes += n
                        }
                    }
                    fos.flush()
                    fos.close()
                    // Write proper WAV header with actual data size
                    writeWavHeader(out, totalBytes, sampleRate, 1, 16)
                }
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Record failed: ${e.message}", Toast.LENGTH_LONG).show()
                audioRecord?.release()
                audioRecord = null
                isRecording = false
                try { recordingJob?.cancel() } catch (_: Exception) {}
                recordingJob = null
            }
        }
    }

    /**
     * Process recorded audio through the AI pipeline:
     * 1. Transcribe audio using Whisper
     * 2. Send transcript to LLaMA for function calling
     * 3. Execute the returned function and show results
     */
    private fun askModel() {
        if (!isWhisperLoaded && !isLlamaLoaded) {
            Toast.makeText(this, "❌ No models loaded", Toast.LENGTH_SHORT).show()
            return
        }
        
        lifecycleScope.launch(Dispatchers.IO) {
            val audioFile = File(cacheDir, "audio.wav")
            
            if (!audioFile.exists()) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "No recording found! Record first.", Toast.LENGTH_SHORT).show()
                }
                return@launch
            }

            try { recordingJob?.join() } catch (_: Exception) {}
            
            // Step 1: Convert speech to text using Whisper
            val transcript = if (isWhisperLoaded) {
                try {
                    withContext(Dispatchers.Main) {
                        tvPrompt.text = "🎙️ Transcribing..."
                    }
                    NativeBridge.transcribe(audioFile.path)
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "⚠️ Transcription error: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                    "⚠️ Transcription failed"
                }
            } else {
                "⚠️ Whisper not loaded"
            }

            // Step 2: Send transcript to LLaMA for function calling
            if (isLlamaLoaded) {
                try {
                    withContext(Dispatchers.Main) {
                        tvPrompt.text = transcript
                        tvReply.text = "Processing..."
                        btnAsk.isEnabled = false
                    }

                    // Run LLM inference asynchronously
                    val started = NativeBridge.runLlamaAsync(transcript)
                    if (!started) {
                        withContext(Dispatchers.Main) {
                            tvReply.text = "⚠️ Already processing"
                            btnAsk.isEnabled = true
                        }
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "⚠️ LLaMA error: ${e.message}", Toast.LENGTH_LONG).show()
                        btnAsk.isEnabled = true
                    }
                }
            } else {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "⚠️ LLaMA not loaded", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /**
     * Copy a model file from app assets to internal storage
     * Uses atomic copy with temporary file to avoid corruption
     */
    private fun copyAsset(assetPath: String): File {
        val filename = assetPath.substringAfterLast('/')
        val outFile = File(filesDir, filename)
        if (!outFile.exists() || outFile.length() == 0L) {
            val tmp = File(filesDir, "$filename.copying")
            if (tmp.exists()) try { tmp.delete() } catch (_: Exception) {}
            try {
                // Copy with 1MB buffer for efficiency
                assets.open(assetPath).use { input ->
                    tmp.outputStream().use { out ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            val n = input.read(buffer)
                            if (n <= 0) break
                            out.write(buffer, 0, n)
                        }
                        out.flush()
                    }
                }
                if (tmp.length() > 0L) {
                    if (outFile.exists()) try { outFile.delete() } catch (_: Exception) {}
                    if (!tmp.renameTo(outFile)) {
                        tmp.inputStream().use { `in` ->
                            outFile.outputStream().use { it.write(`in`.readBytes()) }
                        }
                        try { tmp.delete() } catch (_: Exception) {}
                    }
                } else {
                    try { tmp.delete() } catch (_: Exception) {}
                    throw IllegalStateException("Asset copy failed: $assetPath")
                }
            } catch (e: Exception) {
                try { tmp.delete() } catch (_: Exception) {}
                if (!outFile.exists() || outFile.length() == 0L) throw e
            }
        }
        return outFile
    }

    /**
     * Locate and setup Whisper speech-to-text model
     * Tries to find ggml-tiny.en.bin or any .bin file in assets/models/
     */
    private fun setupWhisperModel(): File? {
        return try {
            copyAsset("models/ggml-tiny.en.bin")
        } catch (_: Exception) {
            try {
                // Fallback: find any GGML model file
                val names = assets.list("models")?.toList().orEmpty()
                val cand = names.firstOrNull { it.endsWith(".bin", true) && it.contains("ggml", true) }
                cand?.let { copyAsset("models/$it") }
            } catch (_: Exception) {
                null
            }
        }
    }

    /**
     * Locate and setup LLaMA function calling model
     * Searches assets, then external storage for .gguf files
     * Falls back to showing user instructions if no model found
     */
    private fun setupLlamaModel(): File? {
        // Try to load model from app assets first
        val fromAssets: File? = try {
            copyAsset("models/unsloth.Q2_K.gguf")
        } catch (_: Exception) {
            try {
                // Fallback: find any .gguf file in assets
                val names = assets.list("models")?.toList().orEmpty()
                val cand = names.firstOrNull { it.endsWith(".gguf", true) }
                cand?.let { copyAsset("models/$it") }
            } catch (_: Exception) {
                null
            }
        }?.takeIf { it.length() > 0L }

        if (fromAssets != null && fromAssets.length() > 0L) return fromAssets

        // If no model in assets, check external storage
        val extDir = getExternalFilesDir(null)
        val extCandidates = mutableListOf<File>()
        if (extDir != null && extDir.isDirectory) {
            extCandidates += File(extDir, "unsloth.Q2_K.gguf")
            extDir.listFiles()?.filter { it.isFile && it.name.endsWith(".gguf", true) }?.let { extCandidates += it }
        }
        // Copy model from external storage to internal storage for faster access
        val ext = extCandidates.firstOrNull { it.exists() && it.length() > 0L }
        if (ext != null) {
            val dest = File(filesDir, ext.name)
            try {
                ext.inputStream().use { `in` ->
                    dest.outputStream().use { out ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            val n = `in`.read(buffer)
                            if (n <= 0) break
                            out.write(buffer, 0, n)
                        }
                        out.flush()
                    }
                }
                if (dest.length() > 0L) return dest
            } catch (_: Exception) {}
        }

        runOnUiThread {
            val hint = (extDir?.absolutePath ?: "<ext not available>")
            Toast.makeText(
                this,
                "Place a .gguf under: $hint and reopen the app",
                Toast.LENGTH_LONG
            ).show()
        }
        return fromAssets
    }
    
    /**
     * Callback called when LLaMA completes function call generation
     * Parses JSON response, executes the function, and displays result
     */
    override fun onLlamaResult(result: String) {
        runOnUiThread {
            try {
                Log.d("MainActivity", "Raw LLM result: $result")
                
                // Step 3: Fix common JSON formatting issues from LLM
                var fixedResult = result.trim()
                
                // Remove double commas that can occur from incomplete responses
                fixedResult = fixedResult.replace(",,", ",")
                
                if (fixedResult.startsWith("[") && !fixedResult.endsWith("]")) {
                    // Add missing closing bracket and brace
                    if (fixedResult.contains("\"name\"") && !fixedResult.contains("\"arguments\"")) {
                        // Remove trailing comma if present before adding arguments
                        if (fixedResult.endsWith(",")) {
                            fixedResult = fixedResult.dropLast(1)
                        }
                        fixedResult += ", \"arguments\": {}}]"
                    } else if (fixedResult.contains("\"arguments\"") && !fixedResult.endsWith("}")) {
                        fixedResult += "}]"
                    } else {
                        fixedResult += "]"
                    }
                } else if (fixedResult.startsWith("{") && !fixedResult.endsWith("}")) {
                    // Add missing closing brace
                    if (fixedResult.contains("\"name\"") && !fixedResult.contains("\"arguments\"")) {
                        // Remove trailing comma if present before adding arguments
                        if (fixedResult.endsWith(",")) {
                            fixedResult = fixedResult.dropLast(1)
                        }
                        fixedResult += ", \"arguments\": {}}"
                    } else {
                        fixedResult += "}"
                    }
                }
                
                Log.d("MainActivity", "Fixed result: $fixedResult")
                
                val funcCall: FunctionCall? = try {
                    val listType = object : TypeToken<List<FunctionCall>>() {}.type
                    val list: List<FunctionCall> = Gson().fromJson(fixedResult, listType) ?: emptyList()
                    list.firstOrNull()
                } catch (_: Exception) {
                    try {
                        Gson().fromJson<FunctionCall>(fixedResult, object : TypeToken<FunctionCall>() {}.type)
                    } catch (_: Exception) { 
                        Log.e("MainActivity", "Failed to parse JSON: $fixedResult")
                        null 
                    }
                }

                val finalResult = funcCall?.let {
                    Log.d("MainActivity", "Parsed function call: ${it.name} with args: ${it.arguments}")
                    try {
                        FunctionHandler.dispatch(this@MainActivity, it.name, it.arguments)
                    } catch (e: Exception) {
                        "⚠️ Dispatch error: ${e.message}"
                    }
                } ?: "⚠️ Could not parse response: $result"

                tvReply.text = finalResult
                btnAsk.isEnabled = true
            } catch (e: Exception) {
                tvReply.text = "⚠️ Callback error: ${e.message}"
                btnAsk.isEnabled = true
            }
        }
    }
}
