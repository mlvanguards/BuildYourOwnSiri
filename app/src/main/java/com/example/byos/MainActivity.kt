package com.example.byos

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Bundle
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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class FunctionCall(
    val name: String,
    val arguments: Map<String, String>
)

class MainActivity : AppCompatActivity() {
    private lateinit var whisperFile: File
    private lateinit var llamaFile: File
    private var recorder: MediaRecorder? = null
    private var isRecording = false

    private lateinit var btnRecord: Button
    private lateinit var btnAsk: Button
    private lateinit var tvPrompt: TextView
    private lateinit var tvReply: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 1) load native lib
        NativeBridge.load()

        setContentView(R.layout.activity_main)

        btnRecord = findViewById(R.id.btnRecord)
        btnAsk    = findViewById(R.id.btnAsk)
        tvPrompt  = findViewById(R.id.tvPrompt)
        tvReply   = findViewById(R.id.tvReply)

        ensureMicPermission()

        whisperFile = copyAsset("models/ggml-tiny.en.bin")
        llamaFile   = copyAsset("models/unsloth.Q4_K_M.gguf")

        // 2) init models
        if (!NativeBridge.initWhisper(whisperFile.path)) {
            Toast.makeText(this, "❌ Whisper init failed", Toast.LENGTH_LONG).show()
        }
        if (!NativeBridge.initLlama(llamaFile.path)) {
            Toast.makeText(this, "❌ LLaMA init failed", Toast.LENGTH_LONG).show()
        }

        btnRecord.setOnClickListener { toggleRecording() }
        btnAsk   .setOnClickListener { askModel()      }
    }

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

    private fun toggleRecording() {
        val out = File(cacheDir, "audio.wav")
        if (isRecording) {
            // stop
            try {
                recorder?.apply {
                    stop()
                    release()
                }
                Toast.makeText(this, "Saved: ${out.absolutePath}", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Stop failed: ${e.message}", Toast.LENGTH_SHORT).show()
            } finally {
                recorder = null
                isRecording = false
                btnRecord.text = "Record"
            }
        } else {
            // start
            recorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
                setOutputFile(out.absolutePath)
                try {
                    prepare()
                    start()
                    isRecording = true
                    btnRecord.text = "Stop"
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivity, "Record failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun askModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            val wav = File(cacheDir, "audio.wav")
            if (!wav.exists()) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "No recording found!", Toast.LENGTH_SHORT).show()
                }
                return@launch
            }

            // 1) Transcribe
            val transcript = try {
                NativeBridge.transcribe(wav.path)
            } catch (e: Exception) {
                "⚠️ Transcription error: ${e.message}"
            }

            // 2) Run LLaMA
            val json = try {
                NativeBridge.runLlama(transcript)
            } catch (e: Exception) {
                "⚠️ LLaMA error: ${e.message}"
            }

            // 3) Parse JSON
            val funcCall: FunctionCall? = try {
                Gson().fromJson<FunctionCall>(
                    json,
                    object : TypeToken<FunctionCall>() {}.type
                )
            } catch (_: Exception) {
                null
            }

            // 4) Dispatch or raw JSON
            val result = funcCall?.let {
                try {
                    FunctionHandler.dispatch(this@MainActivity, it.name, it.arguments)
                } catch (e: Exception) {
                    "⚠️ Dispatch error: ${e.message}"
                }
            } ?: json

            // 5) Update UI
            withContext(Dispatchers.Main) {
                tvPrompt.text = transcript
                tvReply .text = result
            }
        }
    }

    private fun copyAsset(assetPath: String): File {
        val filename = assetPath.substringAfterLast('/')
        val outFile = File(filesDir, filename)
        if (!outFile.exists()) {
            assets.open(assetPath).use { input ->
                outFile.outputStream().use { it.write(input.readBytes()) }
            }
        }
        return outFile
    }
}
