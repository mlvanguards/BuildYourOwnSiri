#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <android/log.h>
#include <mutex>
#include <sched.h>
#include <sys/resource.h>

// ====== AI LIBRARY INCLUDES ======
#include "whisper.h"
#include "llama.h"

// Android logging macros
#define LOG_TAG "BYOS_LLAMA"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global AI model contexts
static whisper_context* wctx = nullptr;
static llama_model* g_llama_model = nullptr;
static llama_context* g_llama_ctx = nullptr;

// Function calling system prompt and tokenized version
static std::string           g_sys_prefix;      // The actual system prompt text
static std::vector<llama_token> g_sys_tokens;   // Pre-tokenized system prompt for efficiency
static const llama_vocab*    g_vocab = nullptr; // Model vocabulary for tokenization
static int g_sys_n_past = 0;                    // KV cache position (always 0 - we clear cache each time)

// JNI callback mechanism for async results
static JavaVM* g_jvm = nullptr;           // Java VM reference for callback
static jobject g_callback_obj = nullptr;  // Global reference to callback object
static jmethodID g_callback_mid = nullptr; // Method ID for onLlamaResult callback

// Thread management for async inference
static std::mutex g_mutex;               // Protects inference state
static std::thread g_inference_thread;   // Background inference thread
static bool g_inference_running = false; // Prevents concurrent inference

// Generation parameters
static int g_max_tokens = 128;          // Maximum tokens to generate per request

// ====== UTILITY FUNCTIONS ======

// Optimize thread performance for inference
static void pin_big() {
    cpu_set_t set; CPU_ZERO(&set);
    // Try to pin to high-performance cores (typically 4-7 on big.LITTLE)
    for (int c = 4; c <= 7; ++c) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        LOGI("Failed to pin to big cores, using default CPU affinity");
    }
    // Increase thread priority for better inference performance
    if (setpriority(PRIO_PROCESS, 0, -10) != 0) {
        LOGI("Failed to set high priority, continuing with default");
    }
}

// Convert JNI string to C++ std::string with proper memory management
std::string jstr2std(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* cstr = env->GetStringUTFChars(jstr, nullptr);
    std::string result(cstr);
    env->ReleaseStringUTFChars(jstr, cstr);
    return result;
}

// Parse WAV files for Whisper (expects 16-bit mono PCM at 16kHz)
struct WavInfo { int sample_rate; int num_channels; int bits_per_sample; size_t num_samples; };

static bool read_wav_mono_s16_16k(const std::string& path, std::vector<float>& pcmf32, WavInfo& info) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;

    auto read_u32 = [&](uint32_t& v) { return fread(&v, sizeof(v), 1, fp) == 1; };
    auto read_u16 = [&](uint16_t& v) { return fread(&v, sizeof(v), 1, fp) == 1; };

    uint32_t riff, riff_size, wave;
    if (!read_u32(riff) || !read_u32(riff_size) || !read_u32(wave)) { fclose(fp); return false; }
    if (riff != 0x46464952u /*'RIFF'*/ || wave != 0x45564157u /*'WAVE'*/) { fclose(fp); return false; }

    uint32_t fmt = 0; uint32_t fmt_size = 0;
    uint16_t audio_format = 0, num_channels = 0, block_align = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, byte_rate = 0;
    uint32_t data = 0, data_size = 0;

    while (!feof(fp)) {
        uint32_t chunk_id = 0, chunk_size = 0;
        if (!read_u32(chunk_id) || !read_u32(chunk_size)) break;
        long next = ftell(fp) + chunk_size;
        if (chunk_id == 0x20746d66u /*'fmt '*/) {
            fmt = chunk_id; fmt_size = chunk_size;
            read_u16(audio_format);
            read_u16(num_channels);
            read_u32(sample_rate);
            read_u32(byte_rate);
            read_u16(block_align);
            read_u16(bits_per_sample);
        } else if (chunk_id == 0x61746164u /*'data'*/) {
            data = chunk_id; data_size = chunk_size;
            break;
        }
        fseek(fp, next, SEEK_SET);
    }

    if (fmt == 0 || data == 0) { fclose(fp); return false; }
    if (audio_format != 1 /* PCM */) { fclose(fp); return false; }
    if (num_channels != 1 || bits_per_sample != 16) { fclose(fp); return false; }

    const size_t num_samples = data_size / (bits_per_sample / 8);
    std::vector<int16_t> pcm16(num_samples);
    if (fread(pcm16.data(), sizeof(int16_t), num_samples, fp) != num_samples) { fclose(fp); return false; }
    fclose(fp);

    pcmf32.resize(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        pcmf32[i] = pcm16[i] / 32768.0f;
    }

    info.sample_rate = (int) sample_rate;
    info.num_channels = num_channels;
    info.bits_per_sample = bits_per_sample;
    info.num_samples = num_samples;
    return sample_rate == 16000;
}

// ====== JNI FUNCTIONS ======

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_byos_NativeBridge_initWhisper(JNIEnv *env, jobject, jstring modelPath) {
    std::string path = jstr2std(env, modelPath);
    
    // Initialize Whisper speech-to-text model
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // CPU-only for Android compatibility
    
    wctx = whisper_init_from_file_with_params(path.c_str(), cparams);
    return (wctx != nullptr) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_byos_NativeBridge_transcribe(JNIEnv *env, jobject, jstring audioPath) {
    if (!wctx) {
        return env->NewStringUTF("⚠️ Whisper not initialized");
    }
    
    std::string path = jstr2std(env, audioPath);
    
    // Load and validate audio file format
    std::vector<float> pcm_data;
    WavInfo winfo;
    if (!read_wav_mono_s16_16k(path, pcm_data, winfo)) {
        return env->NewStringUTF("Unsupported audio format. Record mono 16-bit PCM @16kHz (WAV)");
    }
    
    // Configure Whisper for fast speech recognition
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads = std::max(2, (int)std::thread::hardware_concurrency() - 1);
    wparams.no_timestamps = true;
    wparams.single_segment = true;   // Process as single segment (faster for short audio)
    wparams.no_context = true;       // Don't use context from previous segments
    wparams.max_len = 64;            // Limit output length for voice commands
    wparams.translate = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    
    // Run speech recognition
    if (whisper_full(wctx, wparams, pcm_data.data(), pcm_data.size()) != 0) {
        return env->NewStringUTF("⚠️ Whisper inference failed");
    }
    
    // Extract transcribed text from all segments
    const int n_segments = whisper_full_n_segments(wctx);
    std::string result;
    
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(wctx, i);
        if (text) {
            result += text;
        }
    }
    
    if (result.empty()) {
        result = "";
    }
    
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_byos_NativeBridge_initLlama(JNIEnv *env, jobject, jstring modelPath) {
    if (g_llama_ctx) {
        LOGI("LLaMA already initialized, reusing existing context");
        return JNI_TRUE;
    }
    std::string path = jstr2std(env, modelPath);

    LOGI("initLlama(): requested model path = %s", path.c_str());

    llama_backend_init();

    // forward llama.cpp internal logs to logcat
    llama_log_set(
        [](ggml_log_level level, const char* text, void* /*user_data*/){
            if (!text) return;
            switch (level) {
                case GGML_LOG_LEVEL_ERROR: LOGE("%s", text); break;
                case GGML_LOG_LEVEL_WARN:  LOGI("[W] %s", text); break;
                default:                   LOGI("%s", text); break;
            }
        },
        nullptr
    );

    // Configure model loading parameters for Android
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only for compatibility 
    mparams.use_mmap = true;   // Memory map model file for efficiency
    mparams.use_mlock = false; // Don't lock pages (causes issues on Android)
    mparams.progress_callback = [](float progress, void* /*user_data*/) -> bool {
        int pct = (int) (progress * 100.0f);
        LOGI("model load progress: %d%%", pct);
        return true; // continue loading
    };
    mparams.progress_callback_user_data = nullptr;

    // Load the LLaMA model from file
    g_llama_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_llama_model) {
        LOGE("llama_model_load_from_file failed. Model not loaded.");
            return JNI_FALSE;
    }

    // Configure inference context for stability on Android
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 600;     // Context window (max tokens in conversation)
    cparams.n_batch = 64;    // Process tokens in small batches for stability
    cparams.n_threads = 2;   // Use 2 threads for reasonable performance
    cparams.n_ubatch = 64;   // Unified batch size for consistency
    cparams.rope_freq_base = 10000.0f;
    cparams.rope_freq_scale = 1.0f;
    cparams.embeddings = false;  // We need text generation, not embeddings
    cparams.offload_kqv = false; // Keep everything on CPU for simplicity
    g_llama_ctx = llama_init_from_model(g_llama_model, cparams);
    if (!g_llama_ctx) {
        LOGE("llama_init_from_model failed. Context not created.");
        return JNI_FALSE;
    }

    // Cache model vocabulary for tokenization operations
    g_vocab = llama_model_get_vocab(g_llama_model);
    LOGI("Model loaded successfully");

    // Build function calling system prompt from FunctionRegistry
    {
        jclass regCls = env->FindClass("com/example/byos/FunctionRegistry");
        if (regCls) {
            jmethodID mid = env->GetStaticMethodID(regCls, "toolsJson", "()Ljava/lang/String;");
            if (mid) {
                jstring jTools = (jstring) env->CallStaticObjectMethod(regCls, mid);
                if (jTools) {
                     const char* cstr = env->GetStringUTFChars(jTools, nullptr);
                     std::string tools_json(cstr ? cstr : "");
                     env->ReleaseStringUTFChars(jTools, cstr);
                     env->DeleteLocalRef(jTools);

                     // Create system prompt that instructs the model to output JSON function calls
                     std::string tools_json_str = tools_json;
                     g_sys_prefix = "SYSTEM: You are a function calling assistant. Respond ONLY with valid JSON.\n\nFORMAT: [{\"name\": \"function_name\", \"arguments\": {}}]\n\nFUNCTIONS: " + tools_json_str + "\n\nRULES:\n- ONLY output JSON\n- NO explanations\n- NO other text\n\nUSER: ";

                     LOGI("System prefix created with %d tokens", g_sys_tokens.size());

                     // Pre-tokenize system prompt for efficiency (avoids re-tokenizing each request)
                     g_sys_tokens.resize(600); // Much larger buffer
                     int n = llama_tokenize(g_vocab, g_sys_prefix.c_str(), (int)g_sys_prefix.size(),
                                             g_sys_tokens.data(), (int)g_sys_tokens.size(), true, true);
                     if (n < 0) {
                       LOGE("CRITICAL: Tokenization failed with error %d", n);
                       n = -n; // Convert negative to positive for size
                     }
                     if (n > 600) n = 600; // Cap at buffer size
                     g_sys_tokens.resize(n);
                     LOGI("Pre-tokenized system prefix: %d tokens", n);
                     g_sys_n_past = 0; // Always start fresh (we clear KV cache between requests)
                }
            }
            env->DeleteLocalRef(regCls);
        }
    }

    LOGI("LLaMA initialized successfully");
    return JNI_TRUE;
}

// ====== ASYNC CALLBACK HELPER ======

void callJavaCallback(const std::string& result) {
    if (!g_jvm || !g_callback_obj || !g_callback_mid) {
        LOGE("callJavaCallback: missing callback setup");
        return;
    }
    
    JNIEnv* env;
    int getEnvStat = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (getEnvStat == JNI_EDETACHED) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) != 0) {
            LOGE("callJavaCallback: failed to attach thread");
            return;
        }
    } else if (getEnvStat != JNI_OK) {
        LOGE("callJavaCallback: failed to get JNI env");
        return;
    }
    
    // Check if callback object is still valid
    if (!g_callback_obj) {
        if (getEnvStat == JNI_EDETACHED) {
            g_jvm->DetachCurrentThread();
        }
        return;
    }
    
    jstring jresult = env->NewStringUTF(result.c_str());
    env->CallVoidMethod(g_callback_obj, g_callback_mid, jresult);
    env->DeleteLocalRef(jresult);
    
    if (getEnvStat == JNI_EDETACHED) {
        g_jvm->DetachCurrentThread();
    }
}

// ====== MAIN INFERENCE FUNCTION ======
// This function runs the LLM to convert user input into JSON function calls
// It runs on a background thread to avoid blocking the UI

void runLlamaAsync(const std::string& user_input) {
    LOGI("runLlamaAsync: Starting inference");
    
    // Verify all required components are initialized
     if (!g_llama_ctx || !g_llama_model || g_sys_tokens.empty()) {
         LOGE("runLlamaAsync: missing required components");
         callJavaCallback("[]");
         return;
     }
    
    if (user_input.empty()) {
        LOGE("runLlamaAsync: empty input");
        callJavaCallback("[]");
        return;
    }
    
        LOGI("runLlamaAsync: Starting tokenization");

    // Tokenize user input and combine with pre-tokenized system prompt
     std::vector<llama_token> user_tokens(256); // Buffer for user input
     int n_user = llama_tokenize(g_vocab, user_input.c_str(), (int)user_input.size(),
                              user_tokens.data(), (int)user_tokens.size(), true, true);
     if (n_user < 0) {
      LOGE("runLlamaAsync: User tokenization failed with error %d", n_user);
      callJavaCallback("[]");
      return;
    }
     if (n_user > 256) n_user = 256; // Cap at 256 tokens maximum
     user_tokens.resize(n_user);
    
    // Combine pre-tokenized system prefix + user tokens
     std::vector<llama_token> full_tokens;
     full_tokens.reserve(g_sys_tokens.size() + n_user);
     full_tokens.insert(full_tokens.end(), g_sys_tokens.begin(), g_sys_tokens.end());
     full_tokens.insert(full_tokens.end(), user_tokens.begin(), user_tokens.end());
     int n_full = full_tokens.size();
     
     LOGI("runLlamaAsync: Processing %d tokens (%d system + %d user)", n_full, g_sys_tokens.size(), n_user);

         // Process tokens in chunks to avoid memory issues and crashes
    const int chunk_size = 64;
     int processed_tokens = 0;
     
     while (processed_tokens < n_full) {
         int current_chunk_size = std::min(chunk_size, n_full - processed_tokens);
         
         llama_batch chunkb = llama_batch_init(current_chunk_size, 0, 1);
         if (!chunkb.token) {
             LOGE("runLlamaAsync: failed to create chunk batch");
             callJavaCallback("[]");
             return;
         }
         
         chunkb.n_tokens = current_chunk_size;
         for (int i = 0; i < current_chunk_size; ++i) {
             chunkb.token[i]    = full_tokens[processed_tokens + i];
                         chunkb.pos[i]      = processed_tokens + i;  // Position in sequence
            chunkb.n_seq_id[i] = 1;                      // One sequence ID
            chunkb.seq_id[i][0]= 0;                      // Sequence 0
            chunkb.logits[i]   = (i == current_chunk_size - 1 && processed_tokens + current_chunk_size >= n_full); // Only need logits on final token
         }


         if (llama_decode(g_llama_ctx, chunkb) != 0) {
             LOGE("runLlamaAsync: llama_decode failed on chunk");
             llama_batch_free(chunkb);
             callJavaCallback("[]");
             return;
         }
         llama_batch_free(chunkb);
         processed_tokens += current_chunk_size;
     }
     


         const int base_pos = n_full; // Starting position for generation

    // Initialize greedy sampler (always picks most likely token)
    llama_sampler* sampler = llama_sampler_init_greedy();
    if (!sampler) { LOGE("failed to create sampler"); callJavaCallback("[]"); return; }

    auto t0 = std::chrono::steady_clock::now();
         auto timed_out = [&]{
         using namespace std::chrono;
         return duration_cast<seconds>(steady_clock::now() - t0).count() > 15; // Timeout to prevent hanging
     };

    std::string out; // Accumulate generated text here
    
    // Generate tokens one by one until we have complete JSON
    try {
        for (int i = 0; i < g_max_tokens; ++i) {
            if (timed_out()) { LOGI("timeout after %d tokens", i); break; }

            // Sample next most likely token
            llama_token id = llama_sampler_sample(sampler, g_llama_ctx, -1);
            if (llama_vocab_is_eog(g_vocab, id)) break; // End of generation token

            // Convert token to text and add to output
            char buf[256];
            int n = llama_token_to_piece(g_vocab, id, buf, sizeof(buf), 0, true);
            if (n > 0) out.append(buf, n);

            // Feed the generated token back into the model for next prediction
            llama_batch genb = llama_batch_init(1, 0, 1);
            genb.n_tokens = 1;
            genb.token[0]    = id;
            genb.pos[0]      = base_pos + i;  // Position in sequence
            genb.n_seq_id[0] = 1;
            genb.seq_id[0][0]= 0;
            genb.logits[0]   = 1; // We need logits for next prediction

            if (llama_decode(g_llama_ctx, genb) != 0) {
                LOGE("decode failed at gen step %d", i);
                llama_batch_free(genb);
                break;
            }
            llama_batch_free(genb);

            // Check if we've generated complete JSON - stop early to prevent garbage
            if (out.find("[{\"name\"") != std::string::npos && out.find("\"arguments\"") != std::string::npos) {
                // Count braces and brackets to check for complete JSON
                int open_braces = 0;
                int open_brackets = 0;
                for (char c : out) {
                    if (c == '{') open_braces++;
                    if (c == '}') open_braces--;
                    if (c == '[') open_brackets++;
                    if (c == ']') open_brackets--;
                }
                
                // If we have balanced braces/brackets and end with ']', we have complete JSON
                if (open_braces == 0 && open_brackets == 0 && out.back() == ']') {
                    break;
                }
            }

        }
    } catch (const std::exception& e) {
        LOGE("runLlamaAsync: exception during sampling: %s", e.what());
        out = "[]";
    } catch (...) {
        LOGE("runLlamaAsync: unknown exception during sampling");
        out = "[]";
    }
    
    // Clean up sampler
    llama_sampler_free(sampler);

    // Clear KV cache so next request starts fresh
    llama_memory_t mem = llama_get_memory(g_llama_ctx);
    if (mem) {
         llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/0, /*p1=*/-1);
    }

    LOGI("runLlamaAsync: RAW OUTPUT: '%s'", out.c_str());

    // Ensure we return valid JSON even if empty
    if (out.empty()) out = "[]";

    // Fallback: if model generated natural language instead of JSON, use keyword matching
     if (out.find("[{\"name\"") == std::string::npos || 
         out.find("You can use") != std::string::npos || 
         out.find("Your task is") != std::string::npos ||
         out.find("sequence of") != std::string::npos ||
         out.find("commands necessary") != std::string::npos ||
         out.find("NOT A PLAN") != std::string::npos || 
         out.find("UTTERANCE") != std::string::npos) {
         LOGI("runLlamaAsync: Using keyword fallback");
         
         // Use simple keyword matching to determine user intent
         std::string lower_input = user_input;
         std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
         
         if (lower_input.find("flashlight") != std::string::npos || lower_input.find("light") != std::string::npos || lower_input.find("torch") != std::string::npos) {
             out = "[{\"name\": \"toggle_flashlight\", \"arguments\": {}}]";
         } else if (lower_input.find("battery") != std::string::npos || lower_input.find("power") != std::string::npos) {
             out = "[{\"name\": \"get_battery_status\", \"arguments\": {}}]";
         } else if (lower_input.find("volume") != std::string::npos || lower_input.find("sound") != std::string::npos) {
             out = "[{\"name\": \"set_volume\", \"arguments\": {\"level\": 50}}]";
         } else if (lower_input.find("music") != std::string::npos || lower_input.find("play") != std::string::npos) {
             out = "[{\"name\": \"play_music\", \"arguments\": {\"track_name\": \"default\"}}]";
         } else if (lower_input.find("close") != std::string::npos && lower_input.find("app") != std::string::npos) {
             out = "[{\"name\": \"close_application\", \"arguments\": {\"app_name\": \"current\"}}]";
         } else {
             // Default fallback
             out = "[{\"name\": \"get_battery_status\", \"arguments\": {}}]";
         }

     }

    LOGI("runLlamaAsync: Final result: %s", out.c_str());
    callJavaCallback(out);
}

// ====== ASYNC JNI FUNCTIONS ======

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_byos_NativeBridge_initAsyncCallback(JNIEnv *env, jobject thiz, jobject callback) {
    // Store JavaVM for later use
    env->GetJavaVM(&g_jvm);
    
    // Create global reference to callback object
    g_callback_obj = env->NewGlobalRef(callback);
    
    // Get callback method ID
    jclass callbackClass = env->GetObjectClass(callback);
    g_callback_mid = env->GetMethodID(callbackClass, "onLlamaResult", "(Ljava/lang/String;)V");
    
    if (!g_callback_mid) {
        LOGE("initAsyncCallback: failed to get callback method");
        return JNI_FALSE;
    }
    
    LOGI("Async callback initialized successfully");
    return JNI_TRUE;
}

 extern "C" JNIEXPORT jboolean JNICALL
 Java_com_example_byos_NativeBridge_runLlamaAsync(JNIEnv *env, jobject, jstring prompt) {
     std::lock_guard<std::mutex> lock(g_mutex);
     
     if (g_inference_running) {
         LOGE("runLlamaAsync: inference already running, rejecting new request");
         return JNI_FALSE;
     }
     
     // Wait for previous thread to complete if it exists and is joinable
     if (g_inference_thread.joinable()) {
         g_inference_thread.join();
     }
     
     std::string user_input = jstr2std(env, prompt);
     
     g_inference_running = true;
     g_inference_thread = std::thread([user_input]() {
         pin_big(); // Pin to big cores for better performance
         runLlamaAsync(user_input);
         std::lock_guard<std::mutex> lock(g_mutex);
         g_inference_running = false;
     });
     
     return JNI_TRUE;
 }

extern "C" JNIEXPORT void JNICALL
Java_com_example_byos_NativeBridge_cleanupCallback(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_callback_obj) {
        env->DeleteGlobalRef(g_callback_obj);
        g_callback_obj = nullptr;
    }
    g_callback_mid = nullptr;
    g_jvm = nullptr;
    
    LOGI("Async callback cleaned up");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_byos_NativeBridge_cleanup(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    // Wait for any running inference to complete
    if (g_inference_running && g_inference_thread.joinable()) {
        g_inference_thread.join();
    }
    
    // Clean up LLaMA resources
    if (g_llama_ctx) {
        llama_free(g_llama_ctx);
        g_llama_ctx = nullptr;
    }
    
    if (g_llama_model) {
        llama_model_free(g_llama_model);
        g_llama_model = nullptr;
    }
    
    // Clean up Whisper resources
    if (wctx) {
        whisper_free(wctx);
        wctx = nullptr;
    }
    
    // Clear cached data
    g_sys_tokens.clear();
    g_sys_prefix.clear();
    g_vocab = nullptr;
    
    LOGI("All AI resources cleaned up");
}


