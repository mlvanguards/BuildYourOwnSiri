#include <jni.h>
#include <string>
#include <vector>
#include <thread>           // for hardware_concurrency
#include <cstring>

#include "whisper.h"
#include "ggml.h"
#include "llama.h"

// globals
static whisper_context* wctx   = nullptr;
static llama_model*    lmodel = nullptr;
static llama_context*  lctx   = nullptr;

//
// JNI helpers
//
static std::string jstr2std(JNIEnv* env, jstring js) {
    const char* s = env->GetStringUTFChars(js, nullptr);
    std::string out(s);
    env->ReleaseStringUTFChars(js, s);
    return out;
}

//
// init Whisper from a GGUF model file
//
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_byos_NativeBridge_initWhisper(
        JNIEnv* env, jobject /* self */, jstring jpath) {

    auto path = jstr2std(env, jpath);
    wctx = whisper_init_from_file(path.c_str());
    return wctx != nullptr;
}

//
// init LLaMA from a GGUF model file
//
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_byos_NativeBridge_initLlama(
        JNIEnv* env, jobject /* self */, jstring jpath) {

    auto path = jstr2std(env, jpath);

    // load model with default params
    llama_model_params mparams = llama_model_default_params();
    lmodel = llama_model_load_from_file(path.c_str(), mparams);  // :contentReference[oaicite:12]{index=12}
    if (!lmodel) return JNI_FALSE;

    // create context with default params (set threads)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_threads = std::thread::hardware_concurrency();
    lctx = llama_init_from_model(lmodel, cparams);
    return lctx != nullptr;
}

//
// Transcribe audio with Whisper
//
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_byos_NativeBridge_transcribe(
        JNIEnv* env, jobject /* self */, jstring jaudioPath) {

    if (!wctx) {
        return env->NewStringUTF("⚠️ Whisper not initialized");
    }

    auto audioPath = jstr2std(env, jaudioPath);

    // TODO: actually load your WAV/PCM into `float * pcm_data` and `int n_samples`
    float* pcm_data = /* … */ nullptr;
    int   n_samples = /* … */ 0;

    if (!pcm_data || n_samples <= 0) {
        return env->NewStringUTF("⚠️ Failed to load audio");
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_realtime = false;

    whisper_full(wctx, wparams, pcm_data, n_samples);

    int n_segs = whisper_full_n_segments(wctx);
    std::string out;
    for (int i = 0; i < n_segs; i++) {
        out += whisper_full_get_segment_text(wctx, i);
        out += "\n";
    }

    return env->NewStringUTF(out.c_str());
}

//
// Run a little LLaMA generation
//
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_byos_NativeBridge_runLlama(
        JNIEnv* env, jobject /* self */, jstring jprompt) {

    if (!lctx) {
        return env->NewStringUTF("⚠️ LLaMA not initialized");
    }

    // 1) get prompt as std::string
    auto prompt = jstr2std(env, jprompt);

    // 2) tokenize
    const llama_vocab* vocab = llama_model_get_vocab(lmodel);      // :contentReference[oaicite:13]{index=13}
    size_t len = prompt.size();
    std::vector<llama_token> tokens(len + 1);
    int32_t n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            (int)len,
            tokens.data(),
            (int)tokens.size(),
            /*add_bos=*/true,
            /*parse_special=*/false
    );

    // 3) initial encode
    llama_batch batch{};
    batch.n_tokens = n_tokens;
    batch.token    = tokens.data();
    llama_encode(lctx, batch);                                     // :contentReference[oaicite:14]{index=14}

    // 4) build a sampler chain (top-k → top-p → temperature → greedy)
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));  // :contentReference[oaicite:15]{index=15}
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    // 5) generate up to 128 tokens
    std::string out;
    for (int i = 0; i < 128; i++) {
        // llama_decode will update the context for next step
        llama_batch next_batch{};
        next_batch.n_tokens = 1;
        llama_decode(lctx, next_batch);                             // :contentReference[oaicite:16]{index=16}

        // sample one token from the last decoded step
        llama_token tok = llama_sampler_sample(chain, lctx, -1);    // :contentReference[oaicite:17]{index=17}
        llama_sampler_accept(chain, tok);

        if (tok == llama_token_eos(vocab)) break;
        out += llama_vocab_get_text(vocab, tok);                     // :contentReference[oaicite:18]{index=18}
    }

    llama_sampler_free(chain);                                      // clean up
    return env->NewStringUTF(out.c_str());
}
