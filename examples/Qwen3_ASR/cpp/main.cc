// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "qwen3_asr.h"
#include "audio_utils.h"
#include "time_utils.h"

#define OFFLINE_MAX_AUDIO_SECONDS (20 * 60)

int64_t first_token;
int64_t inference_start_time;
float audio_duration_sec;
bool first_decode = true;
bool language_flag = true;

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 1.0f,
    .repeat_penalty = 1.2f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

/*-------------------------------------------
                Callback Function
-------------------------------------------*/
int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    Tokenizer *tokenizer = (Tokenizer *)userdata;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n\n--------------------Finished-------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING)
    {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n\n--------------Max new token reached------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_STOP)
    {
        printf("\n\n-----------------------Stop--------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {   
        if (first_decode) {
            printf("language res: ");
            first_token = getCurrentTimeUs();
            first_decode = false;
        }
        if (language_flag && result->token_ids[0] == 151704) {
            language_flag = false;
            printf("\ntext res:\n");
        }
        std::string piece = tokenizer->TokenToPiece(result->token_ids[0]);
        printf("%s", piece.c_str());
        fflush(stdout);
    }
    return 0;
}

int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{
    int n_tokens = 0;
    Tokenizer *tokenizer = (Tokenizer *)userdata;
    n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    struct embedding_info* embed_info = (struct embedding_info*)userdata;

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (int n = 0; n < num_tokens; n++) {
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
            embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

void printf_perf(rknn_perf_metrics_t *p, float audio_duration_sec, int64_t inference_start) 
{

    printf("LLM part performance:\n");
    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n", 
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");


    float ttft_us = (float)(first_token - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;  
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens; 
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token);
    float decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");

    if (p->audio_latency > 0) {
        printf(" Audio latency = %.2f ms, FPS = %.2f\n", 
            (int)p->audio_latency / 1000.f, 1000.f * 1000.f / (int)p->audio_latency);
    }

    float total_inference_us = (float)(p->llm_end_time - inference_start);
    float rtf = (audio_duration_sec > 0) ? (total_inference_us / 1000.0f / 1000.0f / audio_duration_sec) : 0.0f;
    float ttft_include_encoder = prefill_ms + (int)p->audio_latency / 1000.0f;
    printf("\n");
    printf(" Audio Duration = %.2f s\n", audio_duration_sec);
    printf(" Total Inference = %.2f ms\n", total_inference_us / 1000.0f);
    printf(" RTF = %.4f (%.2fx)\n", rtf, rtf);
    printf(" TTFT (include encoder) = %.2f ms\n", ttft_include_encoder);
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 10)
    {
        printf("%s <audio_model_path> <audio_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <audio_core_mask> <llm_core_mask> <audio_path>\n", argv[0]);
        return -1;
    }

    const char *audio_model_path   = argv[1];
    const char *audio_weight_path  = argv[2];
    const char *llm_model_path     = argv[3];
    const char *llm_weight_path    = argv[4];
    const char *tokenizer_path     = argv[5];
    const char *embedding_path     = argv[6];
    uint32_t    audio_core_mask    = strtoul(argv[7], nullptr, 16);
    uint32_t    llm_core_mask      = strtoul(argv[8], nullptr, 16);
    const char *audio_path         = argv[9];

    int ret;
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));

    // RKNN Context
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // Tokenizer
    Tokenizer* tokenizer;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat           emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(rknn3_llm_param));

    // Input Audio
    audio_buffer_t src_audio;
    memset(&src_audio, 0, sizeof(audio_buffer_t));

    // Audio Embed
    size_t n_audio_tokens = 1;
    size_t n_embed_audio = 1;
    float16* audio_embeds = NULL;

    // LLM Multi Model Tensor
    int n_inputs = 1;

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Load Tokenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }
    
    tokenizer->GetVocabInfo(&(vocab_info));
    printf("vocab_info: vocab_size=%d special_bos_id=%d special_eos_id=%d\n", vocab_info.vocab_size,
            vocab_info.special_bos_id[0], vocab_info.special_eos_id[0]);

    // Read Embedding
    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1) {
        printf("Failed to open embedding file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1) {
        printf("Failed to get embedding file size\n");
        goto out;
    }

    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        goto out;
    }

    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    // Set LLM parameters
    params.logits_name               = (char*)"logits";
    params.max_context_len           = MAX_CONTEXT_LEN;
    params.sampling_param            = SAMPLE_PARAMS;
    params.vocab_info.vocab_size     = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id    = vocab_info.linefeed_id;

    // LLM Callback
    callback.result_callback    = result_callback;
    callback.result_userdata    = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    ret = init_qwen3_asr_model(&rknn_app_ctx, llm_model_path, llm_weight_path, audio_model_path, audio_weight_path,
                                &params, n_params, callback, audio_core_mask, llm_core_mask);
    if (ret != 0) {
        printf("init_qwen3_asr_model fail! ret=%d llm_model_path=%s audio_model_path=%s\n", ret, llm_model_path, audio_model_path);
        goto out;
    }

    // Read Audio
    ret = read_audio(audio_path, &src_audio);
    if (ret != 0) {
        printf("read audio fail! ret=%d audio_path=%s\n", ret, audio_path);
        goto out;
    }

    if (src_audio.sample_rate > 0) {
        int max_audio_frames = src_audio.sample_rate * OFFLINE_MAX_AUDIO_SECONDS;
        if (src_audio.num_frames > max_audio_frames) {
            printf("audio is longer than %d seconds, truncate from %.2f s to %.2f s\n",
                   OFFLINE_MAX_AUDIO_SECONDS,
                   (float)src_audio.num_frames / src_audio.sample_rate,
                   (float)max_audio_frames / src_audio.sample_rate);
            src_audio.num_frames = max_audio_frames;
        }
    }

    // Audio Embed
    n_audio_tokens = get_n_audio(&(rknn_app_ctx.audio), src_audio.num_frames);
    n_embed_audio = n_audio_tokens * rknn_app_ctx.audio.embeds_shape[1];
    audio_embeds = (float16*)malloc((n_embed_audio) * sizeof(float16));

    first_decode = true;
    language_flag = true;
    inference_start_time = getCurrentTimeUs();
    ret = inference_qwen3_asr_model(&rknn_app_ctx, &src_audio, audio_embeds, n_audio_tokens, n_inputs, &perf);
    if (ret != 0)
    {
        printf("inference qwen3_asr model fail! ret=%d\n", ret);
        goto out;
    }

    audio_duration_sec = (src_audio.sample_rate > 0) ? 
        ((float)src_audio.num_frames / src_audio.sample_rate) : 0.0f;
    printf_perf(&perf, audio_duration_sec, inference_start_time);

out:
    ret = release_qwen3_asr_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release qwen3_asr model fail! ret=%d\n", ret);
    }

    if (embedding_info.fd != -1) {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL) {
            munmap((void*)embedding_info.embedding_data, emb_st.st_size);
            embedding_info.embedding_data = NULL;
        }
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }

    if (tokenizer != NULL)
    {
        delete tokenizer;
        tokenizer = NULL;
    }

    if (audio_embeds != NULL)
    {
        free(audio_embeds);
    }

    if (src_audio.data != NULL)
    {
        free(src_audio.data);
        src_audio.data = NULL;
    }

    return ret;
}
