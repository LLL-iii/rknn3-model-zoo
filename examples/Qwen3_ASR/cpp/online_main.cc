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
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include "audio_utils.h"
#include "qwen3_asr.h"
#include "time_utils.h"

#define ASR_TEXT_TOKEN 151704           // Qwen3_ASR 语种识别和ASR结果分隔特殊token
#define ONLINE_MAX_AUDIO_SECONDS 60     // 流式模式下，最长推理60秒(音频过长会导致RTF降低)
#define ONLINE_CHUNK_MS 1000            // 流式输入chunk大小，固定1秒
#define ONLINE_MAX_NEW_TOKENS 50        // 流式单轮最大输出长度，结合回退修正，单轮最长输出对应约6秒音频，通常不会超过35个token

// 回退相关配置：流式输入下，音频末尾的ASR输出结果还无法确认，需要等后续音频进行确认后再输出，保证精度
#define ONLINE_ROLLBACK_TOKENS 5        // 回退策略，首5轮输入回退
#define ONLINE_ROLLBACK_ROUNDS 4        // 回退策略，最后4个token回退

#define ANSI_COLOR_RED "\033[31m"
#define ANSI_COLOR_BLUE "\033[34m"
#define ANSI_COLOR_RESET "\033[0m"

int64_t first_token;
bool first_decode = true;

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 1.0f,
    .repeat_penalty = 1.2f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

struct online_state
{
    Tokenizer*           tokenizer;
    std::vector<int32_t> round_token_ids;
    bool                 text_started;
};

struct decoded_asr_text
{
    std::string text;
    bool        text_started;
};

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
    online_state* ctx = (online_state*)userdata;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\nError occurred during inference\n");
        return 0;
    }
    if (state == RKLLM_RUN_FINISH || state == RKLLM_RUN_WAITING || state == RKLLM_RUN_STOP)
    {
        return 0;
    }
    if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n[online] max new token reached\n");
        return 0;
    }
    if (state != RKLLM_RUN_NORMAL)
    {
        return 0;
    }

    if (first_decode)
    {
        first_token = getCurrentTimeUs();
        first_decode = false;
    }

    for (int i = 0; i < result->num_tokens; ++i)
    {
        ctx->round_token_ids.push_back(result->token_ids[i]);
    }

    return 0;
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max)
{
    Tokenizer* tokenizer = (Tokenizer*)userdata;
    int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    struct embedding_info* embed_info = (struct embedding_info*)userdata;

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16))
    {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (uint64_t n = 0; n < num_tokens; n++)
    {
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16),
               embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
               embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

static int append_token_embeds(const embedding_info& embed_info,
                               const std::vector<int32_t>& token_ids,
                               std::vector<float16>* output_embeds)
{
    size_t old_size = output_embeds->size();
    output_embeds->resize(old_size + token_ids.size() * embed_info.embedding_dim);
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        int32_t token_id = token_ids[i];
        if (token_id < 0 || token_id >= embed_info.vocab_size)
        {
            printf("invalid token id: %d\n", token_id);
            return -1;
        }
        memcpy(output_embeds->data() + old_size + i * embed_info.embedding_dim,
               embed_info.embedding_data + token_id * embed_info.embedding_dim,
               embed_info.embedding_dim * sizeof(float16));
    }
    return 0;
}

static decoded_asr_text decode_asr_text(Tokenizer* tokenizer, const std::vector<int32_t>& token_ids, bool initial_text_started)
{
    decoded_asr_text result;
    result.text_started = initial_text_started;
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        int32_t token_id = token_ids[i];
        if (!result.text_started)
        {
            if (token_id == ASR_TEXT_TOKEN)
            {
                result.text_started = true;
            }
            continue;
        }
        result.text += tokenizer->TokenToPiece(token_id);
    }
    return result;
}

static void print_round_perf(const rknn_perf_metrics_t* perf)
{
    float ttft_ms = 0.0f;
    float decode_ms = 0.0f;
    if (!first_decode)
    {
        ttft_ms = (float)(first_token - perf->llm_start_time) / 1000.0f;
        decode_ms = (float)(perf->llm_end_time - first_token) / 1000.0f;
    }
    printf("perf: audio=%.2f ms, prefill_tokens=%d, decode_tokens=%d, ttft=%.2f ms, decode=%.2f ms\n",
           perf->audio_latency / 1000.0f,
           perf->n_prefill_tokens,
           perf->n_decode_tokens,
           ttft_ms,
           decode_ms);
}

static void print_single_line_result(const std::string& committed_text, const std::string& unfix_text)
{
    printf("\r%s", committed_text.c_str());
    if (!unfix_text.empty())
    {
        printf(ANSI_COLOR_BLUE "%s" ANSI_COLOR_RESET, unfix_text.c_str());
    }
    printf("\033[K");
    fflush(stdout);
}

static int build_llm_input(const rknn_app_context_t* app_ctx,
                           const embedding_info& embed_info,
                           const std::vector<float16>& audio_embed_history,
                           const std::vector<int32_t>& committed_token_ids,
                           std::vector<float16>* llm_input_embeds,
                           uint64_t* n_tokens)
{
    size_t hidden_size = app_ctx->audio.embeds_shape[1];
    size_t audio_tokens = audio_embed_history.size() / hidden_size;
    size_t committed_tokens = committed_token_ids.size();

    if (audio_tokens + committed_tokens + PREFIX_N_TOKENS + POSTFIX_N_TOKENS > MAX_CONTEXT_LEN)
    {
        printf("online input exceeds max context len=%d\n", MAX_CONTEXT_LEN);
        return -1;
    }

    llm_input_embeds->clear();
    llm_input_embeds->reserve((audio_tokens + committed_tokens + PREFIX_N_TOKENS + POSTFIX_N_TOKENS) * hidden_size);

    llm_input_embeds->insert(llm_input_embeds->end(),
                             app_ctx->prefix_embeds,
                             app_ctx->prefix_embeds + PREFIX_N_TOKENS * hidden_size);
    llm_input_embeds->insert(llm_input_embeds->end(), audio_embed_history.begin(), audio_embed_history.end());
    llm_input_embeds->insert(llm_input_embeds->end(),
                             app_ctx->postfix_embeds,
                             app_ctx->postfix_embeds + POSTFIX_N_TOKENS * hidden_size);
    if (append_token_embeds(embed_info, committed_token_ids, llm_input_embeds) != 0)
    {
        return -1;
    }

    *n_tokens = audio_tokens + committed_tokens + PREFIX_N_TOKENS + POSTFIX_N_TOKENS;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 10 && argc != 11)
    {
        printf("%s <audio_model_path> <audio_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <audio_core_mask> <llm_core_mask> <audio_path> [--single-line]\n",
               argv[0]);
        return -1;
    }

    const char* audio_model_path   = argv[1];
    const char* audio_weight_path  = argv[2];
    const char* llm_model_path     = argv[3];
    const char* llm_weight_path    = argv[4];
    const char* tokenizer_path     = argv[5];
    const char* embedding_path     = argv[6];
    uint32_t    audio_core_mask    = strtoul(argv[7], NULL, 16);
    uint32_t    llm_core_mask      = strtoul(argv[8], NULL, 16);
    const char* audio_path         = argv[9];
    bool        single_line_mode   = false;

    if (argc == 11)
    {
        if (strcmp(argv[10], "--single-line") == 0 || strcmp(argv[10], "-s") == 0)
        {
            single_line_mode = true;
        }
        else
        {
            printf("invalid option: %s\n", argv[10]);
            return -1;
        }
    }

    int ret = 0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_ctx));

    Tokenizer* tokenizer = NULL;
    VocabInfo vocab_info;

    struct embedding_info embedding_info;
    struct stat emb_st;
    memset(&embedding_info, 0, sizeof(embedding_info));
    embedding_info.fd = -1;

    audio_buffer_t full_audio;
    memset(&full_audio, 0, sizeof(full_audio));

    online_state online_ctx = {};
    std::vector<float16> audio_embed_history;
    std::vector<int32_t> committed_token_ids;
    std::vector<float16> llm_input_embeds;
    bool committed_output_started = false;
    int frames_per_chunk = 0;
    int remaining_rollback_rounds = ONLINE_ROLLBACK_ROUNDS;
    int round_idx = 0;

    rknn3_llm_param params;
    memset(&params, 0, sizeof(params));

    RKLLMCallback callback;
    memset(&callback, 0, sizeof(callback));

    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        ret = -1;
        goto out;
    }

    tokenizer->GetVocabInfo(&vocab_info);
    printf("vocab_info: vocab_size=%d special_bos_id=%d special_eos_id=%d\n",
           vocab_info.vocab_size,
           vocab_info.special_bos_id[0],
           vocab_info.special_eos_id[0]);

    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1)
    {
        printf("Failed to open embedding file: %s\n", embedding_path);
        ret = -1;
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1)
    {
        printf("Failed to get embedding file size\n");
        ret = -1;
        goto out;
    }

    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED)
    {
        printf("Failed to mmap embedding file\n");
        embedding_info.embedding_data = NULL;
        ret = -1;
        goto out;
    }

    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    params.logits_name                 = (char*)"logits";
    params.max_context_len             = MAX_CONTEXT_LEN;
    params.sampling_param              = SAMPLE_PARAMS;
    params.vocab_info.vocab_size       = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id      = vocab_info.linefeed_id;

    online_ctx.tokenizer = tokenizer;
    online_ctx.text_started = false;

    callback.result_callback    = result_callback;
    callback.result_userdata    = &online_ctx;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    ret = init_qwen3_asr_model(&rknn_app_ctx,
                               llm_model_path,
                               llm_weight_path,
                               audio_model_path,
                               audio_weight_path,
                               &params,
                               1,
                               callback,
                               audio_core_mask,
                               llm_core_mask);
    if (ret != 0)
    {
        printf("init_qwen3_asr_model fail! ret=%d llm_model_path=%s audio_model_path=%s\n", ret, llm_model_path, audio_model_path);
        goto out;
    }

    if (single_line_mode)
    {
        printf(ANSI_COLOR_RED "single-line mode enabled: only suitable for single-line ASR output, unfix text is shown in blue.\n" ANSI_COLOR_RESET);
    }

    ret = read_audio(audio_path, &full_audio);
    if (ret != 0)
    {
        printf("read audio fail! ret=%d audio_path=%s\n", ret, audio_path);
        goto out;
    }

    if (full_audio.sample_rate > 0)
    {
        int max_audio_frames = full_audio.sample_rate * ONLINE_MAX_AUDIO_SECONDS;
        if (full_audio.num_frames > max_audio_frames)
        {
            printf("audio is longer than %d seconds, truncate from %.2f s to %.2f s\n",
                   ONLINE_MAX_AUDIO_SECONDS,
                   (float)full_audio.num_frames / full_audio.sample_rate,
                   (float)max_audio_frames / full_audio.sample_rate);
            full_audio.num_frames = max_audio_frames;
        }
    }

    frames_per_chunk = full_audio.sample_rate * ONLINE_CHUNK_MS / 1000;
    if (frames_per_chunk <= 0)
    {
        printf("invalid frames_per_chunk=%d\n", frames_per_chunk);
        ret = -1;
        goto out;
    }

    for (int frame_offset = 0; frame_offset < full_audio.num_frames; frame_offset += frames_per_chunk)
    {
        int chunk_frames = std::min(frames_per_chunk, full_audio.num_frames - frame_offset);
        audio_buffer_t chunk_audio;
        chunk_audio.data = full_audio.data + frame_offset * full_audio.num_channels;
        chunk_audio.num_frames = chunk_frames;
        chunk_audio.num_channels = full_audio.num_channels;
        chunk_audio.sample_rate = full_audio.sample_rate;

        size_t chunk_audio_tokens = get_n_audio(&(rknn_app_ctx.audio), chunk_audio.num_frames);
        size_t chunk_embed_size = chunk_audio_tokens * rknn_app_ctx.audio.embeds_shape[1];
        float16* chunk_audio_embeds = (float16*)malloc(chunk_embed_size * sizeof(float16));
        if (chunk_audio_embeds == NULL)
        {
            printf("malloc chunk_audio_embeds failed\n");
            ret = -1;
            goto out;
        }

        rknn_perf_metrics_t perf;
        memset(&perf, 0, sizeof(perf));
        first_token = 0;
        first_decode = true;
        online_ctx.round_token_ids.clear();

        int64_t audio_start = getCurrentTimeUs();
        ret = inference_qwen3_asr_audio(&(rknn_app_ctx.audio), &chunk_audio, chunk_audio_embeds);
        perf.audio_latency = getCurrentTimeUs() - audio_start;
        if (ret != 0)
        {
            free(chunk_audio_embeds);
            printf("inference_qwen3_asr_audio fail! ret=%d\n", ret);
            goto out;
        }

        audio_embed_history.insert(audio_embed_history.end(), chunk_audio_embeds, chunk_audio_embeds + chunk_embed_size);
        free(chunk_audio_embeds);

        uint64_t n_tokens = 0;
        ret = build_llm_input(&rknn_app_ctx, embedding_info, audio_embed_history, committed_token_ids, &llm_input_embeds, &n_tokens);
        if (ret != 0)
        {
            goto out;
        }

        rknn3_llm_tensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        tensor.name = NULL;
        tensor.prompt = NULL;
        tensor.embed = llm_input_embeds.data();
        tensor.tokens = NULL;
        tensor.n_tokens = n_tokens;
        tensor.enable_thinking = false;

        ret = inference_qwen3_asr_llm_ex(&(rknn_app_ctx.llm), tensor, 1, 0, ONLINE_MAX_NEW_TOKENS, &perf);
        if (ret != 0)
        {
            printf("inference_qwen3_asr_llm_ex fail! ret=%d\n", ret);
            goto out;
        }

        bool is_last_round = (frame_offset + chunk_frames >= full_audio.num_frames);
        size_t stable_token_count = 0;
        if (is_last_round)
        {
            stable_token_count = online_ctx.round_token_ids.size();
        }
        else if (remaining_rollback_rounds > 0)
        {
            remaining_rollback_rounds--;
            stable_token_count = 0;
        }
        else if (online_ctx.round_token_ids.size() > ONLINE_ROLLBACK_TOKENS)
        {
            stable_token_count = online_ctx.round_token_ids.size() - ONLINE_ROLLBACK_TOKENS;
        }

        std::vector<int32_t> stable_tokens(online_ctx.round_token_ids.begin(),
                                           online_ctx.round_token_ids.begin() + stable_token_count);
        std::vector<int32_t> rollback_tokens(online_ctx.round_token_ids.begin() + stable_token_count,
                                             online_ctx.round_token_ids.end());
        committed_token_ids.insert(committed_token_ids.end(), stable_tokens.begin(), stable_tokens.end());

        decoded_asr_text committed_text = decode_asr_text(tokenizer, committed_token_ids, false);
        decoded_asr_text commit_add_text = decode_asr_text(tokenizer, stable_tokens, committed_output_started);
        decoded_asr_text unfix_text = decode_asr_text(tokenizer, rollback_tokens, committed_text.text_started);
        committed_output_started = commit_add_text.text_started;

        round_idx++;
        if (single_line_mode)
        {
            print_single_line_result(committed_text.text, unfix_text.text);
            // Simulate the waiting time for the next audio chunk in real streaming input.
            // Each chunk is 1 second of audio, but model inference only takes tens of ms.
            // This sleep makes the single-line display refresh at roughly the same pace
            // as actual real-time streaming input would arrive.
            usleep(800000);
        }
        else
        {
            printf("\n================ Online Round %d ================\n", round_idx);
            printf("commit_add_text:\n%s\n", commit_add_text.text.c_str());
            printf("unfix_text:\n%s\n", unfix_text.text.c_str());
            print_round_perf(&perf);
        }
    }

    if (single_line_mode)
    {
        printf("\n");
    }

    {
        decoded_asr_text final_committed_text = decode_asr_text(tokenizer, committed_token_ids, false);
        printf("\n================ Final Commit Result ================\n");
        printf("%s\n", final_committed_text.text.c_str());
    }

out:
    ret = release_qwen3_asr_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release qwen3_asr model fail! ret=%d\n", ret);
    }

    if (embedding_info.fd != -1)
    {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL)
        {
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

    if (full_audio.data != NULL)
    {
        free(full_audio.data);
        full_audio.data = NULL;
    }

    return ret;
}
