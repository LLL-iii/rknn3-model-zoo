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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "time_utils.h"

#include "rknn_qwen3_reranker.h"

int init_qwen3_reranker_model(rknn_qwen3_reranker_context *llm_ctx, const char *model_path, const char *weight_path, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    rknn3_devices devs;
    // Query available devices
    memset(&devs, 0, sizeof(devs));
    ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_find_devices failed! ret=%d\n", ret);
        return -1;
    }
    printf("Found %d RK182X devices\n", devs.n_devices);
    for (int i = 0; i < devs.n_devices; i++) {
        printf("  Device %d: transfer_type=%s, id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
    }

    // Select the first device
    if (devs.n_devices == 0) {
        printf("No RK182X devices found\n");
        return -1;
    } else if (devs.n_devices == 1) {
        // If only one device found, the init_extend can be NULL
        printf("Info: Only one device found (id=%s), init_extend can be NULL\n", devs.devices[0].id);
    } else {
        printf("Multiple devices found, using the first one (id=%s)\n", devs.devices[0].id);
    }
    rknn3_init_extend init_extend = {0};
    init_extend.device_id = devs.devices[0].id;


    // RKNN Init
    ret = rknn3_init(&ctx, &init_extend);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    // Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0)
    {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    llm_ctx->rknn_ctx = ctx;
    return ret;
}

int release_qwen3_reranker(rknn_qwen3_reranker_context *llm_ctx)
{
    if (llm_ctx->rknn_sess)
    {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }

    if (llm_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_qwen3_reranker(rknn_qwen3_reranker_context *llm_ctx, rknn3_llm_tensor tensor, int n_inputs, rknn_perf_metrics_t *perf)
{
    if ((!llm_ctx) || !(llm_ctx->rknn_sess))
    {
        printf("llm_ctx or rknn_session is NULL");
        return -1;
    }

    int ret;
    rknn3_llm_input inputs[n_inputs];
    rknn3_llm_infer_param llm_infer_param;

    memset(inputs, 0, sizeof(inputs));
    memset(&(llm_infer_param), 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;

    // Set Input Data
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input = tensor;

    // Run
    printf("rknn_session_run\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0)
    {
        printf("rknn_session_run fail! ret=%d\n", ret);
        return ret;
    }

    // Query State
    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
    if (ret < 0)
    {
        printf("rknn_session_query_state fail! ret=%d\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;

    return ret;
}