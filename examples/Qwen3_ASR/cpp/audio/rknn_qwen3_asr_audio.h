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


#ifndef _RKNN_DEMO_QWEN3_ASR_AUDIO_UTILS_H_
#define _RKNN_DEMO_QWEN3_ASR_AUDIO_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int batch_size;
    int window_size;
    int n_mels;

    uint32_t* embeds_shape;
    uint32_t embeds_ndims;
} rknn_qwen3_asr_audio_context;

int init_qwen3_asr_audio(rknn_qwen3_asr_audio_context* audio_ctx, const char* model_path, const char* weight_path, uint32_t core_mask);

int release_qwen3_asr_audio(rknn_qwen3_asr_audio_context* audio_ctx);

int inference_qwen3_asr_audio(rknn_qwen3_asr_audio_context* audio_ctx, audio_buffer_t* audio, float16* audio_embeds);

int get_n_audio(rknn_qwen3_asr_audio_context* audio_ctx, int audio_len);

#endif //_RKNN_DEMO_QWEN3_ASR_AUDIO_UTILS_H_