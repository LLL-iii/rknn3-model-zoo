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


#ifndef _RKNN_DEMO_QWEN2_5_OMNI_AUDIO_UTILS_H_
#define _RKNN_DEMO_QWEN2_5_OMNI_AUDIO_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

#define MAX_SHAPE_ID        10

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    int n_shapes;
    rknn3_shape_info *shape_infos;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int n_mels;
    int n_frame[MAX_SHAPE_ID];
    int padded_mask_size[MAX_SHAPE_ID];
    int attn_mask_size[MAX_SHAPE_ID];
    int embeds_dim0[MAX_SHAPE_ID];
    int embeds_dim1;
} rknn_qwen2_5_omni_audio_context;

int init_qwen2_5_omni_audio(rknn_qwen2_5_omni_audio_context* audio_ctx, const char* model_path, const char* weight_path, uint32_t core_mask);

int release_qwen2_5_omni_audio(rknn_qwen2_5_omni_audio_context* audio_ctx);

int get_n_audio(rknn_qwen2_5_omni_audio_context* audio_ctx, int audio_len);

int inference_qwen2_5_omni_audio(rknn_qwen2_5_omni_audio_context* audio_ctx, audio_buffer_t* audio, float16* audio_embeds);

#endif //_RKNN_DEMO_QWEN2_5_OMNI_AUDIO_UTILS_H_