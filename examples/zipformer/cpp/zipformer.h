// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#ifndef _RKNN_DEMO_ZIPFORMER_H_
#define _RKNN_DEMO_ZIPFORMER_H_

#include "rknn3_api.h"
#include "audio_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include "process.h"

#define BLANK_ID 0
#define UNK_ID 2

typedef struct
{
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;
} rknn_app_context_t;

typedef struct
{
    rknn_app_context_t encoder_context;
    rknn_app_context_t decoder_context;
    rknn_app_context_t joiner_context;
} rknn_zipformer_context_t;

int init_zipformer_model(const char *model_path, const char *weight_path, rknn_app_context_t *app_ctx, uint32_t core_mask);
int inference_zipformer_model(rknn_zipformer_context_t *app_ctx, audio_buffer_t audio, VocabEntry *vocab, std::vector<std::string> &recognized_text,
                              std::vector<float> &timestamp, float &audio_length);
int release_zipformer_model(rknn_app_context_t *app_ctx);

#endif //_RKNN_DEMO_ZIPFORMER_H_