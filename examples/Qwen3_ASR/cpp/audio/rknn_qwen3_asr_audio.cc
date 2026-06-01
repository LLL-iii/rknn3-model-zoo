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
#include <cmath>
#include <iostream>
#include <fstream>

#include "rknn_qwen3_asr_audio.h"
#include "common.h"
#include "file_utils.h"
#include "audio_utils.h"

#define HOP_LENGTH 160
#define N_FFT 400
#define MEL_FILTERS_PATH "mel_128_filters.txt"
#define DOWN_SAMPLE_RATE 8


static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
      shape_str += std::to_string(attrs->shape[j]);
      if (j < attrs->n_dims - 1) {
        shape_str += ", ";
      }
    }
  
    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
      stride_str += std::to_string(attrs->stride[j]);
      if (j < attrs->n_stride - 1) {
        stride_str += ", ";
      }
    }
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}


int init_qwen3_asr_audio(rknn_qwen3_asr_audio_context* audio_ctx, const char* model_path, const char* weight_path, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    audio_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    audio_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    audio_ctx->rknn_ctx = ctx;
    audio_ctx->io_num = io_num;
    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        audio_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        audio_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(audio_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        audio_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        audio_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(audio_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NCHW)
    {
        printf("model is NCHW input layout\n");
        audio_ctx->batch_size = input_attrs[0].shape[0];
        audio_ctx->n_mels = input_attrs[0].shape[2];
        audio_ctx->window_size = input_attrs[0].shape[3];
        printf("input batch_size=%d, input window_size=%d, input n_mels=%d\n",
                audio_ctx->batch_size,
                audio_ctx->window_size, audio_ctx->n_mels);
    }
    else 
    {   
        printf("model is not NCHW input layout, model input error!\n");
        return -1;
    }

    if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED)
    {
        printf("model is UNDEFINED output layout\n");
        audio_ctx->embeds_shape = audio_ctx->outputs[0].attr->shape;
        audio_ctx->embeds_ndims = audio_ctx->outputs[0].attr->n_dims;
        for(int i=0; i<audio_ctx->embeds_ndims; i++)
            printf("audio_ctx->embeds_shape[%d]=%d\n", i, audio_ctx->embeds_shape[i]);
    }
    else
    {
        printf("model is not UNDEFINED output layout, model output error!\n");
        return -1;
    }

    return ret;
}

int release_qwen3_asr_audio(rknn_qwen3_asr_audio_context* audio_ctx)
{
    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        if (audio_ctx->inputs[i].mem) {
            rknn3_destroy_mem(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem);
        }
        if (audio_ctx->inputs[i].attr != NULL) {
            free(audio_ctx->inputs[i].attr);
            audio_ctx->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        if (audio_ctx->outputs[i].mem) {
            rknn3_destroy_mem(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem);
        }
        if (audio_ctx->outputs[i].attr != NULL) {
            free(audio_ctx->outputs[i].attr);
            audio_ctx->outputs[i].attr = NULL;
        }
    }
    if (audio_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(audio_ctx->rknn_ctx);
        audio_ctx->rknn_ctx = 0;
    }
    return 0;
}

int get_n_audio(rknn_qwen3_asr_audio_context* audio_ctx, int audio_len) {
    int n_frame = floor((audio_len - N_FFT) / HOP_LENGTH ) + 1;
    int left_data_len = n_frame % audio_ctx->window_size;
    int n_left_token = ceil(1.0 * left_data_len / DOWN_SAMPLE_RATE);
    int n_token_per_window = ceil(1.0 * audio_ctx->window_size / DOWN_SAMPLE_RATE);
    return floor(n_frame/audio_ctx->window_size) * n_token_per_window + n_left_token;
}

int inference_qwen3_asr_audio(rknn_qwen3_asr_audio_context* audio_ctx, audio_buffer_t* audio, float16* audio_embeds)
{
    if ((!audio_ctx) || (!audio))
    {
        printf("audio_ctx or audio is NULL\n");
        return -1;
    }
    int ret;

    // Read mel filters from file
    int batch_size = audio_ctx->batch_size;
    int n_mels = audio_ctx->n_mels;
    int window_size = audio_ctx->window_size;
    int mels_filters_size = N_FFT / 2 + 1;
    float *mel_filters = (float *)malloc(n_mels * mels_filters_size * sizeof(float));
    ret = read_mel_filters(MEL_FILTERS_PATH, mel_filters, n_mels * mels_filters_size);
    if (ret != 0) {
        printf("read mel_filters fail! Please check if the file \"%s\" exists.\n", MEL_FILTERS_PATH);
        free(mel_filters);
        return -1;
    }

    // Preprocess audio data, create padded_feature
    int n_frame = floor((audio->num_frames - N_FFT) / HOP_LENGTH ) + 1;
    std::vector<float> padded_feature(n_mels * n_frame, 0.0f);
    int actual_len;
    audio_preprocess(audio, mel_filters, N_FFT, HOP_LENGTH, n_mels, n_frame * HOP_LENGTH, padded_feature, &actual_len);
    free(mel_filters);
    float* audio_input_feat = padded_feature.data();

    for(int chunk=0; chunk*batch_size*window_size < n_frame; chunk++) {
        // Set Input Data
        int real_data_len = std::min(batch_size * window_size, n_frame - chunk * batch_size * window_size);
        int n_chunk_token = audio_ctx->embeds_shape[0];
        
        float* dst_base = (float*)audio_ctx->inputs[0].mem->virt_addr;
        
        if (real_data_len < batch_size * window_size) {
            memset(dst_base, 0, (batch_size * n_mels * window_size) * sizeof(float));
            n_chunk_token = real_data_len / window_size * ceil(1.0 * window_size / DOWN_SAMPLE_RATE) + ceil(1.0 * (real_data_len % window_size) / DOWN_SAMPLE_RATE);
        }
        
        for (int b = 0; b < batch_size; b++) {
            int batch_offset = b * n_mels * window_size;
            int src_frame_start = chunk * batch_size * window_size + b * window_size;
            int batch_real_len = std::max(0, std::min(window_size, real_data_len - b * window_size));
            
            if (batch_real_len > 0) {
                for (int i = 0; i < n_mels; i++) {
                    float* dst = dst_base + batch_offset + i * window_size;
                    float* src = (float*)audio_input_feat + i * n_frame + src_frame_start;
                    memcpy(dst, src, batch_real_len * sizeof(float));
                }
            }
        }

        // sync inputs
        for (int i = 0; i < audio_ctx->io_num.n_input; i++)
        {
            ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
                goto out;
            }
        }
        
        // Run
        ret = rknn3_run(audio_ctx->rknn_ctx, audio_ctx->inputs, audio_ctx->io_num.n_input, audio_ctx->outputs, audio_ctx->io_num.n_output);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            goto out;
        }

        // Sync Outputs
        for (int i = 0; i < audio_ctx->io_num.n_output; i++)
        {
            ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
                goto out;
            }
        }

        // Get Output
        memcpy((float16*)audio_embeds+(chunk*audio_ctx->embeds_shape[0]*audio_ctx->embeds_shape[1]), (float16*)audio_ctx->outputs[0].mem->virt_addr, n_chunk_token*audio_ctx->embeds_shape[1]*sizeof(float16));
    }

out:

    return ret;
}