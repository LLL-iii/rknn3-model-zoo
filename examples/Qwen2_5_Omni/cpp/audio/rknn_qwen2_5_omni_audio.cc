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
#include <vector>

#include "rknn_qwen2_5_omni_audio.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "audio_utils.h"


int init_qwen2_5_omni_audio(rknn_qwen2_5_omni_audio_context* audio_ctx, const char* model_path, const char* weight_path, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
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
    if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn3_shape_config shape_config;
    ret = rknn3_query(ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &shape_config, sizeof(shape_config));
    if (ret != RKNN3_SUCCESS) {
        printf("Query dynamic shape config failed! ret=%d\n", ret);
        // 如果不支持动态形状，继续使用静态形状
        shape_config.n_shapes = 1;
        shape_config.current_shape_id = 0;
    } else {
        printf("Model supports %d shape combinations\n", shape_config.n_shapes);
        printf("Current shape ID: %d\n", shape_config.current_shape_id);
    }

    rknn3_shape_info *shape_infos = (rknn3_shape_info *)malloc(sizeof(rknn3_shape_info) * shape_config.n_shapes);
    memset(shape_infos, 0, sizeof(rknn3_shape_info) * shape_config.n_shapes);

    // 为每个形状信息分配输入输出属性内存
    for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
        shape_infos[i].shape_id = i;
        shape_infos[i].input_attrs = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr) * io_num.n_input);
        shape_infos[i].output_attrs = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr) * io_num.n_output);
    }

    // 查询所有形状信息
    ret = rknn3_query(ctx, RKNN3_QUERY_DYNAMIC_SHAPE_INFO, shape_infos, sizeof(rknn3_shape_info) * shape_config.n_shapes);
    if (ret != RKNN3_SUCCESS) {
        printf("Query dynamic shape info failed! ret=%d\n", ret);
        for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
            if (shape_infos[i].input_attrs)     free(shape_infos[i].input_attrs);
            if (shape_infos[i].output_attrs)    free(shape_infos[i].output_attrs);
        }
        free(shape_infos);
        shape_infos = NULL;
        return ret;
    }

    // 打印所有形状信息
    for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
        printf("Shape %d (ID: %d)%s:\n", i, shape_infos[i].shape_id, shape_infos[i].is_default ? " [Default]" : "");
        for (uint32_t j = 0; j < shape_infos[i].n_inputs; j++) {
            rknn3_tensor_attr *attr = &shape_infos[i].input_attrs[j];
            printf("  Input %d (%s): [", attr->index, attr->name);
            for (uint32_t k = 0; k < attr->n_dims; k++) {
                printf("%d%s", attr->shape[k], (k < attr->n_dims - 1) ? ", " : "");
            }
            printf("] Aligned size: %lu bytes\n", attr->aligned_size);
        }
    }

    audio_ctx->rknn_ctx = ctx;
    audio_ctx->io_num = io_num;
    audio_ctx->n_shapes = shape_config.n_shapes;

    // 检查attrs, 并根据shape获取相关参数
    bool attrs_err = false;
    for (int i = 0; i < shape_config.n_shapes; i++) {
        rknn3_tensor_attr* input_attrs = shape_infos[i].input_attrs;
        rknn3_tensor_attr* output_attrs = shape_infos[i].output_attrs;
        if (input_attrs[0].layout == RKNN3_TENSOR_UNDEFINED) {
            audio_ctx->n_mels = input_attrs[0].shape[1];
            audio_ctx->n_frame[i] = input_attrs[0].shape[2];
            printf("shape %d: input n_mels=%d, n_frame=%d\n", i, audio_ctx->n_mels, audio_ctx->n_frame[i]);
        } else {
            printf("model is not UNDEFINED input 0 layout, model input 0 error!\n");
            attrs_err = true;
            break;
        }
        if (input_attrs[1].layout == RKNN3_TENSOR_UNDEFINED) {
            audio_ctx->padded_mask_size[i] = input_attrs[1].shape[input_attrs[1].n_dims-1];
            printf("shape %d: input padded_mask_size=%d\n", i, audio_ctx->padded_mask_size[i]);
        } else {
            printf("model is not UNDEFINED input 1 layout, model input 1 error!\n");
            attrs_err = true;
            break;
        }
        if (io_num.n_input > 2) {
            if (input_attrs[2].layout == RKNN3_TENSOR_UNDEFINED) {
                audio_ctx->attn_mask_size[i] = input_attrs[2].shape[input_attrs[2].n_dims-1];
                printf("shape %d: input attn_mask_size=%d\n", i, audio_ctx->attn_mask_size[i]);
            } else {
                printf("model is not UNDEFINED input 2 layout, model input 2 error!\n");
                attrs_err = true;
                break;
            }
        }
        if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED) {
            int n_dims = output_attrs[0].n_dims;
            audio_ctx->embeds_dim0[i] = output_attrs[0].shape[n_dims-2];
            audio_ctx->embeds_dim1 = output_attrs[0].shape[n_dims-1];
            printf("shape %d: output audio embeds dim0=%d,  audio embeds dim1=%d\n", i, audio_ctx->embeds_dim0[i], audio_ctx->embeds_dim1);
        } else {
            printf("model is not UNDEFINED output layout, model output error!\n");
            attrs_err = true;
            break;
        }
    }
    if (attrs_err) {
        for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
            if (shape_infos[i].input_attrs)     free(shape_infos[i].input_attrs);
            if (shape_infos[i].output_attrs)    free(shape_infos[i].output_attrs);
        }
        free(shape_infos);
        shape_infos = NULL;
        return -1;
    }

    audio_ctx->shape_infos = shape_infos;

    audio_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    audio_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        audio_ctx->inputs[i].mem  = rknn3_create_mem(ctx, shape_infos[0].input_attrs[i].aligned_size, shape_infos[0].input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        audio_ctx->outputs[i].mem  = rknn3_create_mem(ctx, shape_infos[0].output_attrs[i].aligned_size, shape_infos[0].output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    }

    return ret;
}

int release_qwen2_5_omni_audio(rknn_qwen2_5_omni_audio_context* audio_ctx)
{
    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        if (audio_ctx->inputs[i].mem)
            rknn3_destroy_mem(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem);
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        if (audio_ctx->outputs[i].mem)
            rknn3_destroy_mem(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem);
    }
    free(audio_ctx->inputs);
    free(audio_ctx->outputs);
    audio_ctx->inputs = NULL;
    audio_ctx->outputs = NULL;

    for (uint32_t i = 0; i < audio_ctx->n_shapes; i++) {
        if (audio_ctx->shape_infos[i].input_attrs)     free(audio_ctx->shape_infos[i].input_attrs);
        if (audio_ctx->shape_infos[i].output_attrs)    free(audio_ctx->shape_infos[i].output_attrs);
    }
    free(audio_ctx->shape_infos);
    audio_ctx->shape_infos = NULL;

    if (audio_ctx->rknn_ctx) {
        rknn3_destroy(audio_ctx->rknn_ctx);
        audio_ctx->rknn_ctx = 0;
    }
    return 0;
}

void vecfp32_to_fp16(std::vector<float> &vec, float16 *dst) {
    for (int i = 0; i < vec.size(); i++) {
        dst[i] = fp32_to_fp16(vec[i]);
    }
}

#define HOP_LENGTH 160
#define N_FFT 400
#define MEL_FILTERS_PATH "mel_128_filters.txt"

int get_shape_id(rknn_qwen2_5_omni_audio_context* audio_ctx, int audio_len) {
    int n_frame_ = (audio_len + HOP_LENGTH - 1) / HOP_LENGTH;
    int i = audio_ctx->n_shapes - 1;
    for (; i >= 0; i--) {
        if (n_frame_ <= audio_ctx->n_frame[i]) { 
            break;
        }
    }
    return i < 0 ? 0 : i;
}

int get_n_audio(rknn_qwen2_5_omni_audio_context* audio_ctx, int audio_len) {
    int shape_id = get_shape_id(audio_ctx, audio_len);
    return audio_ctx->embeds_dim0[shape_id];
}

int inference_qwen2_5_omni_audio(rknn_qwen2_5_omni_audio_context* audio_ctx, audio_buffer_t* audio, float16* audio_embeds)
{
    if ((!audio_ctx) || (!audio)) {
        printf("audio_ctx or audio is NULL");
        return -1;
    }

    int shape_id = get_shape_id(audio_ctx, audio->num_frames);
    int n_mels = audio_ctx->n_mels;
    int n_frame = audio_ctx->n_frame[shape_id];
    int attn_mask_size = audio_ctx->attn_mask_size[shape_id];

    // Read mel filters from file
    int mels_filters_size = N_FFT / 2 + 1;
    float *mel_filters = (float *)malloc(n_mels * mels_filters_size * sizeof(float));
    int ret = read_mel_filters(MEL_FILTERS_PATH, mel_filters, n_mels * mels_filters_size);
    if (ret != 0) {
        printf("read mel_filters fail! Please check if the file \"%s\" exists.\n", MEL_FILTERS_PATH);
        free(mel_filters);
        return -1;
    }

    // Preprocess audio data, cearte padded_feature
    printf("input size: %d x %d\n", n_mels, n_frame);
    std::vector<float> padded_feature(n_mels * n_frame, 0.0f);
    int actual_len;
    audio_preprocess(audio, mel_filters, N_FFT, HOP_LENGTH, n_mels, n_frame * HOP_LENGTH, padded_feature, &actual_len);
    printf("audio_data size: %ld\n", padded_feature.size());
    free(mel_filters);

    // Create padded_mask
    std::vector<float> padded_mask(n_frame, 0.0f);
    for (int i = 0; i < std::min(actual_len, n_frame); i++) {
        padded_mask[i] = 1.0f;
    }
    printf("padded_mask size: %ld\n", padded_mask.size());

    // Create attention_mask
    std::vector<float> attention_mask(attn_mask_size * attn_mask_size, -1e8f);
    for (int i = 0; i < attn_mask_size; ++i) {
        for (int j = 0; j < attn_mask_size; ++j) {
            attention_mask[i * attn_mask_size + j] = 0.0f;
        }
    }
    printf("attention_mask size: %ld\n", attention_mask.size());
    vecfp32_to_fp16(padded_feature, (float16*)(audio_ctx->inputs[0].mem->virt_addr));
    vecfp32_to_fp16(padded_mask, (float16*)(audio_ctx->inputs[1].mem->virt_addr));
    vecfp32_to_fp16(attention_mask, (float16*)(audio_ctx->inputs[2].mem->virt_addr));

    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        audio_ctx->inputs[i].attr = &(audio_ctx->shape_infos[shape_id].input_attrs[i]);
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        audio_ctx->outputs[i].attr = &(audio_ctx->shape_infos[shape_id].output_attrs[i]);
    }

    printf("rknn3_set_shape, shape_id = %d\n", shape_id);
    ret = rknn3_set_shape(audio_ctx->rknn_ctx, shape_id);
    if (ret < 0) {
        printf("rknn3_set_shape fail! ret=%d\n", ret);
        return ret;
    }

    // Sync Inputs
    for (int i = 0; i < audio_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Run
    ret = rknn3_run(audio_ctx->rknn_ctx, audio_ctx->inputs, audio_ctx->io_num.n_input, audio_ctx->outputs, audio_ctx->io_num.n_output);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }

    // Sync Outputs
    for (int i = 0; i < audio_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Get Output
    int embeds_size = audio_ctx->embeds_dim0[shape_id] * audio_ctx->embeds_dim1 * 2;
    memcpy(audio_embeds, (float16*)audio_ctx->outputs[0].mem->virt_addr, embeds_size);

    return ret;
}