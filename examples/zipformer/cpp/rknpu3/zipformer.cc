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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "zipformer.h"
#include "process.h"
#include "float16.h"

// Convert NCHW FP16 to NC1HWC2 FP16 (for encoder cached states)
static int NCHW_fp16_to_NC1HWC2_fp16(const float16* src, float16* dst, int batch, int h, int w, int channel, int sub_c)
{
  // Use size_t to prevent integer overflow
  size_t hw = (size_t)w * (size_t)h;
  size_t C1 = (channel + sub_c - 1) / sub_c;  // Ceiling division
  size_t align_c = C1 * sub_c;

  // Validate input parameters to prevent overflow
  if (batch <= 0 || h <= 0 || w <= 0 || channel <= 0 || sub_c <= 0) {
    printf("Error: Invalid dimensions in NCHW_fp16_to_NC1HWC2_fp16\n");
    return -1;
  }

  // Check for potential size overflow
  size_t required_src_size = (size_t)batch * (size_t)channel * hw;
  size_t required_dst_size = (size_t)batch * align_c * hw;
  if (required_src_size > SIZE_MAX / sizeof(float16) || required_dst_size > SIZE_MAX / sizeof(float16)) {
    printf("Error: Buffer size would overflow in NCHW_fp16_to_NC1HWC2_fp16\n");
    return -1;
  }

  for (int b = 0; b < batch; b++) {
    const float16* src_b = src + b * channel * hw;
    float16* dst_b = dst + b * align_c * hw;

    for (int c = 0; c < channel; ++c) {
      int plane = c / sub_c;
      float16* dstPlane = plane * hw * sub_c + dst_b;
      int offset = c % sub_c;

      for (int cur_h = 0; cur_h < h; ++cur_h) {
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int cur_hw = cur_h * w + cur_w;
          dstPlane[sub_c * cur_hw + offset] = src_b[c * hw + cur_hw];
        }
      }
    }

    // Pad remaining channels with zeros
    for (int c = channel; c < (int)align_c; ++c) {
      int plane = c / sub_c;
      float16* dstPlane = plane * hw * sub_c + dst_b;
      int offset = c % sub_c;

      for (int cur_h = 0; cur_h < h; ++cur_h) {
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int cur_hw = cur_h * w + cur_w;
          dstPlane[sub_c * cur_hw + offset] = fp32_to_fp16(0.0f);  // Zero padding
        }
      }
    }
  }

  return 0;
}

static void dump_tensor_attr(rknn3_tensor_attr* attr)
{
    std::string shape_str = "";
    for (int j = 0; j < attr->n_dims; j++) {
      shape_str += std::to_string(attr->shape[j]);
      if (j < attr->n_dims - 1) {
        shape_str += ", ";
      }
    }

    printf("  index=%d, name=%s, n_dims=%d, shape=[%s], n_elems=%d, aligned_size=%zu, fmt=%s, type=%s, qnt_type=%s, core_id=%d\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, (size_t)attr->aligned_size,
           rknn3_get_layout_string(attr->layout),
           rknn3_get_type_string(attr->dtype),
           rknn3_get_qnt_type_string(attr->qnt_type), attr->core_id);
}

int init_zipformer_model(const char *model_path, const char *weight_path, rknn_app_context_t *app_ctx, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // RKNN3 Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn3_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN3 Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("rknn3_load_model_from_path fail! ret=%d\n", ret);
        return -1;
    }

    // Init RKNN3 Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn3_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn3_query fail! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    std::vector<rknn3_tensor_attr> input_attrs(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query fail! ret=%d\n", ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    std::vector<rknn3_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query fail! ret=%d\n", ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;

    // Allocate memory for inputs and outputs
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    if (app_ctx->inputs == NULL) {
        printf("malloc app_ctx->inputs failed!\n");
        rknn3_destroy(ctx);
        return -1;
    }

    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (app_ctx->outputs == NULL) {
        printf("malloc app_ctx->outputs failed!\n");
        free(app_ctx->inputs);
        app_ctx->inputs = NULL;
        rknn3_destroy(ctx);
        return -1;
    }

    for (int i = 0; i < io_num.n_input; i++) {
        app_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (app_ctx->inputs[i].attr == NULL) {
            printf("malloc app_ctx->inputs[%d].attr failed!\n", i);
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(app_ctx->inputs[j].attr);
                app_ctx->inputs[j].attr = NULL;
                rknn3_destroy_mem(ctx, app_ctx->inputs[j].mem);
            }
            free(app_ctx->inputs);
            app_ctx->inputs = NULL;
            free(app_ctx->outputs);
            app_ctx->outputs = NULL;
            rknn3_destroy(ctx);
            return -1;
        }
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    for (int i = 0; i < io_num.n_output; i++) {
        app_ctx->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (app_ctx->outputs[i].attr == NULL) {
            printf("malloc app_ctx->outputs[%d].attr failed!\n", i);
            // Clean up previously allocated memory
            for (int j = 0; j < io_num.n_input; j++) {
                free(app_ctx->inputs[j].attr);
                app_ctx->inputs[j].attr = NULL;
                rknn3_destroy_mem(ctx, app_ctx->inputs[j].mem);
            }
            for (int j = 0; j < i; j++) {
                free(app_ctx->outputs[j].attr);
                app_ctx->outputs[j].attr = NULL;
                rknn3_destroy_mem(ctx, app_ctx->outputs[j].mem);
            }
            free(app_ctx->inputs);
            app_ctx->inputs = NULL;
            free(app_ctx->outputs);
            app_ctx->outputs = NULL;
            rknn3_destroy(ctx);
            return -1;
        }
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    return 0;
}

int release_zipformer_model(rknn_app_context_t *app_ctx)
{
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->inputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
        }
        if (app_ctx->inputs[i].attr != NULL) {
            free(app_ctx->inputs[i].attr);
            app_ctx->inputs[i].attr = NULL;
        }
    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->outputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
        }
        if (app_ctx->outputs[i].attr != NULL) {
            free(app_ctx->outputs[i].attr);
            app_ctx->outputs[i].attr = NULL;
        }
    }

    if (app_ctx->inputs != NULL) {
        free(app_ctx->inputs);
        app_ctx->inputs = NULL;
    }

    if (app_ctx->outputs != NULL) {
        free(app_ctx->outputs);
        app_ctx->outputs = NULL;
    }

    if (app_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }

    return 0;
}

static int inference_encoder_model(rknn_app_context_t *app_ctx)
{
    int ret = 0;

    // Sync inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input, app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn3_run fail! ret=%d\n", ret);
        return ret;
    }

    // Sync outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Update cached states for next iteration (outputs 1-23 go to inputs 1-23)
    for (int i = 1; i < app_ctx->io_num.n_input; i++)
    {
        // Check data types
        bool output_is_fp16 = (app_ctx->outputs[i].attr->dtype == RKNN3_TENSOR_FLOAT16);
        bool input_is_fp16 = (app_ctx->inputs[i].attr->dtype == RKNN3_TENSOR_FLOAT16);
        bool input_is_int32 = (app_ctx->inputs[i].attr->dtype == RKNN3_TENSOR_INT32);
        bool output_is_int32 = (app_ctx->outputs[i].attr->dtype == RKNN3_TENSOR_INT32);

        // Handle different data type combinations
        if (output_is_fp16 && input_is_int32) {
            // Convert FP16 to INT32 (for cached_len_*)
            float16 *output_fp16 = (float16 *)app_ctx->outputs[i].mem->virt_addr;
            int32_t *input_int32 = (int32_t *)app_ctx->inputs[i].mem->virt_addr;
            for (int j = 0; j < app_ctx->outputs[i].attr->n_elems; j++) {
                float fp32_val = fp16_to_fp32(output_fp16[j]);
                input_int32[j] = (int32_t)round(fp32_val);  // Use round for proper conversion
            }
        } else if (output_is_fp16 && input_is_fp16) {
            // Both FP16, need to check format conversion
            if (app_ctx->outputs[i].attr->layout == RKNN3_TENSOR_NCHW &&
                app_ctx->inputs[i].attr->layout == RKNN3_TENSOR_NC1HWC2) {
                // Need format conversion: NCHW -> NC1HWC2
                const float16* output_fp16 = (float16*)app_ctx->outputs[i].mem->virt_addr;
                float16* input_fp16 = (float16*)app_ctx->inputs[i].mem->virt_addr;

                // Extract dimensions
                int batch = app_ctx->outputs[i].attr->shape[0];
                int channel = app_ctx->outputs[i].attr->shape[1];
                int h = app_ctx->outputs[i].attr->shape[2];
                int w = app_ctx->outputs[i].attr->shape[3];
                int sub_c = 16;  // C2 dimension for NC1HWC2

                NCHW_fp16_to_NC1HWC2_fp16(output_fp16, input_fp16, batch, h, w, channel, sub_c);
            } else {
                // Same format, direct copy
                memcpy(app_ctx->inputs[i].mem->virt_addr,
                       app_ctx->outputs[i].mem->virt_addr,
                       app_ctx->inputs[i].attr->aligned_size);
            }
        } else if (output_is_fp16 && !input_is_fp16 && !input_is_int32) {
            // Convert FP16 to FP32
            float16 *output_fp16 = (float16 *)app_ctx->outputs[i].mem->virt_addr;
            float *input_fp32 = (float *)app_ctx->inputs[i].mem->virt_addr;
            for (int j = 0; j < app_ctx->outputs[i].attr->n_elems; j++) {
                input_fp32[j] = fp16_to_fp32(output_fp16[j]);
            }
        } else if (!output_is_fp16 && input_is_fp16) {
            // Convert FP32 to FP16 (shouldn't happen based on model specs, but handle it)
            float *output_fp32 = (float *)app_ctx->outputs[i].mem->virt_addr;
            float16 *input_fp16 = (float16 *)app_ctx->inputs[i].mem->virt_addr;
            for (int j = 0; j < app_ctx->outputs[i].attr->n_elems; j++) {
                input_fp16[j] = fp32_to_fp16(output_fp32[j]);
            }
        } else {
            // Both same non-FP16 type, direct copy or handle layout conversion
            if (app_ctx->inputs[i].attr->layout == RKNN3_TENSOR_NHWC)
            {
                int N = app_ctx->inputs[i].attr->shape[0];
                int H = app_ctx->inputs[i].attr->shape[1];
                int W = app_ctx->inputs[i].attr->shape[2];
                int C = app_ctx->inputs[i].attr->shape[3];
                convert_nchw_to_nhwc((float *)app_ctx->outputs[i].mem->virt_addr,
                                    (float *)app_ctx->inputs[i].mem->virt_addr, N, C, H, W);
            }
            else
            {
                memcpy(app_ctx->inputs[i].mem->virt_addr,
                       app_ctx->outputs[i].mem->virt_addr,
                       app_ctx->inputs[i].attr->aligned_size);
            }
        }
    }

    return ret;
}

static int inference_decoder_model(rknn_app_context_t *app_ctx)
{
    int ret = 0;

    // Sync inputs
    ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[0].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0)
    {
        printf("rknn3_mem_sync decoder input fail! ret=%d\n", ret);
        return ret;
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, 1, app_ctx->outputs, 1);
    if (ret < 0)
    {
        printf("rknn3_run decoder fail! ret=%d\n", ret);
        return ret;
    }

    // Sync outputs
    ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[0].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret < 0)
    {
        printf("rknn3_mem_sync decoder output fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}

static int inference_joiner_model(rknn_app_context_t *app_ctx)
{
    int ret = 0;

    // Sync inputs
    for (int i = 0; i < 2; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync joiner input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, 2, app_ctx->outputs, 1);
    if (ret < 0)
    {
        printf("rknn3_run joiner fail! ret=%d\n", ret);
        return ret;
    }

    // Sync outputs
    ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[0].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret < 0)
    {
        printf("rknn3_mem_sync joiner output fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}

static int greedy_search(rknn_zipformer_context_t *app_ctx, void *encoder_input, void *encoder_output, void *decoder_output, std::vector<int64_t> &hyp, int32_t *hyp_int32,
                         void *joiner_output, VocabEntry *vocab, std::vector<std::string> &recognized_text, std::vector<float> &timestamp, int num_processed_frames, int &frame_offset,
                         bool encoder_output_fp16, bool decoder_output_fp16, bool joiner_output_fp16,
                         float *encoder_output_fp32, float *decoder_output_fp32, float *joiner_output_fp32)
{
    int ret = 0;

    // Track segment count for debugging
    static int segment_count = 0;
    if (num_processed_frames == 0) {
        segment_count = 0;
    } else {
        segment_count = num_processed_frames / N_OFFSET;
    }

    // Set encoder input
    size_t encoder_input_size = (app_ctx->encoder_context.inputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16) ?
                               app_ctx->encoder_context.inputs[0].attr->n_elems * sizeof(float16) :
                               app_ctx->encoder_context.inputs[0].attr->n_elems * sizeof(float);
    memcpy(app_ctx->encoder_context.inputs[0].mem->virt_addr, encoder_input, encoder_input_size);

    ret = inference_encoder_model(&app_ctx->encoder_context);
    if (ret < 0)
    {
        printf("inference_encoder_model fail! ret=%d\n", ret);
        return ret;
    }

    // Convert encoder output to fp32 if needed
    if (encoder_output_fp16) {
        float16 *encoder_out_fp16 = (float16 *)app_ctx->encoder_context.outputs[0].mem->virt_addr;
        for (int i = 0; i < app_ctx->encoder_context.outputs[0].attr->n_elems; i++) {
            encoder_output_fp32[i] = fp16_to_fp32(encoder_out_fp16[i]);
        }
    }

    if (num_processed_frames == 0)
    {
        // Initialize decoder with blank context (Python: hyp = [blank_id] * context_size)
        hyp.clear();
        for (int i = 0; i < CONTEXT_SIZE; i++) {
            hyp.push_back(BLANK_ID);
        }

        // Get last CONTEXT_SIZE elements for decoder input (Python: hyp[-context_size:])
        std::vector<int64_t> decoder_input(hyp.end() - CONTEXT_SIZE, hyp.end());

        // Convert to int32 if needed
        if (app_ctx->decoder_context.inputs[0].attr->dtype == RKNN3_TENSOR_INT32) {
            for (int i = 0; i < CONTEXT_SIZE; i++) {
                hyp_int32[i] = (int32_t)decoder_input[i];
            }
            memcpy(app_ctx->decoder_context.inputs[0].mem->virt_addr, hyp_int32,
                   app_ctx->decoder_context.inputs[0].attr->n_elems * sizeof(int32_t));
        } else {
            memcpy(app_ctx->decoder_context.inputs[0].mem->virt_addr, decoder_input.data(),
                   app_ctx->decoder_context.inputs[0].attr->n_elems * sizeof(int64_t));
        }

        ret = inference_decoder_model(&app_ctx->decoder_context);
        if (ret < 0)
        {
            printf("inference_decoder_model fail! ret=%d\n", ret);
            return ret;
        }

        // Convert decoder output to fp32 if needed
        if (decoder_output_fp16) {
            float16 *decoder_out_fp16 = (float16 *)app_ctx->decoder_context.outputs[0].mem->virt_addr;
            for (int i = 0; i < app_ctx->decoder_context.outputs[0].attr->n_elems; i++) {
                decoder_output_fp32[i] = fp16_to_fp32(decoder_out_fp16[i]);
            }
        }
    }

    for (int i = 0; i < ENCODER_OUTPUT_T; i++)
    {
        float *cur_encoder_output = encoder_output_fp32 + i * DECODER_DIM;

        // Convert to fp16 if needed for joiner input
        if (app_ctx->joiner_context.inputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16) {
            float16 *encoder_input_fp16 = (float16 *)app_ctx->joiner_context.inputs[0].mem->virt_addr;
            for (int j = 0; j < DECODER_DIM; j++) {
                encoder_input_fp16[j] = fp32_to_fp16(cur_encoder_output[j]);
            }
        } else {
            memcpy(app_ctx->joiner_context.inputs[0].mem->virt_addr, cur_encoder_output, DECODER_DIM * sizeof(float));
        }

        if (app_ctx->joiner_context.inputs[1].attr->dtype == RKNN3_TENSOR_FLOAT16) {
            float16 *decoder_input_fp16 = (float16 *)app_ctx->joiner_context.inputs[1].mem->virt_addr;
            for (int j = 0; j < app_ctx->decoder_context.outputs[0].attr->n_elems; j++) {
                decoder_input_fp16[j] = fp32_to_fp16(decoder_output_fp32[j]);
            }
        } else {
            memcpy(app_ctx->joiner_context.inputs[1].mem->virt_addr, decoder_output_fp32, app_ctx->decoder_context.outputs[0].attr->n_elems * sizeof(float));
        }

        ret = inference_joiner_model(&app_ctx->joiner_context);
        if (ret < 0)
        {
            printf("inference_joiner_model fail! ret=%d\n", ret);
            return ret;
        }

        // Convert joiner output to fp32 if needed
        if (joiner_output_fp16) {
            float16 *joiner_out_fp16 = (float16 *)app_ctx->joiner_context.outputs[0].mem->virt_addr;
            for (int j = 0; j < app_ctx->joiner_context.outputs[0].attr->n_elems; j++) {
                joiner_output_fp32[j] = fp16_to_fp32(joiner_out_fp16[j]);
            }
        }

        int next_token = argmax(joiner_output_fp32, app_ctx->joiner_context.outputs[0].attr->n_elems);

        if (next_token != BLANK_ID && next_token != UNK_ID)
        {
            timestamp.push_back(frame_offset + i);

            // Add new token to hyp (Python: hyp.append(y))
            hyp.push_back((int64_t)next_token);

            std::string next_token_str = vocab[next_token].token;
            replace_substr(next_token_str, "▁", " ");
            recognized_text.push_back(next_token_str);

            // Get last CONTEXT_SIZE elements for decoder input (Python: hyp[-context_size:])
            std::vector<int64_t> decoder_input(hyp.end() - CONTEXT_SIZE, hyp.end());

            // Update decoder with new context
            if (app_ctx->decoder_context.inputs[0].attr->dtype == RKNN3_TENSOR_INT32) {
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    hyp_int32[j] = (int32_t)decoder_input[j];
                }
                memcpy(app_ctx->decoder_context.inputs[0].mem->virt_addr, hyp_int32,
                       app_ctx->decoder_context.inputs[0].attr->n_elems * sizeof(int32_t));
            } else {
                memcpy(app_ctx->decoder_context.inputs[0].mem->virt_addr, decoder_input.data(),
                       app_ctx->decoder_context.inputs[0].attr->n_elems * sizeof(int64_t));
            }

            ret = inference_decoder_model(&app_ctx->decoder_context);
            if (ret < 0)
            {
                printf("inference_decoder_model fail! ret=%d\n", ret);
                return ret;
            }

            // Convert decoder output to fp32 if needed
            if (decoder_output_fp16) {
                float16 *decoder_out_fp16 = (float16 *)app_ctx->decoder_context.outputs[0].mem->virt_addr;
                for (int j = 0; j < app_ctx->decoder_context.outputs[0].attr->n_elems; j++) {
                    decoder_output_fp32[j] = fp16_to_fp32(decoder_out_fp16[j]);
                }
            }
        }
    }

    frame_offset += ENCODER_OUTPUT_T;

    return ret;
}

int inference_zipformer_model(rknn_zipformer_context_t *app_ctx, audio_buffer_t audio, VocabEntry *vocab, std::vector<std::string> &recognized_text,
                              std::vector<float> &timestamp, float &audio_length)
{
    int ret;
    recognized_text.clear();
    timestamp.clear();

    // Check data types and allocate buffers accordingly
    bool encoder_input_fp16 = (app_ctx->encoder_context.inputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16);
    bool encoder_output_fp16 = (app_ctx->encoder_context.outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16);
    bool decoder_output_fp16 = (app_ctx->decoder_context.outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16);
    bool joiner_output_fp16 = (app_ctx->joiner_context.outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16);

    // Allocate buffers for inputs and outputs
    float *encoder_input_fp32 = (float *)malloc(app_ctx->encoder_context.inputs[0].attr->n_elems * sizeof(float));
    if (encoder_input_fp32 == NULL) {
        printf("malloc encoder_input_fp32 failed!\n");
        return -1;
    }
    void *encoder_input = encoder_input_fp32; // Will be converted to fp16 if needed

    float *encoder_output_fp32 = (float *)malloc(app_ctx->encoder_context.outputs[0].attr->n_elems * sizeof(float));
    if (encoder_output_fp32 == NULL) {
        printf("malloc encoder_output_fp32 failed!\n");
        free(encoder_input_fp32);
        encoder_input_fp32 = NULL;
        return -1;
    }
    void *encoder_output = encoder_output_fp32;

    std::vector<int64_t> hyp;  // Use vector instead of malloc to match Python's growing list
    int32_t *hyp_int32 = (int32_t *)malloc(app_ctx->decoder_context.inputs[0].attr->n_elems * sizeof(int32_t));
    if (hyp_int32 == NULL) {
        printf("malloc hyp_int32 failed!\n");
        free(encoder_input_fp32);
        free(encoder_output_fp32);
        return -1;
    }

    float *decoder_output_fp32 = (float *)malloc(app_ctx->decoder_context.outputs[0].attr->n_elems * sizeof(float));
    if (decoder_output_fp32 == NULL) {
        printf("malloc decoder_output_fp32 failed!\n");
        free(encoder_input_fp32);
        free(encoder_output_fp32);
        free(hyp_int32);
        return -1;
    }
    void *decoder_output = decoder_output_fp32;

    float *joiner_output_fp32 = (float *)malloc(app_ctx->joiner_context.outputs[0].attr->n_elems * sizeof(float));
    if (joiner_output_fp32 == NULL) {
        printf("malloc joiner_output_fp32 failed!\n");
        free(encoder_input_fp32);
        free(encoder_output_fp32);
        free(hyp_int32);
        free(decoder_output_fp32);
        return -1;
    }
    void *joiner_output = joiner_output_fp32;

    // Allocate fp16 buffers if needed
    float16 *encoder_input_fp16_buf = NULL;
    float16 *encoder_output_fp16_buf = NULL;
    float16 *decoder_output_fp16_buf = NULL;
    float16 *joiner_output_fp16_buf = NULL;

    if (encoder_input_fp16) {
        encoder_input_fp16_buf = (float16 *)malloc(app_ctx->encoder_context.inputs[0].attr->n_elems * sizeof(float16));
        encoder_input = encoder_input_fp16_buf;
    }
    if (encoder_output_fp16) {
        encoder_output_fp16_buf = (float16 *)malloc(app_ctx->encoder_context.outputs[0].attr->n_elems * sizeof(float16));
        encoder_output = encoder_output_fp16_buf;
    }
    if (decoder_output_fp16) {
        decoder_output_fp16_buf = (float16 *)malloc(app_ctx->decoder_context.outputs[0].attr->n_elems * sizeof(float16));
        decoder_output = decoder_output_fp16_buf;
    }
    if (joiner_output_fp16) {
        joiner_output_fp16_buf = (float16 *)malloc(app_ctx->joiner_context.outputs[0].attr->n_elems * sizeof(float16));
        joiner_output = joiner_output_fp16_buf;
    }

    // Initialize cached states for encoder
    for (int i = 1; i < app_ctx->encoder_context.io_num.n_input; i++) {
        if (app_ctx->encoder_context.inputs[i].attr->dtype == RKNN3_TENSOR_INT32 ||
            app_ctx->encoder_context.inputs[i].attr->dtype == RKNN3_TENSOR_INT64) {
            // Initialize integer types to zero
            memset(app_ctx->encoder_context.inputs[i].mem->virt_addr, 0,
                   app_ctx->encoder_context.inputs[i].attr->aligned_size);
        } else {
            // Initialize FP types to zero
            memset(app_ctx->encoder_context.inputs[i].mem->virt_addr, 0,
                   app_ctx->encoder_context.inputs[i].attr->aligned_size);
        }
    }

    knf::FbankOptions fbank_opts;
    fbank_opts.frame_opts.samp_freq = 16000;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.mel_opts.high_freq = -400;
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.frame_opts.snip_edges = false;
    knf::OnlineFbank fbank(fbank_opts);

    int num_frames = 0;
    int num_processed_frames = 0;
    int offset = N_OFFSET;
    int segment = N_SEGMENT;
    float tail_pad_length = 0.0; // sec
    fbank.AcceptWaveform(SAMPLE_RATE, audio.data, audio.num_frames);
    num_frames = fbank.NumFramesReady();
    int frame_offset = 0;

    while ((num_frames - num_processed_frames) > 0)
    {
        if ((num_frames - num_processed_frames) < segment)
        {
            tail_pad_length = (segment - (num_frames - num_processed_frames)) / 100.0f; // sec
            std::vector<float> tail_paddings(int(tail_pad_length * SAMPLE_RATE));
            fbank.AcceptWaveform(SAMPLE_RATE, tail_paddings.data(), tail_paddings.size());
            fbank.InputFinished();
        }
        ret = get_fbank_frames(&fbank, num_processed_frames, segment, encoder_input_fp32);
        if (ret < 0)
        {
            break;
        }

        // Convert encoder input to fp16 if needed
        if (encoder_input_fp16) {
            for (int i = 0; i < app_ctx->encoder_context.inputs[0].attr->n_elems; i++) {
                encoder_input_fp16_buf[i] = fp32_to_fp16(encoder_input_fp32[i]);
            }
        }

        ret = greedy_search(app_ctx, encoder_input, encoder_output, decoder_output, hyp, hyp_int32, joiner_output, vocab, recognized_text, timestamp, num_processed_frames, frame_offset,
                           encoder_output_fp16, decoder_output_fp16, joiner_output_fp16,
                           encoder_output_fp32, decoder_output_fp32, joiner_output_fp32);
        if (ret < 0)
        {
            printf("greedy_search fail! ret=%d\n", ret);
            goto out;
        }
        num_processed_frames += offset;
    }

    audio_length = (float)audio.num_frames / audio.sample_rate + tail_pad_length;

out:
    if (encoder_input_fp32) {
        free(encoder_input_fp32);
        encoder_input_fp32 = NULL;
    }
    if (encoder_output_fp32) {
        free(encoder_output_fp32);
        encoder_output_fp32 = NULL;
    }
    if (hyp_int32) {
        free(hyp_int32);
        hyp_int32 = NULL;
    }
    if (decoder_output_fp32) {
        free(decoder_output_fp32);
        decoder_output_fp32 = NULL;
    }
    if (joiner_output_fp32) {
        free(joiner_output_fp32);
        joiner_output_fp32 = NULL;
    }
    if (encoder_input_fp16_buf) {
        free(encoder_input_fp16_buf);
        encoder_input_fp16_buf = NULL;
    }
    if (encoder_output_fp16_buf) {
        free(encoder_output_fp16_buf);
        encoder_output_fp16_buf = NULL;
    }
    if (decoder_output_fp16_buf) {
        free(decoder_output_fp16_buf);
        decoder_output_fp16_buf = NULL;
    }
    if (joiner_output_fp16_buf) {
        free(joiner_output_fp16_buf);
        joiner_output_fp16_buf = NULL;
    }

    return ret;
}