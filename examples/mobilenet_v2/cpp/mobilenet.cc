// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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
#include <math.h>

#include "mobilenet.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

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

static float deqnt_i8_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

typedef struct {
    float value;
    int index;
} element_t;

static void swap(element_t* a, element_t* b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

static int partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

static void quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

static void softmax_fp(float* array, int size) {
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}

static int convert_int8_to_fp32(const void *src, float *dst, int n_elems, float scale, int32_t zero_point)
{

    for (int i = 0; i < n_elems; i++)
    {
        dst[i] = ((float)((int8_t *)src)[i] - zero_point) * scale;
    }

    return 0;
}

static int convert_fp16_to_fp32(const float16* src, float* dst, int n_elems)
{
    for (int i = 0; i < n_elems; i++)
    {
        dst[i] = fp16_to_fp32(src[i]);
    }
    return 0;
}

static void softmax_i8(int8_t* array, int size, float scale, int32_t zp, float* outputs) {
    // Find the maximum value after dequantization
    float max_val = deqnt_i8_to_f32(array[0], zp, scale);
    for (int i = 1; i < size; i++) {
        float val = deqnt_i8_to_f32(array[i], zp, scale);
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = deqnt_i8_to_f32(array[i], zp, scale) - max_val;
        outputs[i] = expf(val);
        sum += outputs[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        outputs[i] /= sum;
    }
}

static void get_topk_with_indices(float arr[], int size, int k, mobilenet_result* result) {

    // Create an array of elements, saving values ​​and index numbers
    element_t* elements = (element_t*)malloc(size * sizeof(element_t));
    for (int i = 0; i < size; i++) {
        elements[i].value = arr[i];
        elements[i].index = i;
    }

    // Quick sort an array of elements
    quick_sort(elements, 0, size - 1);

    // Get the top K maximum values ​​and their index numbers
    for (int i = 0; i < k; i++) {
        result[i].score = elements[i].value;
        result[i].cls = elements[i].index;
    }

    free(elements);
}


int init_mobilenet_model(const char* model_path, const char* weight_path, rknn_app_context_t* app_ctx, uint32_t core_mask)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn3_context ctx = 0;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("load_model fail!\n");
        return -1;
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
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (output_attrs[0].qnt_type == RKNN3_TENSOR_PER_LAYER_ASYMMETRIC && output_attrs[0].dtype == RKNN3_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_channel = input_attrs[0].shape[3];
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
    }
    else
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
        app_ctx->model_channel = input_attrs[0].shape[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_mobilenet_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx == NULL || app_ctx->rknn_ctx == 0)
    {
        return -1;
    }
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->inputs && app_ctx->inputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
        }
        if (app_ctx->inputs && app_ctx->inputs[i].attr != NULL) {
            free(app_ctx->inputs[i].attr);
            app_ctx->inputs[i].attr = NULL;
        }
    }
    if (app_ctx->inputs) {
        free(app_ctx->inputs);
        app_ctx->inputs = NULL;
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->outputs && app_ctx->outputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
        }
        if (app_ctx->outputs && app_ctx->outputs[i].attr != NULL) {
            free(app_ctx->outputs[i].attr);
            app_ctx->outputs[i].attr = NULL;
        }
    }
    if (app_ctx->outputs) {
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

int inference_mobilenet_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, mobilenet_result* out_result, int topk)
{
    int ret = 0;
    image_buffer_t img;
    size_t dst_elems = 1;
    float *output_data = NULL;

    memset(&img, 0, sizeof(image_buffer_t));

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    img.format = IMAGE_FORMAT_RGB888;
    img.size = get_image_size(&img);
    img.virt_addr = (unsigned char*)malloc(img.size);
    if (img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }

    ret = convert_image(src_img, &img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        ret = -1;
        goto out;
    }

    // Set Input Data
    memcpy(app_ctx->inputs[0].mem->virt_addr, (uint8_t*)img.virt_addr, img.size);

    // Sync inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input, app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret != RKNN3_SUCCESS)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    for (uint32_t j = 0; j < app_ctx->outputs[0].attr->n_dims; j++)
    {
        dst_elems *= app_ctx->outputs[0].attr->shape[j];
    }

    output_data = (float *)malloc(dst_elems * sizeof(float));
    if (output_data == NULL)
    {
        printf("Failed to allocate memory for output_data\n");
        ret = -1;
        goto out;
    }

    if (app_ctx->outputs[0].attr->layout == RKNN3_TENSOR_NCHW || app_ctx->outputs[0].attr->layout == RKNN3_TENSOR_UNDEFINED)
    {
        if (app_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_INT8)
        {
            float scale = app_ctx->outputs[0].attr->qnt_info.scale;
            int32_t zero_point = app_ctx->outputs[0].attr->qnt_info.zero_point;
            convert_int8_to_fp32(app_ctx->outputs[0].mem->virt_addr, output_data, dst_elems, scale, zero_point);
        }
        else if (app_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16)
        {
            convert_fp16_to_fp32((const float16 *)app_ctx->outputs[0].mem->virt_addr, output_data, dst_elems);
        }
        else if (app_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT32)
        {
            memcpy(output_data, app_ctx->outputs[0].mem->virt_addr, dst_elems * sizeof(float));
        }
        else
        {
            printf("Unsupported type for NCHW format: %s\n", rknn3_get_type_string(app_ctx->outputs[0].attr->dtype));
            goto out;
        }
    }
    else
    {
        printf("Unsupported output format: %s\n", rknn3_get_layout_string(app_ctx->outputs[0].attr->layout));
        goto out;
    }

    softmax_fp(output_data, app_ctx->outputs[0].attr->n_elems);
    get_topk_with_indices(output_data, app_ctx->outputs[0].attr->n_elems, topk, out_result);

out:
    if (output_data != NULL)
    {
        free(output_data);
        output_data = NULL;
    }
    if (img.virt_addr != NULL)
    {
        free(img.virt_addr);
        img.virt_addr = NULL;
    }

    return ret;
}