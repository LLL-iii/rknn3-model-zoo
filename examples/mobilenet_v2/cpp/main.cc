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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mobilenet.h"
#include "image_utils.h"
#include "file_utils.h"

#define IMAGENET_CLASSES_FILE "./model/synset.txt"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("%s <model_path> <weight_path> <image_path> <core_mask>\n", argv[0]);
        return -1;
    }

    const char* model_path  = argv[1];
    const char* weight_path = argv[2];
    const char* image_path  = argv[3];
    uint32_t    core_mask   = strtoul(argv[4], nullptr, 16);

    int line_count;
    char** lines = read_lines_from_file(IMAGENET_CLASSES_FILE, &line_count);
    if (lines == NULL) {
        printf("read classes label file fail! path=%s\n", IMAGENET_CLASSES_FILE);
        return -1;
    }

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_mobilenet_model(model_path, weight_path, &rknn_app_ctx, core_mask);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0) {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        return -1;
    }

    int topk = 5;
    mobilenet_result result[topk];

    ret = inference_mobilenet_model(&rknn_app_ctx, &src_image, result, topk);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d\n", ret);
        goto out;
    }

    for (int i = 0; i < topk; i++) {
        printf("[%d] score=%.6f class=%s\n", result[i].cls, result[i].score, lines[result[i].cls]);
    }

out:
    ret = release_mobilenet_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_mobilenet_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    if (lines != NULL) {
        free_lines(lines, line_count);
    }

    return 0;
}
