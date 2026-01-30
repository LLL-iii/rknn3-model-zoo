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
#include <iostream>
#include <fstream>
#include <sstream>

#include "mobilenet.h"
#include "image_utils.h"
#include "file_utils.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("%s <model_path> <weight_path> <data_root> <core_mask>\n", argv[0]);
        return -1;
    }

    const char* model_path  = argv[1];
    const char* weight_path = argv[2];
    const char* data_root  = argv[3];
    uint32_t    core_mask   = strtoul(argv[4], nullptr, 16);

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_mobilenet_model(model_path, weight_path, &rknn_app_ctx, core_mask);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    image_buffer_t src_image;

    char label_path[1024];
    sprintf(label_path, "%s/ILSVRC2012_img_val_256.txt", data_root);
    std::ifstream file(label_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件!" << std::endl;
        return 1;
    }

    std::string line;
    // 逐行读取文件
    int total = 0;
    int correct_1 = 0;
    int correct_5 = 0;
    while (std::getline(file, line)) {
        // 查找逗号位置
        size_t commaPos = line.find(',');
        if (commaPos == std::string::npos) {
            std::cerr << "无效的行格式: " << line << std::endl;
            continue;
        }
        
        // 提取路径（逗号前的部分）
        std::string path = line.substr(0, commaPos);
        
        // 提取标签（逗号后的部分），注意跳过可能的空格
        std::string labelStr = line.substr(commaPos + 1);
        // 去除前后空格
        size_t start = labelStr.find_first_not_of(" \t");
        size_t end = labelStr.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            labelStr = labelStr.substr(start, end - start + 1);
        }
        
        // 转换为整数
        int label;
        try {
            label = std::stoi(labelStr);
        } catch (...) {
            std::cerr << "无法将 " << labelStr << " 转换为整数: " << line << std::endl;
            continue;
        }

        memset(&src_image, 0, sizeof(image_buffer_t));
        char image_path[1024];
        sprintf(image_path, "%s/%s", data_root, path.c_str());
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

        total++;
        if (result[0].cls == label) {
            correct_1++;
        }
        for (int i = 0; i < topk; i++) {
            // printf("[%d] score=%.6f\n", result[i].cls, result[i].score);
            if (result[i].cls == label) {
                correct_5++;
                break;
            }
        }
        printf("run: %d\n", total);
        printf("correct_1: %d\n", correct_1);
        printf("correct_5: %d\n", correct_5);
        printf("top1: %.4f\n", (float)correct_1 / total);
        printf("top5: %.4f\n", (float)correct_5 / total);

        if (src_image.virt_addr != NULL)
        {
            free(src_image.virt_addr);
            src_image.virt_addr = NULL;
        }
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

    return 0;
}
