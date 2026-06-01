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

#include "zipformer.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void convert_nchw_to_nhwc(float *src, float *dst, int N, int channels, int height, int width)
{
    if (src == nullptr || dst == nullptr) {
        fprintf(stderr, "Error: NULL pointer passed to convert_nchw_to_nhwc\n");
        return;
    }

    if (N <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in convert_nchw_to_nhwc: N=%d, channels=%d, height=%d, width=%d\n",
                N, channels, height, width);
        return;
    }

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    dst[n * height * width * channels + h * width * channels + w * channels + c] = src[n * channels * height * width + c * height * width + h * width + w];
                }
            }
        }
    }
}

int get_fbank_frames(knf::OnlineFbank *fbank, int frame_index, int segment, float *frames)
{
    if (fbank == nullptr || frames == nullptr) {
        fprintf(stderr, "Error: NULL parameter passed to get_fbank_frames\n");
        return -1;
    }

    if (frame_index + segment > fbank->NumFramesReady())
    {
        return -1;
    }

    for (int i = 0; i < segment; ++i)
    {
        const float *frame = fbank->GetFrame(i + frame_index);
        memcpy(frames + i * N_MELS, frame, N_MELS * sizeof(float));
    }

    return 0;
}

/**
 * @brief Find the index of the maximum value in an array
 * @param array Input float array
 * @param size Size of the array
 * @return Index of maximum value (0 to size-1), or -1 if invalid input
 * @note Returns -1 for NULL array, size <= 0, or empty array
 */
int argmax(float *array, int size)
{
    // Check bounds first before any array access to prevent potential out-of-bounds access
    if (array == NULL || size <= 0) {
        return -1;  // Invalid input
    }

    int max_index = 0;
    float max_value = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max_value)
        {
            max_value = array[i];
            max_index = i;
        }
    }
    return max_index;
}

void replace_substr(std::string &str, const std::string &from, const std::string &to)
{
    if (from.empty())
        return; // Prevent infinite loop if 'from' is empty
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Advance position by length of the replacement
    }
}

int read_vocab(const char *fileName, VocabEntry *vocab)
{
    FILE *fp;
    char line[512];

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        perror("Error opening file");
        return -1;
    }

    int count = 0;
    while (fgets(line, sizeof(line), fp))
    {
        // Handle line truncation
        size_t line_len = strlen(line);
        if (line_len > 0 && line[line_len - 1] != '\n' && !feof(fp)) {
            // Line was truncated, skip remaining characters
            int c;
            while ((c = fgetc(fp)) != '\n' && c != EOF);
        }

        // Find space position safely
        char *space_pos = strchr(line, ' ');
        if (space_pos == NULL) {
            fprintf(stderr, "Invalid line format in vocab file at line %d\n", count + 1);
            continue;
        }

        vocab[count].index = atoi(space_pos + 1);

        // Use strtok_r for thread safety
        char *saveptr = NULL;  // Initialize to NULL for thread safety
        char *token = strtok_r(line, " ", &saveptr);
        if (token != NULL) {
            vocab[count].token = strdup(token);
            if (vocab[count].token == NULL) {
                perror("Memory allocation failed for token");
                // Clean up previously allocated memory
                for (int i = 0; i < count; i++) {
                    if (vocab[i].token) {
                        free(vocab[i].token);
                        vocab[i].token = NULL;
                    }
                }
                fclose(fp);
                return -1;
            }
        }

        count++;
    }

    fclose(fp);

    return 0;
}
