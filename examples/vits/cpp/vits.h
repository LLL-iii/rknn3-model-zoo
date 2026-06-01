#ifndef VITS_H_
#define VITS_H_

#include "rknn3_api.h"
#include "text_processor.h"
#include <vector>
#include <string>

// VITS inference context
typedef struct {
    rknn3_context ctx_step1;
    rknn3_context ctx_step2;
    rknn3_input_output_num step1_io_num;
    rknn3_input_output_num step2_io_num;
    rknn3_tensor* step1_inputs;
    rknn3_tensor* step1_outputs;
    rknn3_tensor* step2_inputs;
    rknn3_tensor* step2_outputs;
    bool initialized;

    // Configuration
    int sampling_rate;
    int hop_length;
    int text_window_size;
    int text_context_size;
    int text_max_len;
    int audio_window_size;
    float noise_scale;
    float length_scale;

    // Multi-speaker support
    bool is_multi_speaker;
    int speaker_id;
    int num_speakers;
} vits_context_t;

// Initialize VITS model
int init_vits_model(const char* step1_model, const char* step1_weight,
                    const char* step2_model, const char* step2_weight,
                    vits_context_t* ctx, uint32_t core_mask = 0x1,
                    int speaker_id = 0);

// Release VITS model
int release_vits_model(vits_context_t* ctx);

// Run VITS inference
int inference_vits(vits_context_t* ctx, TextProcessor* processor,
                  const std::string& text, std::vector<float>& audio_output);

// Save audio as WAV file
int save_wav(const char* filename, const std::vector<float>& audio, int sample_rate);

// Helper functions
std::vector<std::vector<std::vector<float>>> GeneratePath(
    const std::vector<std::vector<float>>& w_ceil,
    int b, int t_x, int t_y);

#endif  // VITS_H_
