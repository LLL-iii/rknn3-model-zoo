#include "vits.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include "float16.h"

#define DEFAULT_SAMPLING_RATE 22050
#define DEFAULT_HOP_LENGTH 256
#define DEFAULT_TEXT_WINDOW_SIZE 128
#define DEFAULT_TEXT_CONTEXT_SIZE 16
#define DEFAULT_AUDIO_WINDOW_SIZE 128
#define DEFAULT_NOISE_SCALE 0.667f
#define DEFAULT_LENGTH_SCALE 1.1f     // 稍微减慢语速（原值1.0）
#define DEFAULT_NOISE_SCALE_W 0.8f

// Dump tensor attributes for debugging (from zipformer)
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

// Initialize VITS model
int init_vits_model(const char* step1_model, const char* step1_weight,
                    const char* step2_model, const char* step2_weight,
                    vits_context_t* ctx, uint32_t core_mask, int speaker_id) {
    int ret;
    rknn3_config config;
    rknn3_devices devs;

    // Find available devices
    memset(&devs, 0, sizeof(devs));
    ret = rknn3_find_devices(&devs);
    if (ret < 0 || devs.n_devices == 0) {
        printf("rknn3_find_devices failed: %d\n", ret);
        return -1;
    }
    printf("Found %d RKNN3 devices\n", devs.n_devices);

    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // Initialize context
    memset(ctx, 0, sizeof(vits_context_t));
    ctx->sampling_rate = DEFAULT_SAMPLING_RATE;
    ctx->hop_length = DEFAULT_HOP_LENGTH;
    ctx->text_window_size = DEFAULT_TEXT_WINDOW_SIZE;
    ctx->text_context_size = DEFAULT_TEXT_CONTEXT_SIZE;
    ctx->text_max_len = DEFAULT_TEXT_WINDOW_SIZE + 2 * DEFAULT_TEXT_CONTEXT_SIZE;
    ctx->audio_window_size = DEFAULT_AUDIO_WINDOW_SIZE;
    ctx->noise_scale = DEFAULT_NOISE_SCALE;
    ctx->length_scale = DEFAULT_LENGTH_SCALE;

    // Initialize Step1 model
    printf("========================================\n");
    printf("Initializing Step1 Model\n");
    printf("========================================\n");
    printf("Step1 model path: %s\n", step1_model);
    printf("Step1 weight path: %s\n", step1_weight);

    rknn3_init_extend init_extend = {0};
    init_extend.device_id = devs.devices[0].id;
    printf("Device ID: %s\n", devs.devices[0].id);

    printf("Calling rknn3_init for step1...\n");
    ret = rknn3_init(&ctx->ctx_step1, &init_extend);
    if (ret < 0) {
        printf("ERROR: rknn3_init step1 failed: %d\n", ret);
        return -1;
    }
    printf("rknn3_init step1 success\n");

    printf("Loading step1 model from path...\n");
    ret = rknn3_load_model_from_path(ctx->ctx_step1, step1_model, step1_weight);
    if (ret < 0) {
        printf("ERROR: rknn3_load_model_from_path step1 failed: %d\n", ret);
        printf("Model path: %s\n", step1_model);
        printf("Weight path: %s\n", step1_weight);
        return -1;
    }
    printf("Step1 model loaded successfully\n");

    printf("Calling rknn3_model_init for step1...\n");
    ret = rknn3_model_init(ctx->ctx_step1, &config);
    if (ret < 0) {
        printf("ERROR: rknn3_model_init step1 failed: %d\n", ret);
        return -1;
    }
    printf("Step1 model initialized successfully\n");

    // Get Step1 I/O info
    ret = rknn3_query(ctx->ctx_step1, RKNN3_QUERY_IN_OUT_NUM, &ctx->step1_io_num, sizeof(ctx->step1_io_num));
    if (ret < 0) {
        printf("rknn3_query step1 io_num failed: %d\n", ret);
        return -1;
    }

    // Auto-detect model type based on Step1 output count
    bool is_multi_speaker = (ctx->step1_io_num.n_output == 5);
    ctx->is_multi_speaker = is_multi_speaker;
    ctx->speaker_id = speaker_id;
    ctx->num_speakers = is_multi_speaker ? 109 : 1;

    printf("========================================\n");
    printf("Model Detection Results\n");
    printf("========================================\n");
    printf("Step1 outputs: %u\n", ctx->step1_io_num.n_output);
    printf("Model type: %s\n", is_multi_speaker ? "VCTK (multi-speaker)" : "LJSpeech (single-speaker)");
    printf("Speaker ID: %d\n", speaker_id);
    printf("Total speakers: %d\n", ctx->num_speakers);
    printf("========================================\n\n");

    // Temporary storage for tensor attributes
    std::vector<rknn3_tensor_attr> step1_input_attrs(ctx->step1_io_num.n_input);
    std::vector<rknn3_tensor_attr> step1_output_attrs(ctx->step1_io_num.n_output);

    for (uint32_t i = 0; i < ctx->step1_io_num.n_input; i++) {
        step1_input_attrs[i].index = i;
        ret = rknn3_query(ctx->ctx_step1, RKNN3_QUERY_INPUT_ATTR, &step1_input_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query step1 input %d failed: %d\n", i, ret);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->step1_io_num.n_output; i++) {
        step1_output_attrs[i].index = i;
        ret = rknn3_query(ctx->ctx_step1, RKNN3_QUERY_OUTPUT_ATTR, &step1_output_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query step1 output %d failed: %d\n", i, ret);
            return -1;
        }
    }

    // Initialize Step2 model
    printf("\n========================================\n");
    printf("Initializing Step2 Model\n");
    printf("========================================\n");
    printf("Step2 model path: %s\n", step2_model);
    printf("Step2 weight path: %s\n", step2_weight);

    init_extend.device_id = devs.devices[0].id;
    printf("Calling rknn3_init for step2...\n");
    ret = rknn3_init(&ctx->ctx_step2, &init_extend);
    if (ret < 0) {
        printf("ERROR: rknn3_init step2 failed: %d\n", ret);
        return -1;
    }
    printf("rknn3_init step2 success\n");

    printf("Loading step2 model from path...\n");
    ret = rknn3_load_model_from_path(ctx->ctx_step2, step2_model, step2_weight);
    if (ret < 0) {
        printf("ERROR: rknn3_load_model_from_path step2 failed: %d\n", ret);
        printf("Model path: %s\n", step2_model);
        printf("Weight path: %s\n", step2_weight);
        return -1;
    }
    printf("Step2 model loaded successfully\n");

    printf("Calling rknn3_model_init for step2...\n");
    ret = rknn3_model_init(ctx->ctx_step2, &config);
    if (ret < 0) {
        printf("ERROR: rknn3_model_init step2 failed: %d\n", ret);
        return -1;
    }
    printf("Step2 model initialized successfully\n");

    // Get Step2 I/O info
    ret = rknn3_query(ctx->ctx_step2, RKNN3_QUERY_IN_OUT_NUM, &ctx->step2_io_num, sizeof(ctx->step2_io_num));
    if (ret < 0) {
        printf("rknn3_query step2 io_num failed: %d\n", ret);
        return -1;
    }

    // Temporary storage for tensor attributes
    std::vector<rknn3_tensor_attr> step2_input_attrs(ctx->step2_io_num.n_input);
    std::vector<rknn3_tensor_attr> step2_output_attrs(ctx->step2_io_num.n_output);

    for (uint32_t i = 0; i < ctx->step2_io_num.n_input; i++) {
        step2_input_attrs[i].index = i;
        ret = rknn3_query(ctx->ctx_step2, RKNN3_QUERY_INPUT_ATTR, &step2_input_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query step2 input %d failed: %d\n", i, ret);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->step2_io_num.n_output; i++) {
        step2_output_attrs[i].index = i;
        ret = rknn3_query(ctx->ctx_step2, RKNN3_QUERY_OUTPUT_ATTR, &step2_output_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query step2 output %d failed: %d\n", i, ret);
            return -1;
        }
    }

    printf("Step1: %u inputs, %u outputs\n", ctx->step1_io_num.n_input, ctx->step1_io_num.n_output);
    printf("Step2: %u inputs, %u outputs\n", ctx->step2_io_num.n_input, ctx->step2_io_num.n_output);

    // Mark context as initialized for Step2 cleanup in case of Step1 failures
    ctx->initialized = true;

    // Allocate memory for Step1 tensors
    printf("Allocating memory for Step1 tensors...\n");
    ctx->step1_inputs = (rknn3_tensor*)malloc(ctx->step1_io_num.n_input * sizeof(rknn3_tensor));
    if (ctx->step1_inputs == NULL) {
        printf("malloc step1_inputs failed!\n");
        release_vits_model(ctx);
        return -1;
    }

    ctx->step1_outputs = (rknn3_tensor*)malloc(ctx->step1_io_num.n_output * sizeof(rknn3_tensor));
    if (ctx->step1_outputs == NULL) {
        printf("malloc step1_outputs failed!\n");
        release_vits_model(ctx);
        return -1;
    }

    // Initialize Step1 inputs
    for (uint32_t i = 0; i < ctx->step1_io_num.n_input; i++) {
        ctx->step1_inputs[i].mem = rknn3_create_mem(ctx->ctx_step1, step1_input_attrs[i].aligned_size, step1_input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (ctx->step1_inputs[i].mem == nullptr) {
            printf("Failed to create memory for Step1 input %d\n", i);
            release_vits_model(ctx);
            return -1;
        }

        ctx->step1_inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (ctx->step1_inputs[i].attr == NULL) {
            printf("malloc step1_inputs[%d].attr failed!\n", i);
            release_vits_model(ctx);
            return -1;
        }
        memcpy(ctx->step1_inputs[i].attr, &step1_input_attrs[i], sizeof(rknn3_tensor_attr));
    }

    // Initialize Step1 outputs
    for (uint32_t i = 0; i < ctx->step1_io_num.n_output; i++) {
        ctx->step1_outputs[i].mem = rknn3_create_mem(ctx->ctx_step1, step1_output_attrs[i].aligned_size, step1_output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (ctx->step1_outputs[i].mem == nullptr) {
            printf("Failed to create memory for Step1 output %d\n", i);
            release_vits_model(ctx);
            return -1;
        }

        ctx->step1_outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (ctx->step1_outputs[i].attr == NULL) {
            printf("malloc step1_outputs[%d].attr failed!\n", i);
            release_vits_model(ctx);
            return -1;
        }
        memcpy(ctx->step1_outputs[i].attr, &step1_output_attrs[i], sizeof(rknn3_tensor_attr));
    }

    // Allocate memory for Step2 tensors
    printf("Allocating memory for Step2 tensors...\n");
    ctx->step2_inputs = (rknn3_tensor*)malloc(ctx->step2_io_num.n_input * sizeof(rknn3_tensor));
    if (ctx->step2_inputs == NULL) {
        printf("malloc step2_inputs failed!\n");
        release_vits_model(ctx);
        return -1;
    }

    ctx->step2_outputs = (rknn3_tensor*)malloc(ctx->step2_io_num.n_output * sizeof(rknn3_tensor));
    if (ctx->step2_outputs == NULL) {
        printf("malloc step2_outputs failed!\n");
        release_vits_model(ctx);
        return -1;
    }

    // Initialize Step2 inputs
    for (uint32_t i = 0; i < ctx->step2_io_num.n_input; i++) {
        ctx->step2_inputs[i].mem = rknn3_create_mem(ctx->ctx_step2, step2_input_attrs[i].aligned_size, step2_input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (ctx->step2_inputs[i].mem == nullptr) {
            printf("Failed to create memory for Step2 input %d\n", i);
            release_vits_model(ctx);
            return -1;
        }

        ctx->step2_inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (ctx->step2_inputs[i].attr == NULL) {
            printf("malloc step2_inputs[%d].attr failed!\n", i);
            release_vits_model(ctx);
            return -1;
        }
        memcpy(ctx->step2_inputs[i].attr, &step2_input_attrs[i], sizeof(rknn3_tensor_attr));
    }

    // Initialize Step2 outputs
    for (uint32_t i = 0; i < ctx->step2_io_num.n_output; i++) {
        ctx->step2_outputs[i].mem = rknn3_create_mem(ctx->ctx_step2, step2_output_attrs[i].aligned_size, step2_output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (ctx->step2_outputs[i].mem == nullptr) {
            printf("Failed to create memory for Step2 output %d\n", i);
            release_vits_model(ctx);
            return -1;
        }

        ctx->step2_outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (ctx->step2_outputs[i].attr == NULL) {
            printf("malloc step2_outputs[%d].attr failed!\n", i);
            release_vits_model(ctx);
            return -1;
        }
        memcpy(ctx->step2_outputs[i].attr, &step2_output_attrs[i], sizeof(rknn3_tensor_attr));
    }

    ctx->initialized = true;
    printf("VITS models and memory initialized successfully\n");

    // Print detailed tensor information for debugging
    printf("\nStep1 inputs:\n");
    for (uint32_t i = 0; i < ctx->step1_io_num.n_input; i++) {
        dump_tensor_attr(ctx->step1_inputs[i].attr);
    }

    printf("\nStep1 outputs:\n");
    for (uint32_t i = 0; i < ctx->step1_io_num.n_output; i++) {
        dump_tensor_attr(ctx->step1_outputs[i].attr);
    }

    printf("\nStep2 inputs:\n");
    for (uint32_t i = 0; i < ctx->step2_io_num.n_input; i++) {
        dump_tensor_attr(ctx->step2_inputs[i].attr);
    }

    printf("\nStep2 outputs:\n");
    for (uint32_t i = 0; i < ctx->step2_io_num.n_output; i++) {
        dump_tensor_attr(ctx->step2_outputs[i].attr);
    }
    printf("\n");

    return 0;
}

// Release VITS model
int release_vits_model(vits_context_t* ctx) {
    if (ctx->initialized) {
        // Free Step1 tensors
        if (ctx->step1_inputs) {
            for (uint32_t i = 0; i < ctx->step1_io_num.n_input; i++) {
                if (ctx->step1_inputs[i].mem) {
                    rknn3_destroy_mem(ctx->ctx_step1, ctx->step1_inputs[i].mem);
                    ctx->step1_inputs[i].mem = nullptr;
                }
                if (ctx->step1_inputs[i].attr != NULL) {
                    free(ctx->step1_inputs[i].attr);
                    ctx->step1_inputs[i].attr = NULL;
                }
            }
            free(ctx->step1_inputs);
            ctx->step1_inputs = NULL;
        }

        if (ctx->step1_outputs) {
            for (uint32_t i = 0; i < ctx->step1_io_num.n_output; i++) {
                if (ctx->step1_outputs[i].mem) {
                    rknn3_destroy_mem(ctx->ctx_step1, ctx->step1_outputs[i].mem);
                    ctx->step1_outputs[i].mem = nullptr;
                }
                if (ctx->step1_outputs[i].attr != NULL) {
                    free(ctx->step1_outputs[i].attr);
                    ctx->step1_outputs[i].attr = NULL;
                }
            }
            free(ctx->step1_outputs);
            ctx->step1_outputs = NULL;
        }

        // Free Step2 tensors
        if (ctx->step2_inputs) {
            for (uint32_t i = 0; i < ctx->step2_io_num.n_input; i++) {
                if (ctx->step2_inputs[i].mem) {
                    rknn3_destroy_mem(ctx->ctx_step2, ctx->step2_inputs[i].mem);
                    ctx->step2_inputs[i].mem = nullptr;
                }
                if (ctx->step2_inputs[i].attr != NULL) {
                    free(ctx->step2_inputs[i].attr);
                    ctx->step2_inputs[i].attr = NULL;
                }
            }
            free(ctx->step2_inputs);
            ctx->step2_inputs = NULL;
        }

        if (ctx->step2_outputs) {
            for (uint32_t i = 0; i < ctx->step2_io_num.n_output; i++) {
                if (ctx->step2_outputs[i].mem) {
                    rknn3_destroy_mem(ctx->ctx_step2, ctx->step2_outputs[i].mem);
                    ctx->step2_outputs[i].mem = nullptr;
                }
                if (ctx->step2_outputs[i].attr != NULL) {
                    free(ctx->step2_outputs[i].attr);
                    ctx->step2_outputs[i].attr = NULL;
                }
            }
            free(ctx->step2_outputs);
            ctx->step2_outputs = NULL;
        }

        // Destroy RKNN contexts
        rknn3_destroy(ctx->ctx_step1);
        rknn3_destroy(ctx->ctx_step2);

        ctx->initialized = false;
        printf("VITS models released successfully\n");
    }
    return 0;
}

// Generate path for attention computation
std::vector<std::vector<std::vector<float>>> GeneratePath(
    const std::vector<std::vector<float>>& w_ceil,
    int b, int t_x, int t_y) {

    // Compute cumulative duration
    std::vector<std::vector<float>> cum_duration(b, std::vector<float>(t_x, 0.0f));
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < t_x; j++) {
            if (j == 0) {
                cum_duration[i][j] = w_ceil[i][j];
            } else {
                cum_duration[i][j] = cum_duration[i][j-1] + w_ceil[i][j];
            }
        }
    }

    // Find max duration
    float max_duration = 0.0f;
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < t_x; j++) {
            max_duration = std::max(max_duration, cum_duration[i][j]);
        }
    }
    int t_y_max = static_cast<int>(std::ceil(max_duration));

    // Generate path
    std::vector<std::vector<std::vector<float>>> path(b, std::vector<std::vector<float>>(t_x, std::vector<float>(t_y, 0.0f)));

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < t_x; j++) {
            for (int k = 0; k < t_y_max; k++) {
                if (k < cum_duration[i][j]) {
                    path[i][j][k] = 1.0f;
                }
            }
        }
    }

    // Pad and subtract
    std::vector<std::vector<std::vector<float>>> path_padded(b, std::vector<std::vector<float>>(t_x + 1, std::vector<float>(t_y, 0.0f)));
    for (int i = 0; i < b; i++) {
        for (int j = 1; j <= t_x; j++) {
            for (int k = 0; k < t_y; k++) {
                path_padded[i][j][k] = path[i][j-1][k];
            }
        }
    }

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < t_x; j++) {
            for (int k = 0; k < t_y; k++) {
                path[i][j][k] = path[i][j][k] - path_padded[i][j][k];
            }
        }
    }

    return path;
}

// Run VITS inference
int inference_vits(vits_context_t* ctx, TextProcessor* processor,
                  const std::string& text, std::vector<float>& audio_output) {
    if (!ctx->initialized) {
        fprintf(stderr, "VITS model not initialized\n");
        return -1;
    }

    // Process text
    std::vector<std::string> cleaners = {"english_cleaners2"};
    std::vector<int64_t> stn_tst = processor->TextToSequence(text, cleaners);
    stn_tst = processor->Intersperse(stn_tst, 0);

    printf("Text length: %zu tokens\n", stn_tst.size());

    auto start_time = std::chrono::steady_clock::now();

    // STEP 1: Text Encoder with Sliding Window
    int total_text_len = stn_tst.size();
    int num_text_windows = (total_text_len + ctx->text_window_size - 1) / ctx->text_window_size;

    // Random noise for SDP
    std::vector<float> sdp_rand(2 * ctx->text_max_len);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 2 * ctx->text_max_len; i++) {
        sdp_rand[i] = dist(gen);
    }

    printf("Running Step1 (Text Encoder) with %d windows...\n", num_text_windows);

    // Dynamic tensor allocation based on actual model I/O counts
    int num_step1_inputs = ctx->step1_io_num.n_input;
    int num_step1_outputs = ctx->step1_io_num.n_output;

    // Prepare input data
    std::vector<float> w_full;
    std::vector<std::vector<float>> m_p_full;
    std::vector<std::vector<float>> logs_p_full;
    std::vector<float> x_mask_full;
    std::vector<float> speaker_embedding;  // For VCTK models

    // Reserve capacity to avoid reallocations
    w_full.reserve(total_text_len);
    m_p_full.reserve(total_text_len);
    logs_p_full.reserve(total_text_len);
    x_mask_full.reserve(total_text_len);

    // Print detailed tensor information for debugging
    for (int i = 0; i < num_text_windows; i++) {
        int start_idx = i * ctx->text_window_size;
        int end_idx = std::min(start_idx + ctx->text_window_size, total_text_len);

        int context_start = std::max(0, start_idx - ctx->text_context_size);
        int context_end = std::min(total_text_len, end_idx + ctx->text_context_size);

        // Build extended text window
        std::vector<int64_t> text_window_extended;
        for (int j = context_start; j < context_end; j++) {
            text_window_extended.push_back(stn_tst[j]);
        }

        // Pad if necessary
        while (text_window_extended.size() < static_cast<size_t>(ctx->text_max_len)) {
            text_window_extended.push_back(0);
        }

        // Prepare input data
        // Input 0: x (text) - INT32
        int32_t* x_ptr = (int32_t*)ctx->step1_inputs[0].mem->virt_addr;
        for (int j = 0; j < ctx->text_max_len; j++) {
            x_ptr[j] = static_cast<int32_t>(text_window_extended[j]);
        }

        // Input 1: x_mask - FP16 (need to convert from FP32)
        float16* x_mask_ptr = (float16*)ctx->step1_inputs[1].mem->virt_addr;
        for (int j = 0; j < ctx->text_max_len; j++) {
            x_mask_ptr[j] = fp32_to_fp16(1.0f);
        }

        // Input 2: length_scale - FP16
        float16* length_scale_ptr = (float16*)ctx->step1_inputs[2].mem->virt_addr;
        length_scale_ptr[0] = fp32_to_fp16(ctx->length_scale);

        // Input 3: noise_scale_w - FP16
        float16* noise_scale_w_ptr = (float16*)ctx->step1_inputs[3].mem->virt_addr;
        noise_scale_w_ptr[0] = fp32_to_fp16(DEFAULT_NOISE_SCALE_W);

        // Input 4: sdp_rand - FP16 (convert from FP32)
        float16* sdp_rand_ptr = (float16*)ctx->step1_inputs[4].mem->virt_addr;
        for (int j = 0; j < 2 * ctx->text_max_len; j++) {
            sdp_rand_ptr[j] = fp32_to_fp16(sdp_rand[j]);
        }

        // Input 5: speaker_id - INT64 (only for VCTK models)
        if (ctx->is_multi_speaker && num_step1_inputs >= 6) {
            int64_t* speaker_id_ptr = (int64_t*)ctx->step1_inputs[5].mem->virt_addr;
            speaker_id_ptr[0] = static_cast<int64_t>(ctx->speaker_id);
        }

        // Sync inputs to device
        for (int sync_i = 0; sync_i < num_step1_inputs; sync_i++) {
            int ret_sync = rknn3_mem_sync(ctx->ctx_step1, ctx->step1_inputs[sync_i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret_sync < 0) {
                fprintf(stderr, "rknn3_mem_sync step1 input[%d] failed! ret=%d\n", sync_i, ret_sync);
                return ret_sync;
            }
        }

        // Run inference
        int ret = rknn3_run(ctx->ctx_step1, ctx->step1_inputs, num_step1_inputs, ctx->step1_outputs, num_step1_outputs);
        if (ret < 0) {
            fprintf(stderr, "rknn3_run step1 window %d failed: %d\n", i, ret);
            continue;
        }

        // Sync outputs from device
        for (int sync_i = 0; sync_i < num_step1_outputs; sync_i++) {
            int ret_sync = rknn3_mem_sync(ctx->ctx_step1, ctx->step1_outputs[sync_i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret_sync < 0) {
                fprintf(stderr, "rknn3_mem_sync step1 output[%d] failed! ret=%d\n", sync_i, ret_sync);
                return ret_sync;
            }
        }

        // Extract outputs from Step1
        // Output shapes: w[1,1,160], m_p[1,192,160], logs_p[1,192,160], x_mask_out[1,1,160]
        int actual_context_before = start_idx - context_start;
        int core_start = actual_context_before;
        int core_len = end_idx - start_idx;
        int core_end = std::min(core_start + core_len, ctx->text_max_len);

        // Extract w (duration) - FP16 format
        float16* w_data_fp16 = (float16*)ctx->step1_outputs[0].mem->virt_addr;
        for (int t = core_start; t < core_end; t++) {
            w_full.push_back(fp16_to_fp32(w_data_fp16[t]));
        }

        // Extract m_p (mean) - FP16 format
        float16* m_p_data_fp16 = (float16*)ctx->step1_outputs[1].mem->virt_addr;
        // Get dimensions from tensor attributes: m_p shape is [1, channels, time]
        int m_p_channels = ctx->step1_outputs[1].attr->shape[1];  // Usually 192
        int m_p_time_dim = ctx->step1_outputs[1].attr->shape[2];    // Usually 160

        for (int t = core_start; t < core_end; t++) {
            std::vector<float> m_p_timestep(m_p_channels);
            for (int j = 0; j < m_p_channels; j++) {
                m_p_timestep[j] = fp16_to_fp32(m_p_data_fp16[j * m_p_time_dim + t]);
            }
            m_p_full.push_back(m_p_timestep);
        }

        // Extract logs_p (log standard deviation) - FP16 format
        float16* logs_p_data_fp16 = (float16*)ctx->step1_outputs[2].mem->virt_addr;
        // Same dimensions as m_p
        int logs_p_channels = ctx->step1_outputs[2].attr->shape[1];
        int logs_p_time_dim = ctx->step1_outputs[2].attr->shape[2];

        for (int t = core_start; t < core_end; t++) {
            std::vector<float> logs_p_timestep(logs_p_channels);
            for (int j = 0; j < logs_p_channels; j++) {
                logs_p_timestep[j] = fp16_to_fp32(logs_p_data_fp16[j * logs_p_time_dim + t]);
            }
            logs_p_full.push_back(logs_p_timestep);
        }

        // Extract x_mask_out - FP16 format
        float16* x_mask_out_fp16 = (float16*)ctx->step1_outputs[3].mem->virt_addr;
        for (int t = core_start; t < core_end; t++) {
            x_mask_full.push_back(fp16_to_fp32(x_mask_out_fp16[t]));
        }

        // Extract speaker embedding for VCTK models (output index 4)
        if (ctx->is_multi_speaker && num_step1_outputs >= 5) {
            if (speaker_embedding.empty()) {  // Only extract once from first window
                float16* g_data_fp16 = (float16*)ctx->step1_outputs[4].mem->virt_addr;
                int g_size = ctx->step1_outputs[4].attr->n_elems;
                for (int j = 0; j < g_size; j++) {
                    speaker_embedding.push_back(fp16_to_fp32(g_data_fp16[j]));
                }
                printf("Extracted speaker embedding: %d dimensions\n", g_size);
            }
        }
    }

    printf("Step1 completed\n");

    // Compute audio length from w_full
    std::vector<float> w_ceil;
    for (size_t i = 0; i < w_full.size(); i++) {
        w_ceil.push_back(std::ceil(w_full[i]));
    }

    int y_lengths = std::max(1, static_cast<int>(std::accumulate(w_ceil.begin(), w_ceil.end(), 0.0f)));

    // Debug: print duration statistics
    float avg_duration = w_full.empty() ? 0.0f : std::accumulate(w_full.begin(), w_full.end(), 0.0f) / w_full.size();
    printf("Duration stats: tokens=%zu, avg=%.2f, total_frames=%d (%.2f seconds)\n",
           w_full.size(), avg_duration, y_lengths, y_lengths / (float)ctx->sampling_rate * ctx->hop_length);
    printf("Current length_scale: %.2f, noise_scale_w: %.2f\n", ctx->length_scale, DEFAULT_NOISE_SCALE_W);

    // Get channels dimension from Step1 output[1] (m_p tensor)
    int m_p_channels = ctx->step1_outputs[1].attr->shape[1];  // Usually 192

    // Generate attention path and compute z_p
    // Create 2D w_ceil for GeneratePath function
    std::vector<std::vector<float>> w_ceil_2d(1, std::vector<float>(w_full.size()));
    for (size_t i = 0; i < w_full.size(); i++) {
        w_ceil_2d[0][i] = w_ceil[i];
    }

    std::vector<std::vector<std::vector<float>>> attn_path = GeneratePath(w_ceil_2d, 1, w_full.size(), y_lengths);

    // Compute z_p using attention
    std::vector<std::vector<float>> z_p(m_p_channels, std::vector<float>(y_lengths, 0.0f));

    // Add noise for randomness (reuse existing random number generators)
    for (int i = 0; i < m_p_channels; i++) {
        for (int j = 0; j < y_lengths; j++) {
            float noise_val = dist(gen);
            float m_p_weighted = 0.0f;
            float logs_p_weighted = 0.0f;

            for (size_t t_x = 0; t_x < w_full.size(); t_x++) {
                float attn_weight = attn_path[0][t_x][j];
                m_p_weighted += m_p_full[t_x][i] * attn_weight;
                logs_p_weighted += logs_p_full[t_x][i] * attn_weight;
            }

            z_p[i][j] = m_p_weighted + noise_val * std::exp(logs_p_weighted) * ctx->noise_scale;
        }
    }

    // STEP 2: Audio Generation with Sliding Window
    printf("Running Step2 (Audio Decoder)...\n");
    int num_audio_windows = (y_lengths + ctx->audio_window_size - 1) / ctx->audio_window_size;

    // Dynamic tensor allocation for Step2 based on actual model I/O counts
    int num_step2_inputs = ctx->step2_io_num.n_input;
    int num_step2_outputs = ctx->step2_io_num.n_output;

    for (int i = 0; i < num_audio_windows; i++) {
        int start_idx = i * ctx->audio_window_size;
        int end_idx = std::min(start_idx + ctx->audio_window_size, y_lengths);

        // Prepare z_p window input - FP16
        float16* z_p_ptr = (float16*)ctx->step2_inputs[0].mem->virt_addr;
        // Get z_p dimensions from tensor attributes: [1, channels, time]
        int z_p_channels = ctx->step2_inputs[0].attr->shape[1];  // Usually 192
        int z_p_window_size = ctx->step2_inputs[0].attr->shape[2]; // Usually 128

        for (int t = 0; t < z_p_window_size; t++) {
            int actual_t = start_idx + t;
            if (actual_t < y_lengths) {
                for (int j = 0; j < z_p_channels; j++) {
                    // Input shape: [1, channels, window_size]
                    z_p_ptr[j * z_p_window_size + t] = fp32_to_fp16(z_p[j][actual_t]);
                }
            } else {
                // Padding
                for (int j = 0; j < z_p_channels; j++) {
                    z_p_ptr[j * z_p_window_size + t] = fp32_to_fp16(0.0f);
                }
            }
        }

        // Prepare inputs based on model type
        if (ctx->is_multi_speaker && num_step2_inputs >= 3) {
            // VCTK: [z_p, g, y_mask]
            // Input 1: speaker_embedding - FP16
            float16* g_ptr = (float16*)ctx->step2_inputs[1].mem->virt_addr;
            for (size_t j = 0; j < speaker_embedding.size(); j++) {
                g_ptr[j] = fp32_to_fp16(speaker_embedding[j]);
            }

            // Input 2: y_mask - FP16
            float16* y_mask_ptr = (float16*)ctx->step2_inputs[2].mem->virt_addr;
            for (int t = 0; t < ctx->audio_window_size; t++) {
                int actual_t = start_idx + t;
                y_mask_ptr[t] = fp32_to_fp16((actual_t < y_lengths) ? 1.0f : 0.0f);
            }
        } else {
            // LJS: [z_p, y_mask]
            // Input 1: y_mask - FP16
            float16* y_mask_ptr = (float16*)ctx->step2_inputs[1].mem->virt_addr;
            for (int t = 0; t < ctx->audio_window_size; t++) {
                int actual_t = start_idx + t;
                y_mask_ptr[t] = fp32_to_fp16((actual_t < y_lengths) ? 1.0f : 0.0f);
            }
        }

        // Run Step2 inference
        // Sync inputs to device
        for (int sync_i = 0; sync_i < num_step2_inputs; sync_i++) {
            int ret_sync = rknn3_mem_sync(ctx->ctx_step2, ctx->step2_inputs[sync_i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret_sync < 0) {
                fprintf(stderr, "rknn3_mem_sync step2 input[%d] failed! ret=%d\n", sync_i, ret_sync);
                return ret_sync;
            }
        }

        int ret = rknn3_run(ctx->ctx_step2, ctx->step2_inputs, num_step2_inputs, ctx->step2_outputs, num_step2_outputs);
        if (ret < 0) {
            fprintf(stderr, "rknn3_run step2 window %d failed: %d\n", i, ret);
            continue;
        }

        // Sync output from device
        for (int sync_i = 0; sync_i < num_step2_outputs; sync_i++) {
            int ret_sync = rknn3_mem_sync(ctx->ctx_step2, ctx->step2_outputs[sync_i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret_sync < 0) {
                fprintf(stderr, "rknn3_mem_sync step2 output[%d] failed! ret=%d\n", sync_i, ret_sync);
                return ret_sync;
            }
        }

        // Extract audio output - FP16
        float16* audio_data_fp16 = (float16*)ctx->step2_outputs[0].mem->virt_addr;
        int valid_frames = std::min(ctx->audio_window_size, y_lengths - start_idx);
        int valid_samples = valid_frames * ctx->hop_length;

        // Calculate total output samples from tensor shape
        int total_output_samples = 1;
        for (int j = 0; j < ctx->step2_outputs[0].attr->n_dims; j++) {
            total_output_samples *= ctx->step2_outputs[0].attr->shape[j];
        }

        // Extract all valid audio samples from this window
        int samples_to_extract = std::min(valid_samples, total_output_samples);
        for (int j = 0; j < samples_to_extract; j++) {
            audio_output.push_back(fp16_to_fp32(audio_data_fp16[j]));
        }
    }

    printf("Step2 completed\n");

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Calculate audio duration and RTF
    float audio_duration = (float)audio_output.size() / ctx->sampling_rate;
    float inference_time = duration.count() / 1000.0f;
    float rtf = (audio_duration > 0.0f) ? (inference_time / audio_duration) : 0.0f;

    printf("========================================\n");
    printf("Performance Summary\n");
    printf("========================================\n");
    printf("Audio: %zu samples, %.2f seconds\n", audio_output.size(), audio_duration);
    printf("Total inference: %.2f seconds\n", inference_time);
    printf("RTF: %.3f\n", rtf);
    printf("========================================\n");

    return 0;
}

// Save audio as WAV file
int save_wav(const char* filename, const std::vector<float>& audio, int sample_rate) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open output file: %s\n", filename);
        return -1;
    }

    // WAV header
    int num_samples = audio.size();
    int byte_rate = sample_rate * 2;  // 16-bit mono
    int data_size = num_samples * 2;
    int file_size = 36 + data_size;

    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);

    int fmt_chunk_size = 16;
    int16_t audio_format = 1;  // PCM
    int16_t num_channels = 1;
    int16_t bits_per_sample = 16;
    int block_align = num_channels * bits_per_sample / 8;

    fwrite(&fmt_chunk_size, 4, 1, fp);
    fwrite(&audio_format, 2, 1, fp);
    fwrite(&num_channels, 2, 1, fp);
    fwrite(&sample_rate, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bits_per_sample, 2, 1, fp);

    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);

    // Write audio data
    for (size_t i = 0; i < audio.size(); i++) {
        // Clamp and convert to 16-bit
        float sample = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_t sample_16bit = static_cast<int16_t>(sample * 32767.0f);
        fwrite(&sample_16bit, 2, 1, fp);
    }

    fclose(fp);
    printf("Saved audio to %s\n", filename);
    return 0;
}
