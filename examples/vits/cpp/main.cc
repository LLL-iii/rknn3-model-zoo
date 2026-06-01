#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vits.h"
#include "text_processor.h"
#include <sys/stat.h>
#include <unistd.h>

#define DEFAULT_STEP1_MODEL "./model/vits_ljs_step1_slid_fp.rknn"
#define DEFAULT_STEP1_WEIGHT "./model/vits_ljs_step1_slid_fp.weight"
#define DEFAULT_STEP2_MODEL "./model/vits_ljs_step2_slid_fp.rknn"
#define DEFAULT_STEP2_WEIGHT "./model/vits_ljs_step2_slid_fp.weight"
#define DEFAULT_OUTPUT_LJS "output_audio_ljs_rknn.wav"
#define DEFAULT_OUTPUT_VCTK "output_audio_vctk_rknn.wav"
#define DEFAULT_TEXT "VITS is awesome! This is a text-to-speech system."
#define DEFAULT_SPEAKER_ID 0

void print_usage(const char* prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  --step1_model <path>   Path to Step1 RKNN model (default: %s)\n", DEFAULT_STEP1_MODEL);
    printf("  --step1_weight <path>  Path to Step1 RKNN weight (default: %s)\n", DEFAULT_STEP1_WEIGHT);
    printf("  --step2_model <path>   Path to Step2 RKNN model (default: %s)\n", DEFAULT_STEP2_MODEL);
    printf("  --step2_weight <path>  Path to Step2 RKNN weight (default: %s)\n", DEFAULT_STEP2_WEIGHT);
    printf("  --text <text>          Text to synthesize (default: \"%s\")\n", DEFAULT_TEXT);
    printf("  --output <path>        Output WAV file path (default: auto-detected based on model type)\n");
    printf("  --speaker_id <id>      Speaker ID for VCTK models (default: %d)\n", DEFAULT_SPEAKER_ID);
    printf("  --core_mask <hex>      NPU core mask (default: 0x1)\n");
    printf("  --help                 Show this help message\n");
}

// Helper function to check if file exists
bool file_exists(const char* path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

int main(int argc, char** argv) {
    const char* step1_model = DEFAULT_STEP1_MODEL;
    const char* step1_weight = DEFAULT_STEP1_WEIGHT;
    const char* step2_model = DEFAULT_STEP2_MODEL;
    const char* step2_weight = DEFAULT_STEP2_WEIGHT;
    const char* text = DEFAULT_TEXT;
    const char* output_path = NULL;  // Auto-detect based on model type
    uint32_t core_mask = 0x1;
    int speaker_id = DEFAULT_SPEAKER_ID;
    bool user_specified_output = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--step1_model") == 0 && i + 1 < argc) {
            step1_model = argv[++i];
        } else if (strcmp(argv[i], "--step1_weight") == 0 && i + 1 < argc) {
            step1_weight = argv[++i];
        } else if (strcmp(argv[i], "--step2_model") == 0 && i + 1 < argc) {
            step2_model = argv[++i];
        } else if (strcmp(argv[i], "--step2_weight") == 0 && i + 1 < argc) {
            step2_weight = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
            user_specified_output = true;
        } else if (strcmp(argv[i], "--speaker_id") == 0 && i + 1 < argc) {
            speaker_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--core_mask") == 0 && i + 1 < argc) {
            core_mask = strtoul(argv[++i], NULL, 16);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate speaker ID
    if (speaker_id < 0 || speaker_id > 108) {
        fprintf(stderr, "Invalid speaker_id: %d (must be 0-108)\n", speaker_id);
        return 1;
    }

    // Validate core mask
    if (core_mask == 0) {
        fprintf(stderr, "Invalid core_mask: 0x%x (must be non-zero)\n", core_mask);
        return 1;
    }

    // Validate text input
    if (text == NULL || strlen(text) == 0) {
        fprintf(stderr, "Invalid text: empty or NULL\n");
        return 1;
    }

    // Validate model files exist
    if (!file_exists(step1_model)) {
        fprintf(stderr, "Step1 model file not found: %s\n", step1_model);
        return 1;
    }
    if (!file_exists(step1_weight)) {
        fprintf(stderr, "Step1 weight file not found: %s\n", step1_weight);
        return 1;
    }
    if (!file_exists(step2_model)) {
        fprintf(stderr, "Step2 model file not found: %s\n", step2_model);
        return 1;
    }
    if (!file_exists(step2_weight)) {
        fprintf(stderr, "Step2 weight file not found: %s\n", step2_weight);
        return 1;
    }

    // Auto-detect output path if not specified
    const char* default_output = DEFAULT_OUTPUT_LJS;  // Default to LJS
    if (output_path == NULL) {
        // Check if VCTK model is being used based on step1_model path
        if (step1_model != NULL && strstr(step1_model, "vctk") != NULL) {
            default_output = DEFAULT_OUTPUT_VCTK;
        }
        output_path = default_output;
    }

    printf("========================================\n");
    printf("VITS LJSpeech RKNN3 Sliding Window Inference\n");
    printf("========================================\n");
    printf("Text:    %s\n", text);
    printf("Output:  %s\n", output_path);
    printf("Step1:   %s\n", step1_model);
    printf("Step1W:  %s\n", step1_weight);
    printf("Step2:   %s\n", step2_model);
    printf("Step2W:  %s\n", step2_weight);
    printf("Core:    0x%x\n", core_mask);
    printf("Speaker: %d\n", speaker_id);
    printf("========================================\n\n");

    // Initialize text processor
    printf("Initializing text processor...\n");
    TextProcessor processor;
    if (!processor.Init("./model/espeak-ng-data")) {
        fprintf(stderr, "Failed to initialize text processor\n");
        return -1;
    }
    printf("Text processor initialized successfully\n\n");

    // Initialize VITS models
    printf("Initializing VITS models...\n");
    vits_context_t vits_ctx;
    int ret = init_vits_model(step1_model, step1_weight, step2_model, step2_weight, &vits_ctx, core_mask, speaker_id);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize VITS models\n");
        return -1;
    }
    printf("VITS models initialized successfully\n\n");

    // Print model type
    printf("Model type: %s\n", vits_ctx.is_multi_speaker ? "VCTK (multi-speaker)" : "LJSpeech (single-speaker)");
    if (vits_ctx.is_multi_speaker) {
        printf("Using speaker ID: %d\n", vits_ctx.speaker_id);

        // Update output path for VCTK if user didn't specify
        if (!user_specified_output) {
            output_path = DEFAULT_OUTPUT_VCTK;
            printf("Output path updated to: %s\n", output_path);
        }
    }
    printf("\n");

    // Run inference
    printf("Running inference...\n");
    std::vector<float> audio_output;
    ret = inference_vits(&vits_ctx, &processor, text, audio_output);
    if (ret != 0) {
        fprintf(stderr, "Inference failed\n");
        release_vits_model(&vits_ctx);
        return -1;
    }

    // Save audio
    if (!audio_output.empty()) {
        printf("\nSaving audio...\n");
        ret = save_wav(output_path, audio_output, vits_ctx.sampling_rate);
        if (ret != 0) {
            fprintf(stderr, "Failed to save audio\n");
        }
    } else {
        printf("\nWarning: No audio output generated\n");
    }

    // Release models
    printf("\nReleasing models...\n");
    release_vits_model(&vits_ctx);

    printf("\n========================================\n");
    printf("Inference completed successfully!\n");
    printf("========================================\n");

    return 0;
}
