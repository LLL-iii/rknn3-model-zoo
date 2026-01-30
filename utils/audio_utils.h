#ifndef _RKNN_MODEL_ZOO_AUDIO_UTILS_H_
#define _RKNN_MODEL_ZOO_AUDIO_UTILS_H_

#include "common.h"
#include <vector>

/**
 * @brief Reads an audio file into a buffer.
 *
 * @param path [in] Path to the audio file.
 * @param audio [out] Pointer to the audio buffer structure that will store the read data.
 * @return int 0 on success, -1 on error.
 */
int read_audio(const char *path, audio_buffer_t *audio);

int read_mel_filters(const char *fileName, float *data, int max_lines);

void audio_preprocess(audio_buffer_t *audio, float *mel_filters, int n_fft, int hop_length, int n_mels, int max_audio_length, std::vector<float> &x_mel, int *actual_len);


/**
 * @brief Resamples audio data to a desired sample rate.
 *
 * This function adjusts the sample rate of the provided audio data from 
 * the original sample rate to the desired sample rate. The audio data 
 * is assumed to be in a format compatible with the processing logic.
 *
 * @param audio [in/out] Pointer to the audio buffer structure containing 
 *                       the audio data to be resampled.
 * @param original_sample_rate [in] The original sample rate of the audio data.
 * @param desired_sample_rate [in] The target sample rate to resample the audio data to.
 * @return int 0 on success, -1 on error.
 */
int resample_audio(audio_buffer_t *audio, int original_sample_rate, int desired_sample_rate);

/**
 * @brief Converts audio data to a single channel (mono).
 *
 * This function takes two-channel audio data and converts it to single 
 * channel (mono) by averaging the channels or using another merging strategy.
 * The audio data will be modified in place.
 *
 * @param audio [in/out] Pointer to the audio buffer structure containing 
 *                       the audio data to be converted.
 * @return int 0 on success, -1 on error.
 */
int convert_channels(audio_buffer_t *audio);

#endif // _RKNN_MODEL_ZOO_AUDIO_UTILS_H_
