#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "speech_decoder.h"
#include "talker.h"

namespace {

const int kFeatSize = 16;
const int kChunkSize = 25;
const int kLeftContextSize = 25;
const int kWindowLen = kChunkSize + kLeftContextSize;
const int kTotalUpsample = 1920;
const int kSampleRate = 24000;

void write_wav(const std::string& filename, const std::vector<float>& audio, int sample_rate) {
    int num_channels = 1;
    int bits_per_sample = 32;
    int byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    int block_align = num_channels * (bits_per_sample / 8);
    int data_size = static_cast<int>(audio.size() * sizeof(float));
    int chunk_size = 36 + data_size;

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        printf("failed to write wav: %s\n", filename.c_str());
        return;
    }

    ofs.write("RIFF", 4);
    ofs.write(reinterpret_cast<const char*>(&chunk_size), 4);
    ofs.write("WAVE", 4);

    ofs.write("fmt ", 4);
    int subchunk1_size = 16;
    ofs.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    int audio_format = 3;
    ofs.write(reinterpret_cast<const char*>(&audio_format), 2);
    ofs.write(reinterpret_cast<const char*>(&num_channels), 2);
    ofs.write(reinterpret_cast<const char*>(&sample_rate), 4);
    ofs.write(reinterpret_cast<const char*>(&byte_rate), 4);
    ofs.write(reinterpret_cast<const char*>(&block_align), 2);
    ofs.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    ofs.write("data", 4);
    ofs.write(reinterpret_cast<const char*>(&data_size), 4);
    ofs.write(reinterpret_cast<const char*>(audio.data()), data_size);
}

std::vector<int32_t> transpose_lx16_to_16xl(const std::vector<int32_t>& src_lx16, int length) {
    std::vector<int32_t> dst_16xl(kFeatSize * length, 0);
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < kFeatSize; ++j) {
            dst_16xl[j * length + i] = src_lx16[i * kFeatSize + j];
        }
    }
    return dst_16xl;
}

std::string join_text_from_argv(int argc, char** argv, int start_index) {
    std::ostringstream oss;
    for (int i = start_index; i < argc; ++i) {
        if (i > start_index) {
            oss << " ";
        }
        oss << argv[i];
    }
    return oss.str();
}

std::string build_output_path(const std::string& output_dir) {
    if (output_dir.empty()) {
        return "output.wav";
    }
    if (output_dir.back() == '/') {
        return output_dir + "output.wav";
    }
    return output_dir + "/output.wav";
}

struct DemoState {
    Qwen3TTSSpeechDecoder decoder;
    std::vector<int32_t> pending_codes_lx16;
    std::vector<int32_t> prev_context_lx16;
    std::vector<float> audio_buffer;
    std::thread decoder_thread;
    std::mutex mutex;
    std::condition_variable token_cond;
    std::condition_variable cond;
    std::string output_path = "output.wav";
    // 外部要求退出 demo 时置位，用于安全结束 decoder 线程
    bool stop_requested = false;
    // talker 已经结束，不会再有新的 token 回调
    bool talker_done = false;
    bool finished = false;
    bool failed = false;
};

bool decode_one_window(
    DemoState* state,
    const std::vector<int32_t>& window_frames_lx16,
    int context_size,
    int take_new) {
    std::vector<int32_t> window_frames_16xl = transpose_lx16_to_16xl(window_frames_lx16, kWindowLen);
    std::vector<float> audio_values;
    if (state->decoder.Decode(window_frames_16xl, &audio_values) != 0) {
        printf("speech decoder failed\n");
        return false;
    }

    size_t start = std::min(static_cast<size_t>(context_size * kTotalUpsample), audio_values.size());
    size_t end = std::min(start + static_cast<size_t>(take_new * kTotalUpsample), audio_values.size());
    if (start >= end) {
        printf("invalid decoder crop range\n");
        return false;
    }

    state->audio_buffer.insert(state->audio_buffer.end(), audio_values.begin() + start, audio_values.begin() + end);
    write_wav(state->output_path, state->audio_buffer, kSampleRate);
    printf("decoded chunk samples=%zu, total_samples=%zu\n", end - start, state->audio_buffer.size());
    return true;
}

void decoder_thread_entry(DemoState* state) {
    while (true) {
        std::vector<int32_t> window_frames_lx16(kWindowLen * kFeatSize, 0);
        int context_size = 0;
        int take_new = 0;

        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->token_cond.wait(lock, [state]() {
                int pending_frames = static_cast<int>(state->pending_codes_lx16.size() / kFeatSize);
                bool is_first = state->prev_context_lx16.empty();
                int need_new = is_first ? kWindowLen : kChunkSize;

                if (state->stop_requested) {
                    return true;
                }

                if (state->talker_done) {
                    // talker 已结束后，不再强求凑满一个标准窗口；
                    // 只要有残留 token，就允许进入 drain 流程。
                    return true;
                }

                // 正常生成阶段：只有攒够一窗数据才触发 decoder
                return pending_frames >= need_new;
            });

            int pending_frames = static_cast<int>(state->pending_codes_lx16.size() / kFeatSize);
            if (state->stop_requested && pending_frames <= 0) {
                break;
            }

            if (state->talker_done && pending_frames <= 0) {
                state->prev_context_lx16.clear();
                state->finished = true;
                state->cond.notify_one();
                break;
            }

            bool is_first = state->prev_context_lx16.empty();
            context_size = is_first ? 0 : kLeftContextSize;
            int need_new = is_first ? kWindowLen : kChunkSize;

            if (pending_frames >= need_new) {
                take_new = need_new;
            } else if (state->talker_done && pending_frames > 0) {
                // talker 已结束，但还有尾部残留 token，直接一次性解完
                take_new = pending_frames;
            } else if (state->stop_requested && pending_frames > 0) {
                take_new = pending_frames;
            } else {
                continue;
            }

            if (context_size > 0) {
                std::memcpy(
                    window_frames_lx16.data(),
                    state->prev_context_lx16.data(),
                    sizeof(int32_t) * context_size * kFeatSize);
            }

            std::memcpy(
                window_frames_lx16.data() + context_size * kFeatSize,
                state->pending_codes_lx16.data(),
                sizeof(int32_t) * take_new * kFeatSize);

            // prev_context 取“本次有效帧尾部”的 left_context，而不是固定取 50 窗口尾部，
            // 这样在 drain 场景下不会把 zero padding 误带到下一窗。
            int valid_len = context_size + take_new;
            if (valid_len >= kLeftContextSize) {
                int ctx_start = (valid_len - kLeftContextSize) * kFeatSize;
                state->prev_context_lx16.resize(kLeftContextSize * kFeatSize);
                std::memcpy(
                    state->prev_context_lx16.data(),
                    window_frames_lx16.data() + ctx_start,
                    sizeof(int32_t) * kLeftContextSize * kFeatSize);
            } else {
                state->prev_context_lx16.clear();
            }

            state->pending_codes_lx16.erase(
                state->pending_codes_lx16.begin(),
                state->pending_codes_lx16.begin() + take_new * kFeatSize);
        }

        if (!decode_one_window(state, window_frames_lx16, context_size, take_new)) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->failed = true;
            state->finished = true;
            state->cond.notify_one();
            break;
        }
    }
}

void talker_callback(std::vector<int>* tokens, bool talker_is_generating, void* userdata) {
    DemoState* state = reinterpret_cast<DemoState*>(userdata);
    std::unique_ptr<std::vector<int> > owned_tokens(tokens);

    std::lock_guard<std::mutex> lock(state->mutex);

    if (tokens != NULL && !tokens->empty()) {
        if (tokens->size() != kFeatSize) {
            printf("unexpected token group size: %zu\n", tokens->size());
            state->failed = true;
            state->finished = true;
            state->cond.notify_one();
            return;
        }

        for (size_t i = 0; i < tokens->size(); ++i) {
            state->pending_codes_lx16.push_back((*tokens)[i]);
        }
    } else if (!talker_is_generating) {
        // 收到空帧且 talker 已结束，表示不会再有新的 token 了，
        // decoder 线程后续只需要把剩余 token drain 完即可。
        state->talker_done = true;
    }

    // 唤醒 decoder 线程去检查是否已经满足解码条件
    state->token_cond.notify_one();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <model_dir> <ref_speaker> <output_dir> <text...>\n", argv[0]);
        return -1;
    }

    const std::string model_dir = argv[1];
    const std::string ref_speaker = argv[2];
    const std::string output_dir = argv[3];
    const std::string text = join_text_from_argv(argc, argv, 4);
    if (text.empty()) {
        printf("input text is empty\n");
        return -1;
    }

    DemoState state;
    state.output_path = build_output_path(output_dir);
    Qwen3TTSSpeechDecoder::Config decoder_config;
    decoder_config.model_dir = model_dir;
    if (state.decoder.Init(decoder_config) != 0) {
        printf("failed to init speech decoder\n");
        return -1;
    }

    state.decoder_thread = std::thread(decoder_thread_entry, &state);

    Qwen3TTSTalker talker;
    Qwen3TTSTalkerConfig talker_config;
    talker_config.model_dir = model_dir;
    talker_config.callback = talker_callback;
    talker_config.userdata = &state;
    if (talker.Init(talker_config) != 0) {
        printf("failed to init talker\n");
        return -1;
    }

    Qwen3TTSTalkerRequest request;
    request.text = text;
    request.instruct = "";
    request.language = "auto";
    request.ref_speaker = ref_speaker;
    request.non_streaming_mode = 0;

    if (talker.Process(request) != 0) {
        printf("failed to start talker\n");
        return -1;
    }

    {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.cond.wait(lock, [&state]() { return state.finished; });
    }

    talker.Release();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.stop_requested = true;
        state.token_cond.notify_one();
    }
    if (state.decoder_thread.joinable()) {
        state.decoder_thread.join();
    }
    state.decoder.Release();

    if (state.failed) {
        printf("demo finished with errors\n");
        return -1;
    }

    printf("saved wav to %s\n", state.output_path.c_str());
    return 0;
}
