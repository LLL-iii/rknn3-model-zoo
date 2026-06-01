"""
VITS RKNN sliding window inference - Unified version

Supports both LJSpeech (single-speaker) and VCTK (multi-speaker) models.
- Text encoder: window_size=160 (128 core + 16*2 context)
- Audio decoder: window_size=128
- Automatically detects model type from file names
"""

import os
import sys
import argparse
import time
import numpy as np
from rknn.api import RKNN
from scipy.io.wavfile import write

# Default configuration
DEFAULT_DEVICE_ID = ""
DEFAULT_RKNN_DIR = "../model"

# Model configurations
LJS_CONFIG = {
    'step1': "vits_ljs_step1_slid_fp.rknn",
    'step2': "vits_ljs_step2_slid_fp.rknn",
    'multi_speaker': False,
    'num_speakers': 1
}

VCTK_CONFIG = {
    'step1': "vits_vctk_step1_slid_fp.rknn",
    'step2': "vits_vctk_step2_slid_fp.rknn",
    'multi_speaker': True,
    'num_speakers': 109
}


def intersperse(lst, value):
    """Intersperse value between elements of lst"""
    result = [value]
    for x in lst:
        result.append(x)
        result.append(value)
    return result


def generate_path_numpy(duration, mask):
    """NumPy implementation of generate_path for attention computation"""
    b, _, t_y, t_x = mask.shape
    _, _, t_x_from_duration = duration.shape
    t_x = t_x_from_duration

    cum_duration = np.cumsum(duration, axis=-1)
    cum_duration_flat = cum_duration.reshape(b * t_x)
    max_duration = int(cum_duration_flat.max()) if cum_duration_flat.size > 0 else t_y

    indices_all = np.cumsum(np.ones(max_duration, dtype=np.int32)) - 1
    path_flat = (indices_all.reshape(1, -1) < cum_duration_flat.reshape(-1, 1)).astype(np.float32)

    if max_duration < t_y:
        pad_cols = t_y - max_duration
        path_flat = np.pad(path_flat, ((0, 0), (0, pad_cols)), mode='constant', constant_values=0)
    else:
        path_flat = path_flat[:, :t_y]

    path = path_flat.reshape(b, t_x, t_y)
    path_padded = np.pad(path, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    path = path - path_padded[:, :-1, :]
    path = path.reshape(b, 1, t_x, t_y).transpose(0, 1, 3, 2) * mask

    return path


def detect_model_type(step1_path):
    """Detect if model is LJSpeech or VCTK based on file name"""
    if 'vctk' in step1_path.lower():
        return 'vctk', VCTK_CONFIG
    elif 'ljs' in step1_path.lower() or 'ljspeech' in step1_path.lower():
        return 'ljs', LJS_CONFIG
    else:
        # Default to VCTK if can't determine
        print("Warning: Cannot determine model type from filename, assuming VCTK (multi-speaker)")
        return 'vctk', VCTK_CONFIG


def init_rknn_model(rknn_path, target=None, device_id=None):
    """Initialize RKNN model and runtime"""
    rknn = RKNN(verbose=False)

    # Generate weight path by replacing .rknn with .weight
    weight_path = rknn_path[:-5] + ".weight"
    ret = rknn.load_rknn(rknn_path, weight_path, load_ctx=True)
    if ret != 0:
        print(f"Error: Failed to load RKNN model from {rknn_path}")
        print(f"Weight file: {weight_path}")
        return None

    if device_id:
        if target:
            ret = rknn.init_runtime(target=target, device_id=device_id, core_mask=0x01)
        else:
            ret = rknn.init_runtime(target='rk1828', device_id=device_id, core_mask=0x01)
    else:
        ret = rknn.init_runtime()

    if ret != 0:
        print(f"Error: Failed to initialize runtime for {rknn_path}")
        print(f"Target: {target if target else 'default'}, Device ID: {device_id if device_id else 'default'}")
        return None

    return rknn


def run_rknn_inference(rknn, inputs_list):
    """Run inference on RKNN model"""
    return rknn.inference(inputs=inputs_list, data_format='nchw')


def main():
    parser = argparse.ArgumentParser(description='VITS RKNN Sliding Window Inference - Unified (LJS/VCTK)')
    parser.add_argument('--step1', type=str,
                        help='Path to Step1 RKNN model (auto-detects LJS/VCTK if not specified)')
    parser.add_argument('--step2', type=str,
                        help='Path to Step2 RKNN model')
    parser.add_argument('--type', type=str, choices=['ljs', 'vctk'],
                        help='Model type: ljs (LJSpeech) or vctk (multi-speaker). Auto-detected if --step1 is specified.')
    parser.add_argument('--target', type=str, default=None,
                        help='Target RKNPU platform (e.g., rk1820, rk1828)')
    parser.add_argument('--device_id', type=str, default=DEFAULT_DEVICE_ID,
                        help='RKNN device ID')
    parser.add_argument('--text', type=str,
                        default="VITS is awesome! This is a text-to-speech system.",
                        help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output_audio_rknn.wav',
                        help='Output audio file path')
    parser.add_argument('--speaker_id', type=int, default=0,
                        help='Speaker ID (for VCTK models, 0-108)')
    parser.add_argument('--noise_scale', type=float, default=0.667,
                        help='Noise scale for randomness (default: 0.667)')
    parser.add_argument('--length_scale', type=float, default=1.0,
                        help='Length scale for speech speed (default: 1.0)')
    parser.add_argument('--rknn_dir', type=str, default=DEFAULT_RKNN_DIR,
                        help='Directory containing RKNN models')

    args = parser.parse_args()

    # Determine model paths and configuration
    if args.step1 and args.step2:
        # User specified both models explicitly
        step1_model = args.step1
        step2_model = args.step2
        model_type, config = detect_model_type(step1_model)
    elif args.step1:
        # User specified only step1, detect type and construct step2 path
        step1_model = args.step1
        model_type, config = detect_model_type(step1_model)
        step2_model = step1_model.replace('step1', 'step2')
    elif args.type:
        # User specified type, use default models for that type
        model_type = args.type
        config = VCTK_CONFIG if args.type == 'vctk' else LJS_CONFIG
        step1_model = os.path.join(args.rknn_dir, config['step1'])
        step2_model = os.path.join(args.rknn_dir, config['step2'])
    else:
        # No type or model specified, error
        print("Error: Please specify either --type (ljs/vctk) or --step1 model path")
        print("Examples:")
        print("  python infer_rknn_slid_unified.py --type ljs --text 'Hello world'")
        print("  python infer_rknn_slid_unified.py --step1 ../model/vits_vctk_step1_slid_fp.rknn --text 'Hello world'")
        return 1

    print(f"VITS RKNN Inference - {model_type.upper()} mode")
    if config['multi_speaker']:
        print(f"Multi-speaker: {config['num_speakers']} speakers available")
        print(f"Using speaker ID: {args.speaker_id}")

    # Import text processing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from text import text_to_sequence as actual_text_to_sequence

    # Configuration
    text_cleaners = ["english_cleaners2"]
    sampling_rate = 22050
    hop_length = 256
    text_window_size = 128
    text_context_size = 16
    text_max_len = text_window_size + 2 * text_context_size  # 160
    audio_window_size = 128

    # Process text
    stn_tst = actual_text_to_sequence(args.text, text_cleaners)
    stn_tst = intersperse(stn_tst, 0)
    stn_tst = np.array(stn_tst, dtype=np.int64)

    print(f"Text length: {len(stn_tst)} tokens")

    # Initialize RKNN models
    print(f"Loading RKNN models...")
    start_time = time.time()

    try:
        rknn_step1 = init_rknn_model(step1_model, args.target, args.device_id)
        rknn_step2 = init_rknn_model(step2_model, args.target, args.device_id)
        if rknn_step1 is None or rknn_step2 is None:
            print("Error: Failed to initialize RKNN models")
            return 1
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    init_time = time.time() - start_time
    start_time = time.time()
    print(f"Models loaded in {init_time:.2f}s")

    # STEP 1: Text Encoder with Sliding Window
    total_text_len = len(stn_tst)
    num_text_windows = (total_text_len + text_window_size - 1) // text_window_size

    sdp_rand = np.random.randn(1, 2, text_max_len).astype(np.float32)

    w_windows, m_p_windows, logs_p_windows, x_mask_windows = [], [], [], []
    g_windows = []  # For VCTK speaker embeddings
    step1_time = 0

    for i in range(num_text_windows):
        start_idx = i * text_window_size
        end_idx = min(start_idx + text_window_size, total_text_len)

        context_start = max(0, start_idx - text_context_size)
        context_end = min(total_text_len, end_idx + text_context_size)

        text_window_extended = stn_tst[context_start:context_end]
        text_len_extended = len(text_window_extended)

        if text_len_extended < text_max_len:
            pad_len = text_max_len - text_len_extended
            text_window_extended = np.pad(text_window_extended, (0, pad_len), mode='constant')
            text_len_extended = text_max_len

        x = text_window_extended.reshape(1, -1).astype(np.int64)
        x_lengths = np.array([text_len_extended], dtype=np.int64)

        indices = np.cumsum(np.ones(text_len_extended, dtype=np.int32)) - 1
        x_mask = (indices.reshape(1, -1) < x_lengths.reshape(-1, 1)).astype(np.float32)
        x_mask = x_mask.reshape(1, 1, -1)

        length_scale = np.array([args.length_scale], dtype=np.float32)
        noise_scale_w = np.array([0.8], dtype=np.float32)

        # Prepare step1 inputs based on model type
        if config['multi_speaker']:
            # VCTK: include speaker embedding (6 inputs)
            sid = np.array([args.speaker_id], dtype=np.int64)
            step1_inputs = [
                x.astype(np.int32),
                x_mask.astype(np.float32),
                length_scale,
                noise_scale_w,
                sdp_rand.astype(np.float32),
                sid.astype(np.int32)
            ]
        else:
            # LJS: single speaker (5 inputs)
            step1_inputs = [
                x.astype(np.int32),
                x_mask.astype(np.float32),
                length_scale,
                noise_scale_w,
                sdp_rand.astype(np.float32)
            ]

        window_start = time.time()
        step1_outputs = run_rknn_inference(rknn_step1, step1_inputs)
        step1_time += time.time() - window_start

        if config['multi_speaker']:
            w, m_p, logs_p, x_mask_out, g = step1_outputs
            g_windows.append(g)  # Store speaker embedding
        else:
            w, m_p, logs_p, x_mask_out = step1_outputs[:4]

        actual_context_before = start_idx - context_start
        core_start = actual_context_before
        core_len = end_idx - start_idx
        core_end = min(core_start + core_len, w.shape[2])

        w_windows.append(w[:, :, core_start:core_end])
        m_p_windows.append(m_p[:, :, core_start:core_end])
        logs_p_windows.append(logs_p[:, :, core_start:core_end])
        x_mask_windows.append(x_mask_out[:, :, core_start:core_end])

    print(f"Step1: {num_text_windows} windows, {step1_time:.2f}s")

    # Concatenate text encoder outputs
    w_full = np.concatenate(w_windows, axis=2)
    m_p_full = np.concatenate(m_p_windows, axis=2)
    logs_p_full = np.concatenate(logs_p_windows, axis=2)
    x_mask_full = np.concatenate(x_mask_windows, axis=2)

    # For VCTK, extract speaker embedding
    g = g_windows[0] if config['multi_speaker'] else None

    # Compute audio length
    w_ceil = np.ceil(w_full).astype(np.float32)
    y_lengths_np = np.maximum(np.sum(w_ceil, axis=(1, 2)), 1).astype(np.int64)

    # Compute z_p using attention
    t_y = int(y_lengths_np[0])
    indices = np.cumsum(np.ones(t_y, dtype=np.int32)) - 1
    y_mask_full = (indices.reshape(1, -1) < y_lengths_np.reshape(-1, 1)).astype(np.float32)
    y_mask_full = y_mask_full.reshape(1, 1, -1)

    attn_mask = x_mask_full.reshape(1, 1, 1, -1) * y_mask_full.reshape(1, 1, -1, 1)
    attn = generate_path_numpy(w_ceil, attn_mask)

    m_p_t = m_p_full.transpose(0, 2, 1)
    logs_p_t = logs_p_full.transpose(0, 2, 1)

    m_p_full_attn = np.matmul(attn.squeeze(1), m_p_t).transpose(0, 2, 1)
    logs_p_full_attn = np.matmul(attn.squeeze(1), logs_p_t).transpose(0, 2, 1)

    z_p = m_p_full_attn + np.random.randn(*m_p_full_attn.shape) * np.exp(logs_p_full_attn) * args.noise_scale

    # STEP 2: Audio Generation with Sliding Window
    total_audio_frames = y_lengths_np[0]
    num_audio_windows = (total_audio_frames + audio_window_size - 1) // audio_window_size

    padded_length = num_audio_windows * audio_window_size
    if padded_length > total_audio_frames:
        pad_frames = padded_length - total_audio_frames
        z_p = np.pad(z_p, ((0, 0), (0, 0), (0, pad_frames)), mode='constant', constant_values=0)
        y_mask_full = np.pad(y_mask_full, ((0, 0), (0, 0), (0, pad_frames)), mode='constant', constant_values=0)

    audio_segments = []
    step2_time = 0

    for i in range(num_audio_windows):
        start_idx = i * audio_window_size
        end_idx = min(start_idx + audio_window_size, padded_length)

        z_p_window = z_p[:, :, start_idx:end_idx]
        y_mask_window = y_mask_full[:, :, start_idx:end_idx]

        if z_p_window.shape[2] < audio_window_size:
            pad_frames = audio_window_size - z_p_window.shape[2]
            z_p_window = np.pad(z_p_window, ((0, 0), (0, 0), (0, pad_frames)), mode='constant', constant_values=0)
            y_mask_window = np.pad(y_mask_window, ((0, 0), (0, 0), (0, pad_frames)), mode='constant', constant_values=0)

        # Prepare step2 inputs based on model type
        if config['multi_speaker']:
            # VCTK: include speaker embedding (3 inputs)
            step2_inputs = [
                z_p_window.astype(np.float32),
                g.astype(np.float32),
                y_mask_window.astype(np.float32)
            ]
        else:
            # LJS: single speaker (2 inputs)
            step2_inputs = [
                z_p_window.astype(np.float32),
                y_mask_window.astype(np.float32)
            ]

        window_start = time.time()
        step2_outputs = run_rknn_inference(rknn_step2, step2_inputs)
        step2_time += time.time() - window_start

        audio = step2_outputs[0]

        if i == num_audio_windows - 1:
            valid_frames = total_audio_frames - start_idx
            valid_samples = min(valid_frames * hop_length, audio.shape[2])
        else:
            valid_frames = audio_window_size
            valid_samples = audio_window_size * hop_length

        audio_window = audio[0, 0, :valid_samples]
        audio_segments.append(audio_window)

    # Concatenate audio segments
    final_audio = np.concatenate(audio_segments)

    # Performance summary
    total_time = time.time() - start_time
    audio_duration = len(final_audio) / sampling_rate
    rtf = total_time / audio_duration

    print(f"Audio: {len(final_audio)} samples, {audio_duration:.2f}s")
    print(f"Total inference: {total_time:.2f}s, RTF: {rtf:.3f}")

    # Save audio
    write(args.output, sampling_rate, final_audio)
    print(f"Saved audio to {args.output}")

    # Cleanup
    rknn_step1.release()
    rknn_step2.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())