"""Convert an arbitrary .wav into a 16 kHz mono float32 PCM .wav so that the
Gemma-4 C++ demo (miniaudio-based loader) reads back the *exact* same waveform
that the Python `Gemma3nAudioFeatureExtractor` sees.

Why this is needed
------------------
The C++ pipeline goes:
    miniaudio.ma_decoder(target = f32 mono 16kHz)  ->  rknn_gemma4_audio_preprocess

When the source wav is not already 16 kHz mono float32, miniaudio implicitly
performs:
    - channel down-mix (stereo -> mono)
    - sample-rate conversion (its default resampler is *linear*, low quality)
    - format conversion (s16/s24/f32 -> f32)

Each of those silently introduces a slight time shift / spectral mismatch
relative to what librosa+numpy gives the Python `feature_extractor` on the
exact same source file. Empirically that produces a ~50 ms (5-frame)
offset and ~1 dB distortion at low frequencies in the log-mel input,
which propagates all the way into the audio_tower output and breaks the
Python <-> C++ comparison even though the preprocessing math is correct.

Saving the librosa-loaded waveform as a "canonical" 16 kHz/mono/float32
WAV makes miniaudio's path a no-op decode -> the byte pattern in
`audio->data[]` is identical to what Python uses, and any remaining
mismatch is purely in the C++ STFT/mel code (which is what we want to
isolate).

Usage
-----
    cd .../examples/gemma4/python/audio
    python convert_wav_for_cpp.py \
        --src ../../../Qwen2_5_Omni/data/audio/demo.wav \
        --dst ../../data/audio/demo_16k_mono_f32.wav

Then in the C++ demo, point at `demo_16k_mono_f32.wav` instead of the
original.
"""

from __future__ import annotations

import argparse
import os
import sys


def _try_import_librosa():
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


def _try_import_soundfile():
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Convert any wav to a 16kHz / mono / float32 PCM wav so the "
                     "C++ Gemma-4 demo reads it byte-identically to Python.")
    )
    parser.add_argument(
        "--src", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../Qwen2_5_Omni/data/audio/demo.wav",
        )),
        help="Input wav path (any sample rate / channels / format).",
    )
    parser.add_argument(
        "--dst", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../data/audio/demo_16k_mono_f32.wav",
        )),
        help="Output wav path (will be 16kHz mono float32 PCM).",
    )
    parser.add_argument(
        "--target_sr", type=int, default=16000,
        help="Target sample rate (default 16000, matching the Gemma-4 audio tower).",
    )
    parser.add_argument(
        "--res_type", type=str, default=None,
        help="librosa resampler. Default = librosa's default ('soxr_hq' in "
             "recent versions, which is also what HF processors use under "
             "the hood when calling `librosa.load(..., sr=...)`). "
             "Other useful options: 'soxr_vhq', 'soxr_hq', 'soxr_mq', "
             "'polyphase' (scipy), 'scipy' (scipy.signal.resample), "
             "'kaiser_best' / 'kaiser_fast' (need `pip install resampy`).",
    )
    parser.add_argument(
        "--peek", type=int, default=8,
        help="How many leading samples to print for sanity check (default 8).",
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isfile(src):
        print(f"[error] source wav not found: {src}", file=sys.stderr)
        return 1

    if not _try_import_librosa():
        print("[error] librosa is required: pip install librosa", file=sys.stderr)
        return 2
    if not _try_import_soundfile():
        print("[error] soundfile is required: pip install soundfile", file=sys.stderr)
        return 2

    import librosa
    import soundfile as sf
    import numpy as np

    # 1) Probe the original file (without resampling) so we can report what
    #    miniaudio would have had to convert.
    src_info = sf.info(src)
    print(f"[src] path:       {src}")
    print(f"[src] sample_rate {src_info.samplerate} Hz")
    print(f"[src] channels:   {src_info.channels}")
    print(f"[src] format:     {src_info.format} / {src_info.subtype}")
    print(f"[src] frames:     {src_info.frames} ({src_info.frames / src_info.samplerate:.3f} s)")

    # 2) Load + downmix-to-mono + resample-to-target_sr the same way HF
    #    processors do internally (librosa.load with mono=True, sr=target_sr).
    #    If --res_type is None we let librosa pick its default ('soxr_hq' in
    #    recent versions), which is exactly what `Gemma3nProcessor` /
    #    `apply_chat_template` end up calling. We also walk a graceful
    #    fallback chain so the script still works in environments without
    #    `resampy` / `soxr` installed.
    load_kwargs = dict(sr=args.target_sr, mono=True)
    if args.res_type:
        load_kwargs["res_type"] = args.res_type

    res_chain = [args.res_type] if args.res_type else [None]
    res_chain.extend([rt for rt in ("soxr_hq", "polyphase", "scipy") if rt not in res_chain])

    wav = None
    last_err = None
    for rt in res_chain:
        try_kwargs = dict(load_kwargs)
        if rt is not None:
            try_kwargs["res_type"] = rt
        else:
            try_kwargs.pop("res_type", None)
        try:
            wav, sr = librosa.load(src, **try_kwargs)
            print(f"[py ] loaded with res_type={rt!r}")
            break
        except Exception as e:
            print(f"[py ] librosa.load failed with res_type={rt!r}: "
                  f"{type(e).__name__}: {e}")
            last_err = e

    if wav is None:
        print("[error] all librosa resampler backends failed. "
              "Install one of: `pip install soxr` (recommended), "
              "`pip install resampy`, or `pip install scipy`.",
              file=sys.stderr)
        if last_err is not None:
            raise last_err
        return 3

    wav = wav.astype(np.float32, copy=False)

    print()
    print(f"[py ] target_sr:  {sr} Hz")
    print(f"[py ] frames:     {len(wav)} ({len(wav) / sr:.3f} s)")
    print(f"[py ] dtype:      {wav.dtype}")
    print(f"[py ] head {args.peek}:    {wav[:args.peek].tolist()}")
    print(f"[py ] tail {args.peek}:    {wav[-args.peek:].tolist()}")
    print(f"[py ] min/max/abs.mean: "
          f"{wav.min():+.6f} / {wav.max():+.6f} / {np.abs(wav).mean():.6f}")

    # 3) Save as 16-bit-aligned float32 PCM WAV. With subtype='FLOAT' and
    #    samplerate=target_sr, soundfile writes raw IEEE-float samples and
    #    miniaudio's f32/16kHz/mono target decode is a no-op memcpy.
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sf.write(dst, wav, sr, subtype="FLOAT")

    # Read it back and verify byte-identity.
    wav2, sr2 = sf.read(dst, dtype="float32", always_2d=False)
    if wav2.ndim == 2:
        wav2 = wav2[:, 0]
    bit_match = (wav.shape == wav2.shape) and np.array_equal(
        wav.view(np.uint32), wav2.view(np.uint32)
    )

    print()
    print(f"[dst] path:       {dst}")
    print(f"[dst] sample_rate {sr2} Hz, frames {len(wav2)}, dtype {wav2.dtype}")
    print(f"[dst] head {args.peek}:    {wav2[:args.peek].tolist()}")
    print(f"[dst] tail {args.peek}:    {wav2[-args.peek:].tolist()}")
    print(f"[dst] bit-identical to in-memory waveform: {bit_match}")

    if not bit_match:
        print("[warn] saved waveform does NOT round-trip bit-exactly. The C++ "
              "side may still differ slightly. Try a different writer/version "
              "of soundfile.", file=sys.stderr)

    # 4) HF feature-extractor pre-pad reminder. The processor pads the input
    #    waveform on the right to a multiple of 128 samples before unfolding
    #    with size=frame_length+1=513, step=hop_length=160. Reproduce the same
    #    shape so the C++ side knows exactly how many frames to expect and how
    #    many of them are valid.
    pad_to = 128
    L_padded = ((len(wav) + pad_to - 1) // pad_to) * pad_to
    frame_length = 512
    hop_length = 160
    hf_num_frames = max(0, (L_padded - (frame_length + 1)) // hop_length + 1)
    print()
    print(f"[hf ] L_padded (pad_to_multiple_of=128): {L_padded}")
    print(f"[hf ] expected mel frames after unfold:  {hf_num_frames}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
