import json
import os
import random
import sys
import glob
import io
from types import MethodType
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(SCRIPT_DIR)

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


from Qwen3_TTS.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration


def resolve_path(path: str, base_dir: str = SCRIPT_DIR) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def tokenize_texts(tokenizer, texts: List[str], device: torch.device) -> List[torch.Tensor]:
    input_ids = []
    for text in texts:
        encoded = tokenizer(text=text, return_tensors="pt", padding=True)
        input_id = encoded["input_ids"].to(device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        input_ids.append(input_id)
    return input_ids


def load_cases_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(jsonl_path, "r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_index} in {jsonl_path}") from exc
            if not isinstance(item, dict):
                raise TypeError(f"Each JSONL line must be a dict in {jsonl_path}, got {type(item)} at line {line_index}")
            cases.append(item)
    return cases


def build_model(args):
    model_path = resolve_path(args.model_path, PYTHON_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    config = Qwen3TTSConfig.from_pretrained(model_path)
    config._attn_implementation = args.attn_implementation

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=torch.float32,
        attn_implementation=args.attn_implementation,
    ).eval()
    model._quant_tokenizer = tokenizer

    return model, tokenizer, model_path


def normalize_case(case: Dict[str, Any], case_index: int, source_name: str) -> Dict[str, Any]:
    if "text" not in case or not str(case["text"]).strip():
        raise ValueError(f"Case {case_index} in {source_name} must provide non-empty `text`")
    if "ref_audio" not in case or not str(case["ref_audio"]).strip():
        raise ValueError(f"Case {case_index} in {source_name} must provide non-empty `ref_audio`")

    normalized = dict(case)
    normalized.setdefault("language", "auto")
    normalized["ref_text"] = ""
    normalized["x_vector_only_mode"] = True
    return normalized


def reset_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for pattern in ("inputs_embeds_*.npy", "attention_mask_*.npy", "position_ids_*.npy", "num_logits_to_keep_*.npy"):
        for file_path in glob.glob(os.path.join(output_dir, pattern)):
            os.remove(file_path)
    for file_name in ("_dataset.txt",):
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    parsed = urlparse(audio_path)
    if parsed.scheme in ("http", "https"):
        with urlopen(audio_path) as response:
            audio_bytes = response.read()
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    else:
        audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def build_voice_clone_prompt(model, ref_audio_path: str) -> Dict[str, Any]:
    audio, sr = load_audio(ref_audio_path)
    ref_spk_audio = audio
    if sr != 24000:
        ref_spk_audio = librosa.resample(y=ref_spk_audio, orig_sr=sr, target_sr=24000)

    ref_spk_embedding = model.extract_speaker_embedding(audio=ref_spk_audio, sr=24000)
    return {
        "ref_code": [None],
        "ref_spk_embedding": [ref_spk_embedding],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }


def create_voice_clone_prompt(self, ref_audio: str, ref_text: str = "", x_vector_only_mode: bool = True) -> Dict[str, Any]:
    if not x_vector_only_mode:
        raise NotImplementedError("Only x_vector_only_mode=True is supported in this quant-data script.")
    return build_voice_clone_prompt(self, ref_audio)


def generate_voice_clone(
    self,
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str = "",
    x_vector_only_mode: bool = True,
    non_streaming_mode: bool = False,
    prepare_quant_data_dir: str = "",
    prepare_quant_return_last_hidden: bool = True,
) -> None:
    tokenizer = self._quant_tokenizer
    model_device = next(self.parameters()).device
    input_ids = tokenize_texts(tokenizer, [build_assistant_text(text)], model_device)
    voice_clone_prompt = self.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )
    self.generate(
        input_ids=input_ids,
        ref_ids=[None],
        voice_clone_prompt=voice_clone_prompt,
        languages=[language],
        non_streaming_mode=non_streaming_mode,
        prepare_quant_data_dir=prepare_quant_data_dir,
        prepare_quant_return_last_hidden=prepare_quant_return_last_hidden,
    )


def attach_voice_clone_helpers(model) -> None:
    model.create_voice_clone_prompt = MethodType(create_voice_clone_prompt, model)
    model.generate_voice_clone = MethodType(generate_voice_clone, model)


def run_case(model, output_dir: str, case: Dict[str, Any], case_dir: str) -> None:
    ref_audio_value = str(case["ref_audio"])
    parsed = urlparse(ref_audio_value)
    ref_audio_path = ref_audio_value if parsed.scheme in ("http", "https") else resolve_path(ref_audio_value, case_dir)
    model.generate_voice_clone(
        text=str(case["text"]),
        language=str(case["language"]),
        ref_audio=ref_audio_path,
        ref_text="",
        x_vector_only_mode=True,
        non_streaming_mode=False,
        prepare_quant_data_dir=output_dir,
        prepare_quant_return_last_hidden=True,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Prepare talker quantization data from voice-clone JSONL cases")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../../../CKPT/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quant_data",
        help="directory under current talker folder to save captured numpy inputs",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="attention implementation used when loading the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="random seed for reproducible sampling",
    )
    parser.add_argument(
        "--inputs_jsonl",
        type=str,
        default="./input_cases.jsonl",
        help="jsonl file; each line should contain text, language and ref_audio",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = resolve_path(args.output_dir, SCRIPT_DIR)
    reset_output_dir(output_dir)
    model, tokenizer, model_path = build_model(args)
    attach_voice_clone_helpers(model)

    total_cases = 0

    json_path = resolve_path(args.inputs_jsonl, os.getcwd())
    cases = load_cases_from_jsonl(json_path)
    case_dir = os.path.dirname(json_path)
    for case_index, case in enumerate(cases, start=1):
        normalized = normalize_case(case, case_index, json_path)
        print(
            f"[RUN] line={case_index} "
            f"language={normalized['language']} ref_audio={normalized['ref_audio']}"
        )
        run_case(
            model=model,
            output_dir=output_dir,
            case=normalized,
            case_dir=case_dir,
        )
        total_cases += 1

    print(f"[DONE] model_path={model_path}")
    print(f"[DONE] cases={total_cases}")
    print(f"[DONE] dataset={os.path.join(output_dir, '_dataset.txt')}")
