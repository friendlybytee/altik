"""Локальное распознавание короткой записи микрофона через GigaAM-v3."""

import io
import os
import wave
from pathlib import Path
from threading import Lock

import numpy as np
import streamlit as st

MODEL_ID = "ai-sage/GigaAM-v3"
MODEL_REVISION = "e2e_rnnt"
SAMPLE_RATE = 16000
MAX_SECONDS = 25


@st.cache_resource(show_spinner=False)
def load_speech_model():
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoConfig, AutoModel

    cache = Path(HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    cache_dir = str(cache)
    if os.name == "nt":
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(cache_dir, buffer, len(buffer)):
            cache_dir = buffer.value
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, cache_dir=cache_dir,
    )
    config.cache_dir = cache_dir

    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        config=config, cache_dir=cache_dir,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, Lock()


def transcribe_audio(audio_bytes):
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            if (wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1
                    or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE"):
                raise ValueError("Нужна запись WAV: 16 кГц, моно, PCM 16 бит. Запишите голос кнопкой микрофона.")
            frames = wav.getnframes()
            if not frames:
                raise ValueError("Запись пустая. Запишите вопрос ещё раз.")
            if frames > SAMPLE_RATE * MAX_SECONDS:
                raise ValueError(f"Запись длиннее {MAX_SECONDS} секунд. Сформулируйте вопрос короче.")
            pcm = wav.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("Запись повреждена. Запишите вопрос ещё раз.")
    except (wave.Error, EOFError):
        raise ValueError("Не удалось прочитать запись. Запишите вопрос ещё раз.") from None

    samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    if np.max(np.abs(samples)) < 0.001:
        raise ValueError("Запись слишком тихая. Проверьте микрофон и повторите вопрос.")
    try:
        import torch

        model, lock = load_speech_model()
        parameter = next(model.parameters())
        wav_tensor = torch.from_numpy(samples).unsqueeze(0).to(
            device=parameter.device, dtype=parameter.dtype,
        )
        length = torch.tensor([len(samples)], device=parameter.device)
        with lock, torch.inference_mode():
            encoded, encoded_length = model(wav_tensor, length)
            result = model.model.decoding.decode(
                model.model.head, encoded, encoded_length,
            )[0]
    except ImportError:
        raise RuntimeError("Не хватает библиотек для голоса. Установите зависимости из data/requirements.txt.") from None
    except Exception as error:
        raise RuntimeError(
            "Не удалось запустить GigaAM. При первом запуске нужен интернет для загрузки "
            "модели с Hugging Face. Проверьте зависимости и доступную память. "
            f"Тип ошибки: {type(error).__name__}."
        ) from error
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Речь не распознана. Произнесите вопрос ближе к микрофону.")
    return result.strip()
