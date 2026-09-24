import os
import re
import textwrap
import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from num2words import num2words


class TTSmodel:
    def __init__(self, local_model_path, speaker):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_model_path = local_model_path

        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(f"Файл {self.local_model_path} не найден!")

        # Python handles Unicode Windows paths; PyTorch's native fopen may not.
        with open(self.local_model_path, "rb") as model_file:
            importer = torch.package.PackageImporter(model_file)
            self.model = importer.load_pickle("tts_models", "model")
        self.model.to(self.device)

        self.speaker = speaker
        self.sample_rate = 48000

    def replace_latin_with_transcript(self, text):
        trans_map = {
            'a': 'а', 'b': 'б', 'c': 'с', 'd': 'д', 'e': 'е', 'f': 'ф',
            'g': 'г', 'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л',
            'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'кью', 'r': 'р',
            's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
            'y': 'ы', 'z': 'з',
            'A': 'А', 'B': 'Б', 'C': 'С', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
            'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Дж', 'K': 'К', 'L': 'Л',
            'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'Q': 'Кью', 'R': 'Р',
            'S': 'С', 'T': 'Т', 'U': 'У', 'V': 'В', 'W': 'В', 'X': 'Кс',
            'Y': 'Ы', 'Z': 'З',
        }

        result_chars = []
        for ch in text:
            if ch in trans_map:
                result_chars.append(trans_map[ch])
            else:
                result_chars.append(ch)
        return ''.join(result_chars)

    def prepare_text(self, text):
        def replace_number(match):
            num = int(match.group(0))
            return num2words(num, lang='ru', to='cardinal')

        text = re.sub(r'\d+', replace_number, text)

        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('…', '...')

        text = re.sub(r'\s+', ' ', text).strip()
        text = self.replace_latin_with_transcript(text)
        return text

    def split_text(self, text, max_chars=400):
        chunks = []
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
            chunks.extend(textwrap.wrap(
                sentence, width=max_chars, break_long_words=True,
                break_on_hyphens=False,
            ))
        return chunks

    def generate_audio(self, text, speed = 0.95):
        prepared = self.prepare_text(text)
        chunks = self.split_text(prepared)
        if not chunks:
            raise ValueError("Текст для озвучки пуст.")

        audio_pieces = []
        with torch.no_grad():
            for index, chunk in enumerate(chunks, start=1):
                try:
                    audio = self.model.apply_tts(
                        text=chunk,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate
                    )
                    if audio.numel() == 0:
                        raise RuntimeError("Модель вернула пустой аудиофрагмент.")
                    audio_pieces.append(audio.detach().cpu().numpy())
                except Exception as e:
                    raise RuntimeError(
                        f"Ошибка синтеза фрагмента {index}/{len(chunks)} "
                        f"({len(chunk)} символов): {type(e).__name__}: {e}"
                    ) from e

        full_audio = np.concatenate(audio_pieces)

        if speed != 1.0:
            new_length = int(len(full_audio) / speed)
            full_audio = resample(full_audio, new_length).astype(np.float32)

        return full_audio

    def save_audio(self, audio_numpy, output_path):
        audio_max = np.max(np.abs(audio_numpy))
        if audio_max > 0:
            audio_normalized = audio_numpy / audio_max
        else:
            audio_normalized = audio_numpy

        audio_int16 = (audio_normalized * 32767).astype('int16')
        wavfile.write(output_path, self.sample_rate, audio_int16)
        print(f"Аудио сохранено: {os.path.abspath(output_path)}")
