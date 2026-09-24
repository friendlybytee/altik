"""Распознавание ответа об отсутствии информации."""
import re


def is_no_answer(answer, refusal):
    """Учитываем стандартный отказ и близкие формулировки об отсутствии данных."""
    def normalize(text):
        return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))

    text = normalize(answer)
    return normalize(refusal) in text or bool(re.search(
        r"\bв (?:базе(?: знаний)?|контексте|предоставленном контексте) "
        r"(?:нет|недостаточно|отсутствует) (?:информации|информация|данных|сведений)\b",
        text,
    ))


