"""Проверка вопросов: python tests/evaluate_questions.py data/questions.txt

Вход: UTF-8 TXT, один вопрос на строку (пустые строки пропускаются).
Выход: статистика в консоли и подробный CSV с вопросами и ответами.
Нужны те же зависимости, запущенная Ollama и .env, что и для app.py.
Проверяется наличие ответа, а не его фактическая правильность.
"""

import argparse
import csv
import sys
from pathlib import Path

# При прямом запуске файла Python добавляет в sys.path только папку tests.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_DIR
from src.answers import is_no_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, help="TXT: один вопрос на строку")
    parser.add_argument("--knowledge", type=Path,
                        default=DATA_DIR / "knowledge.txt")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "evaluation_results.csv",
                        help="CSV с результатами (по умолчанию data/evaluation_results.csv)")
    args = parser.parse_args()
    if args.output.resolve() in {args.questions.resolve(), args.knowledge.resolve()}:
        parser.error("Файл результатов не должен совпадать с вопросами или базой знаний.")
    try:
        questions = [line.strip() for line in
                     args.questions.read_text(encoding="utf-8-sig").splitlines()
                     if line.strip()]
        knowledge = args.knowledge.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        parser.exit(1, f"Не удалось прочитать входные файлы: {error}\n")
    if not questions:
        parser.error("Файл вопросов пуст.")
    from src.rag import NO_ANSWER, build_index, generate_answer, search

    print(f"Вопросов: {len(questions)}. Индексирую базу знаний…", flush=True)
    try:
        chunks, index = build_index(knowledge)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"Не удалось построить индекс: {error}\n")

    counts = {"answered": 0, "no_answer": 0, "error": 0}
    labels = {"answered": "ответ", "no_answer": "нет информации", "error": "ошибка"}
    try:
        with args.output.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["question", "status", "answer", "error"])
            writer.writeheader()
            for number, question in enumerate(questions, 1):
                answer, error_text = "", ""
                try:
                    hits = search(question, chunks, index)
                    context = "\n\n".join(
                        f"[Фрагмент {hit['id']}]\n{hit['text']}"
                        for hit in hits if hit["relevant"]
                    )
                    answer = generate_answer(question, context) if context else NO_ANSWER
                    status = "no_answer" if is_no_answer(answer, NO_ANSWER) else "answered"
                except (RuntimeError, ValueError) as error:
                    status, error_text = "error", str(error)
                counts[status] += 1
                writer.writerow(dict(question=question, status=status, answer=answer, error=error_text))
                output.flush()
                print(f"[{number}/{len(questions)}] {labels[status]}: {question}", flush=True)
                if error_text:
                    print(f"  {error_text}", flush=True)
    except OSError as error:
        parser.exit(1, f"Ошибка записи результатов: {error}\n")

    answered, unanswered = counts["answered"], counts["no_answer"]
    completed = answered + unanswered
    print(f"\nВсего вопросов: {len(questions)}")
    print(f"Ответил: {answered}")
    print(f"Нет информации: {unanswered}")
    print(f"Ошибки (не входят в отношение): {counts['error']}")
    print(f"Отношение ответил / нет информации: {answered}:{unanswered}")
    if unanswered:
        print(f"Численное отношение: {answered / unanswered:.4f}")
    else:
        print("Численное отношение не определено: отказов нет (деление на ноль).")
    if completed:
        print(f"Доля ответов среди запросов без ошибок: {answered / completed:.2%}")
    print(f"Подробные результаты: {args.output.resolve()}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
