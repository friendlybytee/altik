import base64
import logging
from pathlib import Path
from threading import Lock

import streamlit as st

from src.paths import BASE_DIR, DATA_DIR, ASSETS_DIR, MODELS_DIR
from src.answers import is_no_answer

from src.classifier import CATEGORIES, classify_question
from src.speech import transcribe_audio

from src.rag import (
    NO_ANSWER, SIMILARITY_THRESHOLD,
    build_index, expand_query, generate_answer, search,
)

st.set_page_config(page_title="Альтик", page_icon="🤖")
st.html(Path(__file__).with_name("style.css"))
picture, heading, brand = st.columns([1, 3, 1])
with picture:
    mascot_slot = st.empty()


MASCOT_PLACEHOLDERS = {
    "science": "🔬", "sport": "🏅", "art": "🎨", "general": "🤖",
    "greeting": "👋", "error": "😵",
}


def show_mascot(state=None):
    st.session_state.mascot_state = state
    with mascot_slot.container():
        animation = ASSETS_DIR / "animations" / f"{state}.gif" if state in MASCOT_PLACEHOLDERS else None
        if animation and animation.is_file():
            data = base64.b64encode(animation.read_bytes()).decode("ascii")
            st.image(f"data:image/gif;base64,{data}", width=110)
        elif state in MASCOT_PLACEHOLDERS:
            st.markdown(f"# {MASCOT_PLACEHOLDERS[state]}")
        elif (ASSETS_DIR / "mascot.png").is_file():
            st.image(str(ASSETS_DIR / "mascot.png"), width=110)
        else:
            st.markdown("# 🤖")


show_mascot(st.session_state.get("mascot_state", "greeting"))
with heading:
    st.title("Альтик")
    st.caption("Поиск по базе знаний Альтаира")
with brand:
    st.image(str(BASE_DIR / "assets" / "altair-logo.png"), width=110)

show_chunks = st.sidebar.checkbox("Показать найденные фрагменты")
voice_output = st.sidebar.checkbox("Озвучивать ответы")
if st.sidebar.button("Очистить историю"):
    st.session_state.messages = []
    show_mascot("greeting")


@st.cache_resource(show_spinner="Индексирую базу знаний…", max_entries=2)
def cached_index(text):
    return build_index(text)


@st.cache_resource(show_spinner=False)
def load_tts_model():
    from src.tts import TTSmodel

    return TTSmodel(str(MODELS_DIR / "v5_5_ru.pt"), "eugene"), Lock()


try:
    knowledge = (DATA_DIR / "knowledge.txt").read_text(encoding="utf-8-sig")
    chunks, index = cached_index(knowledge)
except (OSError, UnicodeError):
    show_mascot("error")
    st.error("Не удалось прочитать knowledge.txt. Создайте data/knowledge.txt в UTF-8.")
    st.stop()
except (RuntimeError, ValueError) as error:
    show_mascot("error")
    st.error(str(error))
    st.stop()


def display_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("audio") is not None:
            st.audio(message["audio"], sample_rate=48000)
        if message.get("tts_error"):
            st.warning(message["tts_error"])
        if show_chunks and message.get("hits"):
            with st.expander("Найденные фрагменты"):
                if message.get("search_query"):
                    st.caption(f"Поисковый запрос: {message['search_query']}")
                for hit in message["hits"]:
                    relevant = hit.get("relevant", hit["score"] >= SIMILARITY_THRESHOLD)
                    status = "в контексте" if relevant else "отброшен"
                    st.caption(f"Фрагмент {hit['id']} · {hit['score']:.3f} · {status}")
                    if hit.get("reason"):
                        st.caption(f"Причина: {hit['reason']}")
                    st.text(hit["text"])


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    display_message(message)

voice_question = None


def clear_voice_draft():
    st.session_state.pop("voice_text", None)


with st.expander("🎙️ Ввести вопрос голосом"):
    st.caption("Нажмите микрофон и запишите вопрос до 25 секунд. При первом распознавании загружается модель.")
    audio = st.audio_input(
        "Записать вопрос", sample_rate=16000, key="voice_recording",
        on_change=clear_voice_draft,
    )
    if st.button("Распознать запись", disabled=audio is None):
        st.session_state.pop("voice_text", None)
        try:
            with st.spinner("Распознаю речь… При первом запуске это может занять несколько минут."):
                st.session_state.voice_text = transcribe_audio(audio.getvalue())
        except (RuntimeError, ValueError) as error:
            show_mascot("error")
            st.error(str(error))
    if "voice_text" in st.session_state:
        st.text_area("Проверьте текст вопроса", key="voice_text")
        if st.button("Отправить голосовой вопрос", disabled=not st.session_state.voice_text.strip()):
            voice_question = st.session_state.voice_text.strip()

question = st.chat_input("Задайте вопрос по базе знаний") or voice_question
if question and question.strip():
    question = question.strip()
    message = {"role": "user", "content": question}
    st.session_state.messages.append(message)
    display_message(message)
    reply = {"role": "assistant", "hits": [], "search_query": expand_query(question)}
    try:
        with st.spinner("Ищу ответ…"):
            hits = search(question, chunks, index)
            reply["hits"] = hits
            context = "\n\n".join(
                f"[Фрагмент {hit['id']}]\n{hit['text']}"
                for hit in hits if hit["relevant"]
            )
            if context:
                category = classify_question(question, hits)
                reply["content"] = generate_answer(
                    question, context, CATEGORIES.get(category),
                )
                if category is None:
                    show_mascot("error")
                else:
                    show_mascot("error" if is_no_answer(reply["content"], NO_ANSWER) else category)
            else:
                reply["content"] = NO_ANSWER
                show_mascot("error")
    except (RuntimeError, ValueError) as error:
        reply.update(content=str(error), error=True)
        show_mascot("error")
    if voice_output and not reply.get("error"):
        try:
            with st.spinner("Озвучиваю ответ…"):
                model, lock = load_tts_model()
                with lock:
                    reply["audio"] = model.generate_audio(reply["content"])
        except Exception as error:
            show_mascot("error")
            logging.getLogger(__name__).exception("Ошибка синтеза речи")
            reply["tts_error"] = (
                f"Не удалось озвучить ответ: {type(error).__name__}: {error}"
            )
    st.session_state.messages.append(reply)
    display_message(reply)
