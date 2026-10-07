"""Streamlit web interface for the Swahili horticulture question-answering assistant.

Run locally:   pip install -r requirements.txt   then   streamlit run streamlit_app.py
Deploy:        see README.md in this folder (Streamlit Community Cloud, free, public link).
"""
import html

import streamlit as st

from qa_pipeline import SwahiliHorticultureQA

st.set_page_config(page_title="Swahili Horticulture QA", page_icon="🌱")


@st.cache_resource
def load_pipeline():
    return SwahiliHorticultureQA()


qa = load_pipeline()

st.title("Swahili Horticulture Question Answering")
st.write(
    "Ask a question **in Swahili** about growing fruit, spices or vegetables. The assistant finds the most relevant passage in a "
    "collection of farming manuals, highlights the answer inside it, and **declines** questions it does not think the manuals can answer."
)
st.caption(
    "Limits: it only knows the 307 manual passages it was built on (horticulture, from the dataset of Lubawa, 2024). It can pick the "
    "wrong passage or the wrong words. It is not agronomic advice; check important decisions with an extension officer."
)

if "question" not in st.session_state:
    st.session_state.question = ""

if qa.examples:
    st.write("**Try an example** (the last ones are outside horticulture):")
    cols = st.columns(len(qa.examples))
    for i, ex in enumerate(qa.examples):
        if cols[i % len(cols)].button(f"Example {i + 1}", help=ex, key=f"ex{i}"):
            st.session_state.question = ex

question = st.text_area("Your question (Swahili)", key="question", height=80, placeholder="Mfano: Maeneo gani yanastawisha vizuri tunda la tufaa?")

if st.button("Ask", type="primary"):
    result = qa.answer(question)
    if result["status"] == "invalid":
        st.warning(result["message"])
    elif result["status"] == "declined":
        st.error("Declined. " + result["message"])
    else:
        st.success("Answer: " + result["answer"])
        p, a, b = result["passage"], result["answer_start"], result["answer_end"]
        st.markdown(
            "**Source passage** (answer highlighted):\n\n"
            f"<div style='line-height:1.6'>{html.escape(p[:a])}<mark>{html.escape(p[a:b])}</mark>{html.escape(p[b:])}</div>",
            unsafe_allow_html=True,
        )
    if result["status"] != "invalid":
        with st.expander("Confidence details"):
            st.write(
                f"Confidence {result['confidence']} (the assistant answers when this is at least {result['threshold']}). "
                f"Passage match score {result['bm25_normalized']}; the model's log-probability for its answer words {result['model_log_probability']}."
            )

with st.expander("About this model"):
    info = qa.info
    st.write(
        f"Reader: **{info.get('name', 'unknown')}**, trained from scratch on about 1,560 Swahili horticulture questions "
        f"(seed {info.get('seed')}, best epoch {info.get('best_epoch')}). Chosen as the best of three trained models "
        "(BiGRU, BiLSTM, Transformer encoder) by validation F1. Retrieval: BM25. All numbers and the code are in the project repository."
    )
