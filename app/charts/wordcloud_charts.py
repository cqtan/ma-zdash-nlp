import streamlit as st
from wordcloud import WordCloud
from app.utils import preprocess
import app.charts.constants as con


def render_wordcloud(df, col_text):
    st.subheader("Word Cloud (Most Frequent words)")

    texts = df[col_text].astype(str).tolist()
    normalized_texts = preprocess.normalize_text_list(texts)

    corpus = []
    for text in normalized_texts:
        tokens = text.split()
        corpus.extend(tokens)

    text = " ".join(corpus)
    wordcloud = WordCloud(
        height=300,
        width=930,
        max_font_size=70,
        max_words=100,
        background_color=con.paper_bgcolor,
        collocations=False,
    ).generate(text)

    st.image(wordcloud.to_array(), use_container_width=True)
