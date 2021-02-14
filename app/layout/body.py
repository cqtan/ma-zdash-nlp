import streamlit as st
from charts.category_charts import render_cat_by_count, render_cat_by_time
from charts.sentiment_charts import (
    render_sen_by_cat,
    render_sen_by_time,
    sen_by_count_donut,
)
from charts.wordcloud_charts import render_wordcloud
from charts.entity_charts import render_entity_by_sen


def render_body(df_preds, df_ents):

    entities = df_ents["entity"].astype(str).tolist()

    # Category distribution (frequency)
    col1, col2 = st.beta_columns(2)
    with col1:
        render_cat_by_count(df_preds)

    # Category by time (percentage)
    with col2:
        render_cat_by_time(df_preds)

    # Sentiment by categories (percentage)
    col1, col2 = st.beta_columns(2)
    with col1:
        render_sen_by_cat(df_preds)

    # Sentiment by time (month)
    with col2:
        render_sen_by_time(df_preds)

    col1, col2 = st.beta_columns([1, 2])
    # Sentiment distribution (percentage)
    with col1:
        sen_by_count_donut(df_preds)

    # Most Frequent words (Wordcloud)
    with col2:
        render_wordcloud(df_preds, "feedback_text_en")

    # Entity by sentiment
    render_entity_by_sen(df_preds, entities)

    col1, col2 = st.beta_columns([4, 1])
    with col1:
        st.dataframe(df_preds)

    with col2:
        st.dataframe(df_ents)