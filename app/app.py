import streamlit as st
import pandas as pd
import time
from layout.header import render_header
from layout.sidebar import render_sidebar
from layout.body import render_body
from nlp import sentiment, mlc, ner

st.set_page_config(layout="wide")

t = time.time()
render_header()

load_btn, run_btn, save_ckb, config = render_sidebar()

entities = []
progress_bar = st.progress(0)
status_text = st.empty()

if run_btn:
    progress_bar.progress(20)
    status_text.text("Data loaded")
    df = config["df"]
    col_text = config["col_text"]

    progress_bar.progress(40)
    status_text.text("Sentiment Analysis in progress...")
    df_new = sentiment.run_sentiment_analysis(df, col_text)
    status_text.text("Done perfoming Sentiment Analysis")

    progress_bar.progress(60)
    status_text.text("Multi-label classification in progress...")
    df_new = mlc.run_multi_label_classification(df_new, col_text)
    status_text.text("Done performing Mulit-Label Classification")

    progress_bar.progress(80)
    status_text.text("Entity Recognition in progress...")
    df_preds, df_ents = ner.run_named_entity_recognition(df_new, col_text)
    status_text.text("Done performing Named Entity Recognition")

    if save_ckb:
        df_preds.to_csv("./app/data/df_preds.csv", index=False)
        df_ents.to_csv("./app/data/df_ents.csv", index=False)

    render_body(df_preds, df_ents)
    progress_bar.progress(100)

    elapsed_time = round(time.time() - t, 2)
    time_unit = "seconds"
    if elapsed_time > 60:
        elapsed_time = round(elapsed_time / 60, 2)
        time_unit = "minutes"

    status_text.success(
        f"Done with all tasks! (Elapsed time: {elapsed_time} {time_unit})"
    )

elif load_btn:
    df_preds = pd.read_csv("./app/data/df_preds.csv")
    df_ents = pd.read_csv("./app/data/df_ents.csv")

    render_body(df_preds, df_ents)
    progress_bar.progress(100)
    status_text.success(f"Loaded previous results: Rows ({len(df_preds)})")
