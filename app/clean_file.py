# from utils.preprocess import remove_names
import pandas as pd
import streamlit as st

from utils.preprocess import remove_names_spacy

df = pd.read_csv("./app/data/fb_10k.csv")
df = df[["dt", "feedback_text_en"]].head(10)

texts = df["feedback_text_en"].astype(str).tolist()
texts_cleaned = remove_names_spacy(texts)
st.text(texts_cleaned)
