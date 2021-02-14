import streamlit as st


def render_footer():
    balloon_btn = st.button("Balloons...")
    if balloon_btn:
        st.balloons()
