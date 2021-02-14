import streamlit as st


def render_header():
    st.title("zDash | NLP")
    with st.beta_expander("Description", expanded=True):
        st.text(
            "Gathering large amounts of annotated textual data by hand is a tedious and time consuming process. \nThe following demo application utilizes a light version of the popular BERT architecture known as \nDistilBERT to perform a small number of Natural Language Processing (NLP) tasks to infer insights \non customer feedback texts. This architecture allowed good confidence in its predictions with the \nuse of a relatively small number of training data. Insights gained are illustrated as a collection \nof charts (dashboard) resulting from the following tasks: "
        )
        st.text(
            "- Sentiment Analysis (negative, neutral, positive)\n- Multi-Label Classification (Theme of the text)\n- Named Entity Recognition (Extraction of certain words)"
        )

        st.text(
            "Results are highly tied to the kind of data the application is trained on, which makes it fairly \ncontext dependent to its domain and tasks. While its accuracy could still be improved the main goal \nwas to quickly prototype an application that does not require much training resources in order to \nachieve usable results in an efficient manner."
        )

        st.text(
            "Setup the application in the sidebar to generate the dashboard housing the insights in the form \nof charts. You can either load the previous results or run a new session."
        )
