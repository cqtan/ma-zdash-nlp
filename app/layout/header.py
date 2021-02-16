import streamlit as st


def render_header():
    st.title("zDash | NLP")
    with st.beta_expander("Description", expanded=True):
        st.text("")

        st.markdown(
            "Gathering large amounts of annotated textual data by hand is a tedious and time consuming process. The following demo application utilizes a light version of the popular BERT architecture known as DistilBERT to perform a small number of Natural Language Processing (NLP) tasks to infer insights on customer feedback texts. This architecture allowed good confidence in its predictions with the use of a relatively small number of training data. Insights gained are illustrated as a collection of charts (dashboard) resulting from the following tasks: "
        )
        st.markdown(
            "- Sentiment Analysis (negative, neutral, positive)\n- Multi-Label Classification (Theme of the text)\n- Named Entity Recognition (Extraction of certain words)"
        )
        st.text("")

        st.markdown(
            "Results are highly tied to the kind of data the application is trained on, which makes it fairly context dependent to its domain and tasks. While its accuracy could still be improved the main goal was to quickly prototype an application that does not require much training resources in order to achieve usable results in an efficient manner."
        )

        st.markdown(
            "Setup the application in the sidebar to generate the dashboard housing the insights in the form of charts. You can either load the previous results or run a new session."
        )

        st.markdown(
            "Models are available on [HuggingFace](https://huggingface.co/CouchCat), which are trained on a small custom customer feedback dataset. This following dataset, however, has been cleaned and does not include numbers or names for demo purposes"
        )
