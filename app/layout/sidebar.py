import streamlit as st
from app.utils.load_data import read_csv_to_config, check_local_availability


def render_sidebar():
    sidebar = st.sidebar
    config = None
    data_link = None
    file_name = None
    file_name_default = "fb_11k.csv"
    col_text_default = "feedback_text_en"
    col_date_default = "dt"

    with sidebar:
        st.title("Setup the app here")
        data_location = st.radio(
            "Choose data location", ("Local", "Cloud Storage"), index=1
        )

        if data_location == "Cloud Storage":
            data_link = st.text_input(
                "Paste link to CSV here",
                value="https://drive.google.com/file/d/1Ilc_ApV-_JDNgFF3mx2SvsOz-XovP8LF/view?usp=sharing",
            )
        else:
            file_name = st.text_input("Enter file name", file_name_default)
            file_check_btn = st.button("Check file availability")
            if file_check_btn:
                is_available = check_local_availability(file_name)
                if is_available:
                    st.success("File found")
                else:
                    st.warning("File not found")

        limit = st.radio(
            "Choose how many rows to load",
            ("Sample (5)", "100", "1000", "complete"),
            index=1,
        )

        col_text = st.text_input("Column name of the input texts", col_text_default)
        col_date = st.text_input("Column name of the date", col_date_default)

        cols_optional = st.text_input("(Optional) Subset of columns to import")
        col1, col2 = st.beta_columns(2)

        # save_ckb = st.checkbox("Save results after running", value=False)

        with col1:
            load_btn = st.button("Load demo data")

        with col2:
            run_btn = st.button("Run NLP")

        if run_btn:
            if col_text and col_date:
                if data_location == "Local":
                    if file_name:
                        config = read_csv_to_config(
                            True, file_name, limit, col_text, col_date, cols_optional
                        )
                        st.text(
                            f"Loaded from local: {limit}/{config['total_rows']} rows"
                        )
                    else:
                        st.warning("Please provide a file name")
                else:
                    if data_link:
                        config = read_csv_to_config(
                            False, data_link, limit, col_text, col_date, cols_optional
                        )
                        st.text(
                            f"Loaded from cloud storage: {limit}/{config['total_rows']} rows"
                        )
                    else:
                        st.warning("Please provide the storage link")
            else:
                st.warning("Please provide both column name for text and date")

    return load_btn, run_btn, config
