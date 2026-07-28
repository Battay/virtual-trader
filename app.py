"""Entry point for the multi-page PSX Streamlit dashboard."""

import streamlit as st


st.set_page_config(page_title="PSX Dashboard", layout="wide")
st.session_state.setdefault("collection_result", None)

navigation = st.navigation(
    [
        st.Page(
            "app_pages/1_Fetch_Data.py",
            title="Fetch data",
            icon=":material/download:",
            default=True,
        ),
        st.Page(
            "app_pages/2_Dataset_Explorer.py",
            title="Dataset explorer",
            icon=":material/table_view:",
        ),
        st.Page(
            "app_pages/3_Stock_Explorer.py",
            title="Stock explorer",
            icon=":material/show_chart:",
        ),
        st.Page(
            "app_pages/4_Automation.py",
            title="Automation",
            icon=":material/schedule:",
        ),
    ],
    position="top",
)
navigation.run()
