"""Entry point for the multi-page PSX Streamlit dashboard."""

import streamlit as st


st.set_page_config(page_title="PSX Dashboard", layout="wide")
st.session_state.setdefault("collection_result", None)

navigation = st.navigation(
    [
        st.Page(
            "app_pages/1_Fetch_Data.py",
            title="Fetch Data",
            icon=":material/download:",
            default=True,
        ),
        st.Page(
            "app_pages/2_Dataset_Explorer.py",
            title="Dataset Explorer",
            icon=":material/table_view:",
        ),
        st.Page(
            "app_pages/3_Stock_Explorer.py",
            title="Stock Explorer",
            icon=":material/show_chart:",
        ),
        st.Page(
            "app_pages/4_Automation.py",
            title="Automation",
            icon=":material/schedule:",
        ),
        st.Page(
            "app_pages/5_Company_Registry.py",
            title="Company Registry",
            icon=":material/corporate_fare:",
        ),
        st.Page(
            "app_pages/6_Training_and_Models.py",
            title="Training & Models",
            icon=":material/model_training:",
        ),
        st.Page(
            "app_pages/7_Historical_Backfill.py",
            title="Historical Backfill",
            icon=":material/history:",
        ),
    ],
    position="top",
)
navigation.run()
