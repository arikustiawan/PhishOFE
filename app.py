import streamlit as st

# --- PAGE SETUP ---
Predict_page = st.Page(
    page="pages/1_Predict_URL_.py",
    title="Predict URL",
    icon=":material/account_circle:",
    default=True,
)

Upload_page = st.Page(
    page="views/sales_dashboard.py",
    title="Sales Dashboard",
    icon=":material/bar_chart:",
)

Performance_page = st.Page(
    page="views/chatbot.py",
    title="Chat Bot",
    icon=":material/smart_toy:",
)
