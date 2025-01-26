import streamlit as st
from streamlit_option_menu import option_menu
# Configure the page to remove the default header and footer
st.set_page_config(
    page_title="Your App Title",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Hide the Streamlit header, footer, and menu
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)
# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # required
        options=["Predict URL", "Projects", "Contact"],  # required
        icons=["house", "book", "envelope"],  # optional
    )

if selected == "Predict URL":
    st.title(f"You have selected {selected}")
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")
