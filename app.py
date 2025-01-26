import streamlit as st
from streamlit_option_menu import option_menu

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # required
        options=["Predict URL", "Projects", "Contact"],  # required
        icons=["house", "book", "envelope"],  # optional
        default_index=0
    )

if selected == "Predict URL":
    #st.title(f"You have selected {selected}")
    exec(open("pages/Predict_URL.py").read())
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")
