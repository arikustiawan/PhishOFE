import streamlit as st
from streamlit_option_menu import option_menu

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Projects", "Contact"],  # required
        icons=["house", "book", "envelope"],  # optional
    )

if selected == "Home":
    st.title(f"You have selected {selected}")
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")
