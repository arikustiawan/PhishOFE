import streamlit as st
from streamlit_option_menu import option_menu

# Configure the page first
st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar menu with default page set to "Predict URL"
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # No title for the menu
        options=["Predict URL", "Projects", "Contact"],  # Menu options
        icons=["house", "book", "envelope"],  # Icons for the menu
        default_index=0  # Default to the first item (index 0 corresponds to "Predict URL")
    )

if selected == "Predict URL":
    # Include the logic or content for the Predict URL page
    st.title(f"You have selected {selected}")
    exec(open("pages/Predict_URL.py").read())  # Executes the code from Predict_URL.py
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")
