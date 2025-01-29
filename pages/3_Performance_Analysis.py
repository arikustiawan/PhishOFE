import streamlit as st

st.title("Model Performance Metrics")

# Check if session state has model results
if "model_results" in st.session_state and st.session_state.model_results:
    results = st.session_state.model_results

    st.success("Model training results loaded successfully!")
    st.write("### Training Metrics:")
    st.write(f"- **Accuracy**: {results['test_accuracy']:.3f}")
    st.write(f"- **Precision**: {results['precision']:.3f}")
    st.write(f"- **Recall**: {results['recall']:.3f}")
    st.write(f"- **F1 Score**: {results['f1_score']:.3f}")
else:
    st.warning("No training results found. Please train the model first.")

