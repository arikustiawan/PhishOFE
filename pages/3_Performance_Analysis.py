import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

st.title("Model Performance Metrics")
st.sidebar.image("logo.jpg", use_container_width=True)

# Check if session state has model results
if "model_results" in st.session_state and st.session_state.model_results:
    results = st.session_state.model_results

    st.success("Model training results loaded successfully!")
    st.write("### Training Metrics:")
    st.write(f"- **Accuracy**: {results['test_accuracy']:.3f}")
    st.write(f"- **Precision**: {results['precision']:.3f}")
    st.write(f"- **Recall**: {results['recall']:.3f}")
    st.write(f"- **F1 Score**: {results['f1_score']:.3f}")

    # Ensure session state has y_test and y_pred_prob for plots
    if "y_test" in st.session_state and "y_pred_prob" in st.session_state:
        y_test = st.session_state.y_test
        y_pred_prob = st.session_state.y_pred_prob

        # 🔹 ROC Curve
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # 🔹 Precision-Recall Curve
        st.write("### Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="green", lw=2, label="Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        st.pyplot(fig)

    else:
        st.warning("No probability predictions found. Please re-train the model.")

else:
    st.warning("No training results found. Please train the model first.")

