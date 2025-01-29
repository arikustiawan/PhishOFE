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
    st.write(f"- **Accuracy**: {results['test_accuracy'] * 100:.2f}%")
    st.write(f"- **Precision**: {results['precision'] * 100:.2f}%")
    st.write(f"- **Recall**: {results['recall'] * 100:.2f}%")
    st.write(f"- **F1 Score**: {results['f1_score'] * 100:.2f}%")

    # Ensure session state has y_test and y_pred_prob for plots
    if "y_test" in st.session_state and "y_pred_prob" in st.session_state:
            y_test = st.session_state.y_test
            y_pred_prob = st.session_state.y_pred_prob
    
            # Create a figure with 2 subplots (side by side)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
    
            axes[0].plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
            axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title("ROC Curve")
            axes[0].legend(loc="lower right")
    
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    
            axes[1].plot(recall, precision, color="green", lw=2, label="Precision-Recall Curve")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve")
            axes[1].legend(loc="lower left")
    
            # Adjust layout and display in Streamlit
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.warning("No probability predictions found. Please re-train the model.")

else:
    st.warning("No training results found. Please train the model first.")

