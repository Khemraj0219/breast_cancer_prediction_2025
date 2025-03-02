import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------------------------
# 🧠 Define Hybrid Model
# ----------------------------------------
class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, stacking_model, hard_voting_model, soft_voting_model, weights):
        self.stacking_model = stacking_model
        self.hard_voting_model = hard_voting_model
        self.soft_voting_model = soft_voting_model
        self.weights = weights

    def predict(self, X):
        stacking_pred = self.stacking_model.predict(X).astype(int)
        hard_voting_pred = self.hard_voting_model.predict(X).astype(int)
        soft_voting_pred = self.soft_voting_model.predict(X).astype(int)

        combined_predictions = np.apply_along_axis(
            lambda x: np.bincount(x, weights=self.weights).argmax(),
            axis=0,
            arr=np.array([stacking_pred, hard_voting_pred, soft_voting_pred])
        )
        return combined_predictions

# ----------------------------------------
# 🚀 Load Hybrid Model
# ----------------------------------------
@st.cache_data
def load_hybrid_model():
    """Loads the trained hybrid model from 'hybrid_model.joblib'."""
    try:
        model = joblib.load("hybrid_model.joblib")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'hybrid_model.joblib' not found! Ensure the model file is in the same directory as this script.")
        return None

# Load the model
model = load_hybrid_model()

# ----------------------------------------
# 🖥 Streamlit Web App Interface
# ----------------------------------------
st.title("🩺 Breast Cancer Prediction Web App")

# ✅ Display Hint Message
st.markdown("📝 **Hint:** Benign (**B**) means **not cancerous**, while Malignant (**M**) means **cancerous**.")

# ✅ Apply CSS to increase font size in tables
st.markdown(
    """
    <style>
        table {
            font-size: 12px !important;
        }
        thead th {
            font-size: 14px !important;  /* Optional: Makes header slightly bigger */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Visitor Counter
if "visitor_count" not in st.session_state:
    st.session_state["visitor_count"] = 0
st.session_state["visitor_count"] += 1

st.write(f"👥 **Total Visitors:** {st.session_state['visitor_count']}")

# ----------------------------------------
#  Data Preprocessing Function
# ----------------------------------------
def preprocess_data(df):
    """Preprocess input data by handling missing values, scaling, and applying PCA."""
    
    if 'diagnosis' not in df.columns:
        st.error("❌ Error: The uploaded CSV file must contain a 'diagnosis' column.")
        return None, None

    # ✅ Convert Diagnosis Column to Binary (0 = Benign, 1 = Malignant)
    y = df['diagnosis'].replace({'B': 0, 'M': 1}).astype(int)
    X = df.drop(columns=['diagnosis'])

    # ✅ Handle Missing Values
    X.dropna(axis=1, how="all", inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # ✅ Apply Power Transformer
    transformer = PowerTransformer(method='yeo-johnson')
    X_transformed = transformer.fit_transform(X)

    # ✅ Apply Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)

    # ✅ Apply PCA (Ensure Minimum Features)
    n_components = min(9, X.shape[1])  # Adjust PCA components dynamically
    X_pca = PCA(n_components=n_components).fit_transform(X_scaled)

    return X_pca, y

# ✅ Upload CSV File
uploaded_file = st.file_uploader("📂 Upload a CSV File", type="csv")

if uploaded_file and model is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # ✅ Preprocess Data
        X_test, y_test = preprocess_data(df)
        
        if X_test is not None and y_test is not None:
            # ✅ Make Predictions
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # ✅ Display Results
            st.metric(label="📊 Model Accuracy", value=f"{accuracy*100:.2f}%")

            # ✅ Generate Classification Summary Table
            def generate_summary_table(y_test, predictions):
                """Creates a structured classification summary table."""
                total_benign = np.sum(y_test == 0)
                total_malignant = np.sum(y_test == 1)

                correctly_classified_benign = np.sum((y_test == 0) & (predictions == 0))
                misclassified_benign = total_benign - correctly_classified_benign

                correctly_classified_malignant = np.sum((y_test == 1) & (predictions == 1))
                misclassified_malignant = total_malignant - correctly_classified_malignant

                grand_total = len(y_test)
                grand_total_correctly_classified = correctly_classified_benign + correctly_classified_malignant
                grand_total_misclassified = misclassified_benign + misclassified_malignant

                summary_df = pd.DataFrame({
                    "Category": [
                        "Total Benign", "Correctly Classified Benign", "Misclassified Benign",
                        "Total Malignant", "Correctly Classified Malignant", "Misclassified Malignant",
                        "Grand Total", "Grand Total Correctly Classified", "Grand Total Misclassified"
                    ],
                    "Count": [
                        total_benign, correctly_classified_benign, misclassified_benign,
                        total_malignant, correctly_classified_malignant, misclassified_malignant,
                        grand_total, grand_total_correctly_classified, grand_total_misclassified
                    ]
                })

                return summary_df

            # ✅ Display Summary Table
            st.write("**Classification Summary Table**")
            summary_df = generate_summary_table(y_test, predictions)
            st.dataframe(summary_df)

            # ✅ Display Graph
            fig, ax = plt.subplots(figsize=(12, 6))  
            sns.barplot(x="Category", y="Count", data=summary_df, palette="coolwarm")
            plt.xticks(rotation=30, ha="right", fontsize=12)  
            plt.xlabel("Classification Categories", fontsize=14, fontweight="bold")
            plt.ylabel("Count", fontsize=14, fontweight="bold")
            plt.title(" Classification Results", fontsize=16, fontweight="bold")
            plt.tight_layout()  
            st.pyplot(fig)

            # ✅ Display Predictions Table
            df["Actual Diagnosis"] = y_test.replace({0: "Benign", 1: "Malignant"})
            df["Predicted Diagnosis"] = predictions
            df["Predicted Diagnosis"] = df["Predicted Diagnosis"].replace({0: "Benign", 1: "Malignant"})
            df["Match"] = np.where(df["Actual Diagnosis"] == df["Predicted Diagnosis"], "✔️ Match", "❌ Mismatch")

            # ✅ Function to highlight rows with dark red & dark green
            def highlight_rows(row):
                """Highlights rows based on match/mismatch."""
                return ['background-color: #228B22; color: white' if row['Match'] == "✔️ Match" 
                        else 'background-color: #B22222; color: white'] * len(row)

            # ✅ Display Predictions Table with Improved Highlighting
            st.write("🔍 **Predictions Table**")
            styled_df = df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ✅ Footer
st.markdown("---")
st.markdown(f"🔹 **Total Visitors to this Web App:** {st.session_state['visitor_count']}") 
st.markdown("🛠 **Built with ❤️ using Streamlit**")
