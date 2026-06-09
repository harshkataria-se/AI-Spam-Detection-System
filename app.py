import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

with st.sidebar:
    st.header("📌 Project Information")

    st.write("### AI Spam Detection System")

    st.write("""
    This application uses Machine Learning
    and Natural Language Processing (NLP)
    to classify messages as Spam or Not Spam.
    """)

    st.write("### Technologies Used")
    st.write("- Python")
    st.write("- Scikit-Learn")
    st.write("- TF-IDF")
    st.write("- Streamlit")

    st.write("### Model")
    st.write("Multinomial Naive Bayes")

# Page Config
st.set_page_config(
    page_title="AI Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# Load Model
model = joblib.load("models/spam_model.pkl")

# Title
st.title("📩 AI Spam Detection System")
st.markdown("Detect whether a message is **Spam** or **Not Spam** using Machine Learning.")

# Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# Input
message = st.text_area(
    "Enter a message",
    height=150,
    placeholder="Type or paste your SMS/email here..."
)

# Predict Button
if st.button("Analyze Message"):

    if message.strip():

        prediction = model.predict([message])[0]

        try:
            probability = model.predict_proba([message]).max() * 100
        except:
            probability = None

        result = "Spam" if prediction == "spam" else "Not Spam"

        if prediction == "spam":
            st.error(f"🚨 Spam Detected")
        else:
            st.success(f"✅ Not Spam")

        if probability:
            st.metric("Confidence Score", f"{probability:.2f}%")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Message": message[:50],
            "Result": result
        })

    else:
        st.warning("Please enter a message.")

# History Section

if st.button("🗑 Clear History"):
    st.session_state.history = []
    st.rerun()

if st.session_state.history:

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history[::-1])

    st.dataframe(
        history_df,
        use_container_width=True
    )

st.markdown("---")
st.caption("Built by Harsh Kataria using Machine Learning and Streamlit")