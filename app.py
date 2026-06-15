import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📰 Fake News Detector for Students")

news = st.text_area("Enter the news article text:")

if st.button("Check News"):
    if news.strip() != "":
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == 0:
            st.error("❌ This news is predicted to be FAKE.")
        else:
            st.success("✅ This news is predicted to be REAL.")
    else:
        st.warning("Please enter some news text.")