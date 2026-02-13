import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQs
with open("faqs.json", "r") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

st.title("🤖 FAQ Chatbot")
st.write("Ask a question and I will try to help you.")

user_input = st.text_input("Your Question:")

if user_input:
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    st.success(answers[index])
