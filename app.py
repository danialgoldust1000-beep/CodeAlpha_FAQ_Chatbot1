import streamlit as st
from googletrans import Translator

translator = Translator()

st.title("🌍 Language Translation Tool")
st.write("Translate text between multiple languages instantly.")

languages = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Arabic": "ar",
    "Persian": "fa",
    "Hindi": "hi"
}

text = st.text_area("Enter text to translate:")

col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox("Source Language", list(languages.keys()))

with col2:
    target_lang = st.selectbox("Target Language", list(languages.keys()))

if st.button("Translate"):
    if text:
        translated = translator.translate(
            text,
            src=languages[source_lang],
            dest=languages[target_lang]
        )
        st.success("Translated Text:")
        st.write(translated.text)
    else:
        st.warning("Please enter text to translate.")

