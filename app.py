import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text_tokens = nltk.word_tokenize(text)

    filtered_tokens = []
    for token in text_tokens:
        if token.isalpha():
            filtered_tokens.append(token)

    y = []
    for token in filtered_tokens:
        if token not in stopwords.words('english') and token not in string.punctuation:
            y.append(token)

    s = []
    for i in y:
        s.append(ps.stem(i))

    return " ".join(s)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/Classifier')

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)

    # Pass the transformed text as a list or array to tfidf.transform()
    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("This is a spamming email")
    else:
        st.header("This is a non-spamming email")

