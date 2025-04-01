import streamlit as st
import pickle
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load the saved vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the Streamlit app
st.title("Netherlands Wikipedia Sentiment Analysis by Malde Saicharan")

st.write('''
This app predicts the sentiment of text based on a model trained on Netherlands Wikipedia data.
Enter a sentence below to analyze its sentiment.
''')

# Input text box
user_input = st.text_area("Enter a sentence:", "The Netherlands is a beautiful country with rich history.")

if st.button("Predict Sentiment"):
    # Transform the input text using the same vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Predict sentiment
    prediction = model.predict(user_input_tfidf)[0]
    
    # Display result
    if prediction == 'positive':
        st.success(f"Sentiment: Positive")
    else:
        st.error(f"Sentiment: Negative")
    
    # Get probability
    proba = model.predict_proba(user_input_tfidf)[0]
    st.write(f"Confidence: {max(proba):.2%}")
    
    # Display TextBlob analysis for comparison
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    st.write(f"TextBlob Polarity: {polarity:.2f}")
    st.write(f"TextBlob Subjectivity: {subjectivity:.2f}")

st.subheader("About this Model")
st.write('''
This sentiment analysis model was trained on text from the Netherlands Wikipedia page.
The data preparation included:
- Scraping and cleaning Wikipedia text
- Sentence tokenization and sentiment analysis with TextBlob
- TF-IDF vectorization
- Balancing with SMOTE
- Training multiple classification models
''')