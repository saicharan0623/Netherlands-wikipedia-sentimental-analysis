import streamlit as st
import pickle
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

# Word Cloud Visualization of the input sentence
if user_input:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
    st.image(wordcloud.to_array(), caption="Word Cloud", use_container_width=True)

if st.button("Predict Sentiment"):
    # Transform the input text using the same vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Predict sentiment
    prediction = model.predict(user_input_tfidf)[0]
    
    # Display result with a color change to make it more visual
    if prediction == 'positive':
        st.success(f"Sentiment: Positive")
    else:
        st.error(f"Sentiment: Negative")
    
    # Get probability and visualize with a bar chart
    proba = model.predict_proba(user_input_tfidf)[0]
    
    # Create a bar chart for positive/negative sentiment probabilities
    sentiment_labels = ['Negative', 'Positive']
    fig, ax = plt.subplots()
    ax.bar(sentiment_labels, proba, color=['red', 'green'])
    ax.set_title('Sentiment Prediction Confidence')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Display TextBlob analysis for comparison
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Visualize polarity and subjectivity using seaborn
    st.write(f"TextBlob Polarity: {polarity:.2f}")
    st.write(f"TextBlob Subjectivity: {subjectivity:.2f}")
    
    # Plot Polarity and Subjectivity
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=['Polarity', 'Subjectivity'], y=[polarity, subjectivity], ax=ax2, palette='coolwarm')
    ax2.set_ylim(-1, 1)
    ax2.set_title('TextBlob Sentiment Analysis')
    st.pyplot(fig2)

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
