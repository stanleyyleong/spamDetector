import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the saved model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Abbreviation dictionary
abbreviation_dict = {
    "n": "and", "u": "you", "r": "are", "ur": "your", 'dun': "dont", "pls": "please",
    "wif": "with", "tmr": "tomorrow", "thx": "thanks", "lol": "laughing out loud",
    "idk": "I don't know", "brb": "be right back", "btw": "by the way", "gr8": "great",
    "b4": "before", "omg": "oh my god", "ttyl": "talk to you later", "imo": "in my opinion",
    "smh": "shaking my head", "np": "no problem", "tbh": "to be honest", "afaik": "as far as I know",
    "irl": "in real life", "ftw": "for the win", "fyi": "for your information", "nvm": "never mind",
    "bc": "because", "cya": "see you", "asap": "as soon as possible", "atm": "at the moment",
    "bff": "best friends forever", "bday": "birthday", "lmk": "let me know", "omw": "on my way",
    "ppl": "people", "thnx": "thanks", "wya": "where you at", "wyd": "what you doing",
    "yw": "you're welcome", "hmu": "hit me up", "rofl": "rolling on the floor laughing",
    "ily": "I love you", "ikr": "I know right", "jk": "just kidding", "msg": "message",
    "sry": "sorry", "sup": "what's up", "tgif": "thank god it's Friday", "yolo": "you only live once"
}

# Function to expand abbreviations
def expand_abbreviations(text):
    words = text.split()
    expanded_words = [abbreviation_dict.get(word, word) for word in words]
    return ' '.join(expanded_words)

# Function to remove punctuation from the text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Expanding abbreviations
    text = expand_abbreviations(text)
    # Remove punctuation
    text = remove_punctuation(text)
    # Tokenizing the text into words
    words = word_tokenize(text)
    # Removing stop words from the tokenized words
    words = [word for word in words if word not in stop_words]
    # Lemmatizing the words
    words = [lemmatizer.lemmatize(word) for word in words]
    # Joining the words back into a single string
    return ' '.join(words)

# Streamlit app
st.title("Spam Email Detector")
st.write("Enter the email text below to check if it's spam or not:")

user_input = st.text_area("Email Text")
if st.button("Predict"):
    if user_input:
        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)
        # Vectorize the preprocessed input
        input_vec = vectorizer.transform([preprocessed_input])
        # Predict using the loaded model
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.write("The email is **Spam**.")
        else:
            st.write("The email is **Ham**.")
    else:
        st.write("Please enter some text to predict.")
