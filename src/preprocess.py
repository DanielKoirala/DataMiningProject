import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK stopwords if not already available
nltk.download('stopwords')

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Removing special characters
    - Converting to lowercase
    - Tokenizing
    - Removing stopwords
    - Applying stemming
    
    Args:
        text (str): The raw text to preprocess.
    
    Returns:
        str: The cleaned and preprocessed text.
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [stemmer.stem(word) for word in text if word not in stop_words]  # Remove stopwords and stem
    return ' '.join(text)

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the Enron dataset.
    
    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with original text, cleaned text, and labels.
    """
    # Load the CSV file and select relevant columns
    data = pd.read_csv(filepath)
    data = data[['Message', 'Spam/Ham']]  # Select relevant columns
    data.dropna(inplace=True)  # Drop rows with missing values
    data['cleaned_text'] = data['Message'].apply(preprocess_text)  # Preprocess the message
    data['label'] = data['Spam/Ham'].map({'ham': 0, 'spam': 1})  # Map labels to binary
    return data

def extract_features(data, max_features=5000):
    """
    Converts preprocessed text data into TF-IDF features.
    
    Args:
        data (pd.DataFrame): DataFrame with a 'cleaned_text' column.
        max_features (int): Maximum number of features for TF-IDF vectorization.
    
    Returns:
        tuple: Transformed feature matrix and the fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data['cleaned_text'])
    return X, vectorizer

# Debugging entry point
if __name__ == "__main__":
    print("preprocess_text, load_and_preprocess_data, and extract_features are ready to use!")
