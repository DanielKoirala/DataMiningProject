import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK stopwords if not already downloaded
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
    Loads and preprocesses a CSV file containing text data.
    
    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with original text, cleaned text, and labels.
    """
    # Read the CSV file (adjust columns based on your dataset structure)
    data = pd.read_csv(filepath, encoding='latin-1', usecols=[0, 1], names=['label', 'text'], skiprows=1)
    data.dropna(inplace=True)  # Drop missing values
    data['cleaned_text'] = data['text'].apply(preprocess_text)  # Preprocess the text column
    return data

def extract_features(data, ngram_range=(1, 2), max_features=5000):
    """
    Converts preprocessed text data into TF-IDF features with n-grams.
    
    Args:
        data (pd.DataFrame): DataFrame with a 'cleaned_text' column.
        ngram_range (tuple): The range of n-grams to include (e.g., (1, 2) for unigrams and bigrams).
        max_features (int): Maximum number of features for TF-IDF vectorization.
    
    Returns:
        tuple: Transformed feature matrix and the fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(data['cleaned_text'])
    return X, vectorizer

# Debugging prints (optional)
if __name__ == "__main__":
    print("preprocess_text, load_and_preprocess_data, and extract_features are ready to use!")
