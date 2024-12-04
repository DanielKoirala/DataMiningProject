import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import csv

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
    try:
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        text = text.split()  # Tokenize
        text = [stemmer.stem(word) for word in text if word not in stop_words]  # Remove stopwords and stem
        return ' '.join(text)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return ""

def clean_dataset(filepath, output_filepath):
    """
    Cleans the dataset by removing control characters and merging multiline entries.
    
    Args:
        filepath (str): Path to the raw CSV file.
        output_filepath (str): Path to save the cleaned CSV file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            merged_line = ""
            for line in infile:
                # Remove control characters
                line = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line)
                # Detect line continuation
                if line.count('"') % 2 != 0:
                    merged_line += line.strip()
                else:
                    merged_line += line.strip()
                    writer.writerow([merged_line])
                    merged_line = ""
            print(f"Cleaned dataset saved to {output_filepath}")
    except Exception as e:
        print(f"Error in clean_dataset: {e}")

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the dataset.
    
    Args:
        filepath (str): Path to the cleaned CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with original text, cleaned text, and labels.
    """
    try:
        # Load the cleaned CSV file and handle problematic rows
        data = pd.read_csv(filepath, encoding="utf-8", quoting=csv.QUOTE_NONE)
        
        # Ensure required columns are present
        if not {'Message', 'Spam/Ham'}.issubset(data.columns):
            raise ValueError("Dataset must contain 'Message' and 'Spam/Ham' columns.")

        # Select relevant columns and drop missing values
        data = data[['Message', 'Spam/Ham']]
        data.dropna(inplace=True)

        # Preprocess text and map labels to binary
        data['cleaned_text'] = data['Message'].apply(preprocess_text)
        data['label'] = data['Spam/Ham'].map({'ham': 0, 'spam': 1})

        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        raise
    except pd.errors.ParserError as e:
        print(f"Error reading file '{filepath}': {e}")
        raise
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        raise

def extract_features(data, max_features=5000):
    """
    Converts preprocessed text data into TF-IDF features.
    
    Args:
        data (pd.DataFrame): DataFrame with a 'cleaned_text' column.
        max_features (int): Maximum number of features for TF-IDF vectorization.
    
    Returns:
        tuple: Transformed feature matrix and the fitted TF-IDF vectorizer.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(data['cleaned_text'])
        return X, vectorizer
    except Exception as e:
        print(f"Error in extract_features: {e}")
        raise

# Debugging entry point
if __name__ == "__main__":
    raw_filepath = "./data/spam_dataset.csv"
    cleaned_filepath = "./data/cleaned_spam_dataset.csv"

    print("Cleaning dataset...")
    clean_dataset(raw_filepath, cleaned_filepath)

    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(cleaned_filepath)
    print(f"Data loaded. Number of rows: {len(data)}")
    print("Extracting features...")
    X, vectorizer = extract_features(data)
    print("Feature extraction complete.")
