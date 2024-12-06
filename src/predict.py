import argparse
import joblib
from preprocess import preprocess_text

# Load saved models and vectorizer
ensemble_model = joblib.load("./models/ensemble_model.pkl")
vectorizer = joblib.load("./models/vectorizer.pkl")

# Predict function with detailed debugging
def predict_email(text, threshold=0.5):
    """
    Predict whether an email is spam or not based on a probability threshold.
    Includes detailed debugging for preprocessing, vectorization, and prediction.
    """
    if not text:
        raise ValueError("Input text must be a non-empty string.")
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    print(f"Original text: {text}")
    print(f"Cleaned text: {cleaned_text}")
    
    # Transform text into TF-IDF features
    features = vectorizer.transform([cleaned_text])
    print(f"Feature vector (non-zero indices): {features.nonzero()}")
    print(f"Feature vector values:\n{features.toarray()}")
    
    # Get probabilities
    probabilities = ensemble_model.predict_proba(features)
    spam_probability = probabilities[0][1]  # Probability of being spam
    print(f"Spam probability: {spam_probability}")

    # Classify based on threshold
    result = "Spam" if spam_probability >= threshold else "Not Spam"
    print(f"Prediction: {result}")
    return result

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Spam classification script")
    parser.add_argument("text", type=str, help="The text to classify")
    args = parser.parse_args()

    # Perform prediction
    predict_email(args.text, threshold=0.1)
