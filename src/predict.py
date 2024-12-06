import joblib
from preprocess import preprocess_text

# Load saved models and vectorizer
ensemble_model = joblib.load("./models/ensemble_model.pkl")
vectorizer = joblib.load("./models/vectorizer.pkl")

# Predict function
def predict_email(text, return_prob=False):
    """
    Predict whether an email is spam or not.

    Args:
        text (str): The input email text to classify.
        return_prob (bool): Whether to return the spam probability.

    Returns:
        str: "Spam" or "Not Spam".
        float (optional): Probability of being spam if return_prob is True.
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Transform text into TF-IDF features
        features = vectorizer.transform([cleaned_text])
        
        # Get prediction
        prediction = ensemble_model.predict(features)
        if return_prob:
            prob = ensemble_model.predict_proba(features)[0][1]  # Probability of being spam
            return ("Spam" if prediction[0] == 1 else "Not Spam", prob)
        return "Spam" if prediction[0] == 1 else "Not Spam"
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    text = "Congratulations! You've won a free gift card. Click here to claim."
    result = predict_email(text, return_prob=True)
    if isinstance(result, tuple):
        print(f"Prediction: {result[0]}, Probability: {result[1]:.2f}")
    else:
        print(result)
