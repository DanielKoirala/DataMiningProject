import joblib
from preprocess import preprocess_text

# Load saved models and vectorizer
ensemble_model = joblib.load("./models/ensemble_model.pkl")
vectorizer = joblib.load("./models/vectorizer.pkl")

# Predict function
def predict_email(text):
    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = ensemble_model.predict(features)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
text = "Congratulations! You've won a free gift card. Click here to claim."
print(predict_email(text))
