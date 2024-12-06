from flask import Flask, request, jsonify, render_template
import joblib
from preprocess import preprocess_text

app = Flask(__name__)

# Load models and vectorizer
try:
    ensemble_model = joblib.load("./models/ensemble_model.pkl")
    vectorizer = joblib.load("./models/vectorizer.pkl")
except Exception as e:
    print(f"Error loading model/vectorizer: {str(e)}")

# Default route with a UI form
@app.route('/')
def home():
    return render_template('home.html')

# Unified prediction logic
def classify_text(email_text, threshold=0.1):
    """
    Preprocess the text, vectorize it, and make a prediction based on a probability threshold.
    Args:
        email_text (str): The input email or SMS text.
        threshold (float): The probability threshold for classifying as spam.
    Returns:
        str: "Spam" or "Not Spam"
    """
    cleaned_text = preprocess_text(email_text)
    features = vectorizer.transform([cleaned_text])
    probabilities = ensemble_model.predict_proba(features)
    spam_probability = probabilities[0][1]  # Probability of being spam
    return "Spam" if spam_probability >= threshold else "Not Spam"

# Predict route for API and form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get email text from API request or form submission
    email_text = None
    if request.is_json:
        email_text = request.json.get('text', '')
    elif request.form.get('sms_text'):
        email_text = request.form.get('sms_text', '')

    # Check if input is provided
    if not email_text:
        return jsonify({'error': 'No sms text provided'}), 400

    # Classify text
    try:
        result = classify_text(email_text, threshold=0.5)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Handle response for form submissions
    if not request.is_json:
        return render_template('result.html', result=result)

    # Return JSON response for API calls
    return jsonify({'spam': result == "Spam"})

if __name__ == '__main__':
    app.run(debug=True)
