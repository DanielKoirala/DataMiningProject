from flask import Flask, request, jsonify, render_template
import joblib
from preprocess import preprocess_text

app = Flask(__name__)

# Load models and vectorizer
ensemble_model = joblib.load("../models/ensemble_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# Default route with a UI form
@app.route('/')
def home():
    return render_template('home.html')

# Predict route for API and form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get email text from API request or form submission
    if request.is_json:
        email_text = request.json.get('text', '')
    else:
        email_text = request.form.get('email_text', '')

    # Check if input is provided
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    # Preprocess and predict
    cleaned_text = preprocess_text(email_text)
    features = vectorizer.transform([cleaned_text])
    prediction = ensemble_model.predict(features)
    is_spam = bool(prediction[0])

    # Handle response for form submissions
    if not request.is_json:
        result_message = "Spam" if is_spam else "Not Spam"
        return render_template('result.html', result=result_message)

    # Return JSON response for API calls
    return jsonify({'spam': is_spam})

if __name__ == '__main__':
    app.run(debug=True)
