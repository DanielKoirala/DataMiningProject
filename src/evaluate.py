import joblib
from sklearn.metrics import classification_report

# Load saved models and vectorizer
nb_model = joblib.load("./models/nb_model.pkl")
rf_model = joblib.load("./models/rf_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# Load test data
from preprocess import load_and_preprocess_data, extract_features
data = load_and_preprocess_data("./data/spam_dataset.csv")
X, _ = extract_features(data)
y = data['label']

# Predictions and report
for model, name in [(nb_model, "Naive Bayes"), (rf_model, "Random Forest")]:
    y_pred = model.predict(X)
    print(f"{name} Report:")
    print(classification_report(y, y_pred))
