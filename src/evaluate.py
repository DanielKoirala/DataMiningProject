import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_data, extract_features

# Load models and vectorizer
nb_model = joblib.load("./models/nb_model.pkl")
rf_model = joblib.load("./models/rf_model.pkl")
vectorizer = joblib.load("./models/vectorizer.pkl")

# Load and preprocess data
data = load_and_preprocess_data("./data/spam_dataset.csv")

# Split into train and test sets for evaluation
X_full, _ = extract_features(data)
y_full = data['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

# Evaluate each model
for model, name in [(nb_model, "Naive Bayes"), (rf_model, "Random Forest")]:
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
