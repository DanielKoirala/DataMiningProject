from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from preprocess import load_and_preprocess_data, extract_features

# Load and preprocess the new dataset
data = load_and_preprocess_data("./data/spam_dataset.csv")  # Update with actual path
X, vectorizer = extract_features(data)
y = data['label']  # Binary labels: 0 for ham, 1 for spam

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

nb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{name} Precision:", precision_score(y_test, y_pred))
    print(f"{name} Recall:", recall_score(y_test, y_pred))

evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Save models
joblib.dump(nb_model, "./models/nb_model.pkl")
joblib.dump(rf_model, "./models/rf_model.pkl")
joblib.dump(vectorizer, "./models/vectorizer.pkl")
