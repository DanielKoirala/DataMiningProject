from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from preprocess import load_and_preprocess_data, extract_features
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
data = load_and_preprocess_data("./data/spam_dataset.csv")

# Extract TF-IDF features with n-grams
ngram_range = (1, 2)  # Unigrams and bigrams
max_features = 5000   # Maximum number of features for TF-IDF
X, vectorizer = extract_features(data, ngram_range=ngram_range, max_features=max_features)
y = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load base models
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train base models
nb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save individual models
joblib.dump(nb_model, "./models/nb_model.pkl")
joblib.dump(rf_model, "./models/rf_model.pkl")

# Create an ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('rf', rf_model)
], voting='soft')

# Train ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate ensemble model
y_pred = ensemble_model.predict(X_test)
print("\nEnsemble Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Save ensemble model
joblib.dump(ensemble_model, "./models/ensemble_model.pkl")
print("\nEnsemble model saved successfully!")
