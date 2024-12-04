from sklearn.ensemble import VotingClassifier
import joblib

from train import X_train, y_train, X_test, y_test, nb_model, rf_model

# Ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('rf', rf_model)
], voting='hard')

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

# Save ensemble model
joblib.dump(ensemble_model, "./models/ensemble_model.pkl")

# Evaluate ensemble model
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred))
print("Ensemble Model Precision:", precision_score(y_test, y_pred))
print("Ensemble Model Recall:", recall_score(y_test, y_pred))
