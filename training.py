# Import necessary libraries for training
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('conv.csv')

# Features and target variable
X = df.drop('Flare Time Period (days)', axis=1)
y = df['Flare Time Period (days)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Get predictions and probabilities
y_pred = rf_classifier.predict(X_test)

# Since predict_proba gives probabilities for each class, handle multi-class case for AUC
if len(set(y_test)) == 2:  # Binary classification
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Probability of positive class
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
else:  # Multi-class classification
    y_pred_proba = rf_classifier.predict_proba(X_test)
    print("ROC AUC Score (One-vs-Rest):", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

# Evaluate the model using classification report
print(classification_report(y_test, y_pred))

# Cross-validation
cv_results = cross_validate(rf_classifier, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

# Print results of each fold
print("Cross-validation results:")
print("Accuracy:", cv_results['test_accuracy'])
print("Precision:", cv_results['test_precision_macro'])
print("Recall:", cv_results['test_recall_macro'])
print("F1 Score:", cv_results['test_f1_macro'])

# Print the average of each metric across all folds
print("\nAverage Accuracy:", cv_results['test_accuracy'].mean())
print("Average Precision:", cv_results['test_precision_macro'].mean())
print("Average Recall:", cv_results['test_recall_macro'].mean())
print("Average F1 Score:", cv_results['test_f1_macro'].mean())

# Save the trained model
joblib.dump(rf_classifier, 'rf_flare_predictor.pkl')
