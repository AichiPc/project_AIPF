import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
import joblib

# Load your dataset
data = pd.read_csv('conv3.csv')

# Define feature columns and target column
X = data.drop(columns=['Days_Until_Next_Flare','Flare Time Period','Flare Frequency_1 per month','Flare Frequency_1 per year','Flare Frequency_2 per month','Flare Frequency_2 per year','Flare Frequency_3 per month','Flare Frequency_4 per year','Flare Frequency_5 per month'])  # Features (exclude the target column)
y = data['Days_Until_Next_Flare']  # Target (days until the next flare)

# Handle Data Imbalance with SMOTE (Optional)
# This balances the target variable, which may help in improving accuracy
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create the Gradient Boosting model (XGBoost)
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Cross-validation for better performance estimation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print(f"Average R-squared score from cross-validation: {cv_scores.mean()}")

# Save the final dataset and model predictions (optional)
#data['Predicted_Flare_Time'] = best_model.predict(X)
#data.to_csv('dataset_with_predictions.csv', index=False)

#print("Dataset with predictions saved as 'dataset_with_predictions.csv'.")

# Save the trained model
joblib.dump(best_model, 'flare_predictor.pkl')