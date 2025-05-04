import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the best model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

print("Loaded the best model from 'best_model.pkl'")

# Load test data
test_file = 'test.csv'
df_test = pd.read_csv(test_file)

# Extract features and target
X_test = df_test.drop('y', axis=1)
y_test = df_test['y']

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate metrics on test set
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print(f"\nTest set metrics:")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R²: {test_r2:.4f}")

# Create prediction dataframe with differences
predictions_df = df_test.copy()
predictions_df['predicted_y'] = y_pred
predictions_df['difference'] = predictions_df['y'] - predictions_df['predicted_y']

# Add percentage difference column
predictions_df['percentage_diff'] = (predictions_df['difference'] / predictions_df['y']) * 100

# Save the predictions to a CSV file
predictions_df.to_csv('predict.csv', index=False)
print("\nPredictions saved to 'predict.csv' with actual values, predicted values, differences, and percentage differences")