import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the training data
train_file = 'train.csv'
df_train = pd.read_csv(train_file)

# Extract features and target
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']

print("Training dataset shape: X =", X_train.shape, "y =", y_train.shape)

# Function to evaluate a model using K-fold cross-validation
def evaluate_model(model, X, y, cv=5):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    return -np.mean(scores)  # Convert from negative MAE to positive

# Dictionary to store all models and their metrics
all_models = {}

# 1. Linear Regression
print("\nEvaluating Linear Regression...")
lr = LinearRegression()
lr_mae_cv = evaluate_model(lr, X_train, y_train)
lr.fit(X_train, y_train)
lr_train_pred = lr.predict(X_train)
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_train_r2 = r2_score(y_train, lr_train_pred)

all_models['Linear Regression'] = {
    'model': lr,
    'MAE': lr_train_mae,
    'RMSE': lr_train_rmse,
    'R2': lr_train_r2
}

print(f"Linear Regression Cross-Val MAE: {lr_mae_cv:.2f}")
print(f"Linear Regression Training - MAE: {lr_train_mae:.2f}, RMSE: {lr_train_rmse:.2f}, R²: {lr_train_r2:.4f}")

# 2. Ridge Regression with hyperparameter tuning
print("\nTuning Ridge Regression...")
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 500.0]}
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_absolute_error')
ridge_grid.fit(X_train, y_train)
best_ridge = ridge_grid.best_estimator_
ridge_train_pred = best_ridge.predict(X_train)
ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)
ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
ridge_train_r2 = r2_score(y_train, ridge_train_pred)

all_models['Ridge Regression'] = {
    'model': best_ridge,
    'MAE': ridge_train_mae,
    'RMSE': ridge_train_rmse,
    'R2': ridge_train_r2,
    'best_params': ridge_grid.best_params_
}

print(f"Best Ridge alpha: {ridge_grid.best_params_['alpha']}")
print(f"Ridge Regression Cross-Val MAE: {-ridge_grid.best_score_:.2f}")
print(f"Ridge Regression Training - MAE: {ridge_train_mae:.2f}, RMSE: {ridge_train_rmse:.2f}, R²: {ridge_train_r2:.4f}")

# 3. Lasso Regression with hyperparameter tuning
print("\nTuning Lasso Regression...")
lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
lasso = Lasso(max_iter=10000)  # Increase max_iter to ensure convergence
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_absolute_error')
lasso_grid.fit(X_train, y_train)
best_lasso = lasso_grid.best_estimator_
lasso_train_pred = best_lasso.predict(X_train)
lasso_train_mae = mean_absolute_error(y_train, lasso_train_pred)
lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso_train_pred))
lasso_train_r2 = r2_score(y_train, lasso_train_pred)

all_models['Lasso Regression'] = {
    'model': best_lasso,
    'MAE': lasso_train_mae,
    'RMSE': lasso_train_rmse,
    'R2': lasso_train_r2,
    'best_params': lasso_grid.best_params_
}

print(f"Best Lasso alpha: {lasso_grid.best_params_['alpha']}")
print(f"Lasso Regression Cross-Val MAE: {-lasso_grid.best_score_:.2f}")
print(f"Lasso Regression Training - MAE: {lasso_train_mae:.2f}, RMSE: {lasso_train_rmse:.2f}, R²: {lasso_train_r2:.4f}")

# 4. Random Forest with hyperparameter tuning
print("\nTuning Random Forest Regressor...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_train_pred = best_rf.predict(X_train)
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_train_r2 = r2_score(y_train, rf_train_pred)

all_models['Random Forest'] = {
    'model': best_rf,
    'MAE': rf_train_mae,
    'RMSE': rf_train_rmse,
    'R2': rf_train_r2,
    'best_params': rf_grid.best_params_
}

print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Random Forest Cross-Val MAE: {-rf_grid.best_score_:.2f}")
print(f"Random Forest Training - MAE: {rf_train_mae:.2f}, RMSE: {rf_train_rmse:.2f}, R²: {rf_train_r2:.4f}")

# Find the best model (lowest MAE)
best_model_name = min(all_models.items(), key=lambda x: x[1]['MAE'])[0]
best_model_info = all_models[best_model_name]
print(f"\nBest model: {best_model_name}")
print(f"MAE: {best_model_info['MAE']:.2f}")
print(f"RMSE: {best_model_info['RMSE']:.2f}")
print(f"R²: {best_model_info['R2']:.4f}")
if 'best_params' in best_model_info:
    print(f"Best parameters: {best_model_info['best_params']}")

# Display feature importances for Random Forest if it's the best model
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    print("\nFeature importances plot saved as 'feature_importances.png'")

# Get top 10 feature names
top_10_features = feature_importance['Feature'][:10].tolist()

# Create a DataFrame with top features and target
correlation_data = X_train[top_10_features].copy()
correlation_data['target'] = y_train

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Top 10 Most Important Features', fontsize=16)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('top_features_correlation.png', dpi=300)
print("\nCorrelation matrix of top features saved as 'top_features_correlation.png'")


# Create metrics DataFrame for plotting
metrics_df = pd.DataFrame({
    'Model': list(all_models.keys()),
    'MAE': [model_info['MAE'] for model_info in all_models.values()],
    'RMSE': [model_info['RMSE'] for model_info in all_models.values()],
    'R2': [model_info['R2'] for model_info in all_models.values()]
})

# Save the best model
best_model = all_models[best_model_name]['model']
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model ({best_model_name}) saved as 'best_model.pkl'")


# 1. MAE Comparison
plt.figure(figsize=(10, 6))
plt.bar(metrics_df['Model'], metrics_df['MAE'], color='#1f77b4') 
plt.title('Mean Absolute Error (MAE) Comparison', fontsize=14)
plt.ylabel('Error Value', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Set y-axis limits for MAE to emphasize differences
min_mae = min(metrics_df['MAE']) * 0.85  # Lower limit to 85% of minimum
plt.ylim(min_mae, max(metrics_df['MAE']) * 1.05)  # Upper limit to 105% of maximum

# Add value labels on top of MAE bars
for i, v in enumerate(metrics_df['MAE']):
    plt.text(i, v + (max(metrics_df['MAE'])-min_mae)*0.01, f"{v:.1f}", 
             ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('mae_comparison.png', dpi=300)
print("\nMAE comparison plot saved as 'mae_comparison.png'")

# 2. RMSE Comparison
plt.figure(figsize=(10, 6))
plt.bar(metrics_df['Model'], metrics_df['RMSE'], color='#ff7f0e') 
plt.title('Root Mean Squared Error (RMSE) Comparison', fontsize=14)
plt.ylabel('Error Value', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Set y-axis limits for RMSE to emphasize differences
min_rmse = min(metrics_df['RMSE']) * 0.85  # Lower limit to 85% of minimum
plt.ylim(min_rmse, max(metrics_df['RMSE']) * 1.05)  # Upper limit to 105% of maximum

# Add value labels on top of RMSE bars
for i, v in enumerate(metrics_df['RMSE']):
    plt.text(i, v + (max(metrics_df['RMSE'])-min_rmse)*0.01, f"{v:.1f}", 
             ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300)
print("RMSE comparison plot saved as 'rmse_comparison.png'")

# 3. R² Score Comparison
plt.figure(figsize=(10, 6))
plt.bar(metrics_df['Model'], metrics_df['R2'], color='#2ca02c') 
plt.title('Model R² Score Comparison', fontsize=14)
plt.ylabel('R² Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Set y-axis limits to emphasize differences between models
min_r2 = min(metrics_df['R2']) * 0.95  # Set lower limit to 95% of minimum value
plt.ylim(min_r2, 1.0)  # Upper limit at perfect score (1.0)

# Add the actual R² values as text labels on top of bars
for i, v in enumerate(metrics_df['R2']):
    plt.text(i, v + 0.01, f"{v:.4f}", 
             ha='center', va='bottom', fontweight='bold')


plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('r2_comparison.png', dpi=300)
print("R² comparison plot saved as 'r2_comparison.png'")

# Save the best model
best_model = all_models[best_model_name]['model']
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model ({best_model_name}) saved as 'best_model.pkl'")




