# Analyzing Factors Influencing Traveler Visits to Hong Kong

This project utilizes machine learning, specifically Random Forest regression, to identify key factors influencing tourist visits to Hong Kong. By analyzing economic, environmental, and social indicators, we aim to provide actionable insights for policymakers to boost tourism, which has declined in recent years.

## Project Overview

The study examines factors affecting daily visitor arrivals to Hong Kong from 2019 to 2025, excluding the COVID-19 period (2020-2022). Using a Random Forest model, we predict arrivals based on features like economic conditions, weather, holidays, and crime rates. Key findings indicate that holidays in source countries, crime rates, and weather significantly influence tourism, suggesting policies focused on safety and holiday promotions could enhance visitor numbers.

## Dataset

Data is sourced from platforms such as the Census and Statistics Department, Hong Kong Immigration Department, and international databases. The dataset includes:

- **Response Variable (Y)**: Daily visitor arrivals.
- **Features (X)**: 40 attributes spanning economic (e.g., GDP, exchange rates), environmental (e.g., temperature, rainfall), social (e.g., crime rates), and cultural (e.g., holidays, events) factors.

### Data Preprocessing

- **Cleaning**: Excluded missing values and the COVID-19 period (2020-2022).
- **Standardization**: Applied to continuous variables for outlier management.
- **Dummy Variables**: Created for categorical variables.
- **Feature Selection**: Reduced 133 attributes to 31 highly correlated features (e.g., "days without holiday," "Labor Day," "HK crime rate") using correlation analysis.

The processed dataset contains 819 records with 31 features.

## Methodology

We evaluated four regression models—Linear Regression, Ridge Regression, LASSO Regression, and Random Forest Regression—using an 80/20 train-test split.

- **Models Evaluated**:
  - Linear Regression
  - Ridge Regression
  - LASSO Regression
  - Random Forest Regression

- **Performance Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)

Random Forest outperformed others with a training MAE of 8,146.33, RMSE of 11,813.83, and R² of 0.8626, though cross-validation (MAE: 16,389.61) suggests potential overfitting.

## Results

The Random Forest model was selected for its strong performance. Key insights include:

- **Significant Predictors**: Holidays in source countries, crime rates, and weather.
- **Test Set Performance**:
  - Mean Percentage Difference: 6.4%
  - Standard Deviation: 5.8%
  - Most errors near 0%, indicating low bias.

### Visualizations

- **Feature Importances**: Identifies top influential features.
- **Prediction Error Distribution**: Shows small error concentration.
- **Correlation Matrix**: Highlights feature-arrival relationships.

## Repository Structure

The repository is organized as follows:

- **data_preprocessing/**: Scripts and data for preprocessing.
  - **data_cleaning_python/**: Python scripts for cleaning (e.g., `crime_data_clean.py`).
  - **features_csv/**: Processed feature files (e.g., `processed_crime_data.csv`).
  - **readyset_data_csv/**: Final datasets (e.g., `train.csv`, `test.csv`).
  - **y_csv/**: Target data (e.g., `statistics_on_daily_passenger_traffic.csv`).
  - **feature_selected_dataset.csv**: Post-selection dataset.
  - **original_dataset.csv**: Cleaned original dataset.

- **predict_model/**: Model-related files.
  - `find_model.py`: Trains and saves the best model (`best_model.pkl`).
  - `predict.py`: Generates predictions (`predict.csv`).
  - `train.csv` & `test.csv`: Training and testing datasets.

- **figures/**: Visualizations (e.g., `feature_importances.png`).
- **the_result/**: Outputs.
  - `best_model.pkl`: Trained Random Forest model.
  - `predict.csv`: Test set predictions.

## Usage

To use this repository:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tonykan1122/SEEM3650_Project.git
   ```
2. **Install dependencies**:
   ```bash
   pip install pandas scikit-learn==1.4.1.post1 numpy matplotlib seaborn
   ```
3. **Train the model**:
   ```bash
   python predict_model/find_model.py
   ```
   - Outputs `best_model.pkl` in `the_result/`.
4. **Generate predictions**:
   ```bash
   python predict_model/predict.py
   ```
   - Saves predictions to `the_result/predict.csv`.

For a step-by-step guide, watch the [YouTube tutorial](https://www.youtube.com/watch?v=your-video-id).

## Pros, Cons, and Potential Improvements

### Pros

- High accuracy due to Random Forest’s non-linear modeling.
- Clear feature importance insights.
- Robust to outliers and diverse scales.

### Cons

- Potential overfitting (training vs. cross-validation MAE gap).
- Higher computational and storage demands.

### Potential Improvements

- **Feature Engineering**: Add interaction terms (e.g., temperature-humidity).
- **Hyperparameter Tuning**: Optimize Random Forest to reduce overfitting.
- **Model Ensemble**: Combine strengths of multiple models.
- **Real-Time Data**: Integrate web scraping for current data.

## Societal Impact

- **Travelers**: Predict peak times to avoid crowds.
- **Government**: Inform policies targeting crime and holidays.
- **Businesses**: Optimize promotions for high-visitor periods.

Over-reliance on the model without critical evaluation could lead to economic risks.

## Future Development

- **Expand Attributes**: Include more events and refine features.
- **User Interface**: Develop a platform for easier model access.
- **Automation**: Add real-time data scraping.
- **Global Adaptation**: Extend the model to other regions.

## Contributors

- Fung Sai Wa (1155194766)
- Kan Man Chung (1155181978)
- Leung Shing Yip (1155193525)

For more details, visit the [GitHub repository](https://github.com/tonykan1122/SEEM3650_Project.git).