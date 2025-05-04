# Analyzing Factors Influencing Traveler Visits to Hong Kong

This project aims to identify key factors influencing tourist visits to Hong Kong using machine learning techniques. By analyzing economic, environmental, and social indicators, we provide insights to help the government formulate policies to boost tourism, which has declined in recent years.

## Dataset

The primary dataset is sourced from the [Census and Statistics Department](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic) and tracks daily inbound and outbound passenger counts. Supplementary data from the [Hong Kong Immigration Department](https://www.immd.gov.hk/opendata/hkt/transport/immigration_clearance/statistics_passenger_traffic_festival_periods.csv) and [international sources](https://www.imf.org/external/datamapper/NGDPD@WEO/OEMDC/ADVEC/WEOWORLD) incorporate economic and environmental variables.

The dataset includes:

- **Response Variable (Y)**: Daily visitor arrivals.
- **Features (X)**: Indicators such as GDP, exchange rates, seasonal indicators, inflation, price levels, crime rates, weather data, holidays, and event counts.

Data preprocessing involved cleaning, standardization, and creating dummy variables for categorical factors. The COVID-19 period (2020-2022) was excluded due to its significant impact on travel.

## Methodology

We employed supervised learning techniques, specifically linear regression and random forest regression, to predict daily visitor arrivals. The dataset was split into training (60%), testing (20%), and validation (20%) sets.

- **Linear Regression**: Models the relationship between features and the response variable.
- **Random Forest Regression**: Captures non-linear relationships and feature interactions.

Performance was evaluated using R-squared and Root Mean Squared Error (RMSE). The random forest model outperformed linear regression on training data but showed signs of overfitting in cross-validation.

## Results

The analysis identified holidays in source countries, crime rates, and weather conditions as significant predictors of tourist visits. These findings suggest that policies promoting safety and leveraging holiday periods could enhance tourism.

## Repository Structure

- **`DataPreprocessing/`**: Scripts and data for preprocessing.
  - `data_cleaning_python/`: Python scripts for cleaning individual data files (e.g., `crime_data_clean.py`).
  - `features_csv/`: Processed feature data in CSV format (e.g., `processed_crime_data.csv`).
  - `readyset_data_csv/`: Ready-to-use datasets (e.g., `train.csv`, `test.csv`).
  - `y_csv/`: Target variable data (e.g., `statistics_on_daily_passenger_traffic.csv`).

- **`figures/`**: Visualizations from the analysis (e.g., `feature_importances.png`).

- **`predict_model/`**: Predictive model files.
  - `best_model.pkl`: Serialized random forest model.
  - `find model.py`: Script to identify the best model.
  - `predict.py`: Script for predictions.
  - `train.csv` and `test.csv`: Training and testing datasets.

## Usage

To use this repository:

1. Clone it to your local machine.
2. Install required Python libraries:

```bash
pip install pandas scikit-learn numpy matplotlib
```

3. Explore `DataPreprocessing/` for data cleaning steps.
4. Run scripts in `predict_model/` for training and predictions.

To make predictions with the pre-trained model:

```bash
python predict_model/predict.py
```

This generates predictions saved to `predict.csv`.

## Figures

The `figures/` directory contains:

- **2017_2019_Travellers.png** and **2023_2025_Travellers.png**: Traveler numbers pre- and post-pandemic.
- **Performance_of_Four_Models.png**: Comparison of regression models.
- **feature_importances.png**: Feature importance in the random forest model.
- **mae_comparison.png**, **r2_comparison.png**, **rmse_comparison.png**: Performance metrics.
- **top_features_correlation.png**: Correlation of top features with arrivals.

## Future Work

Potential improvements include:

- Adding more indicators and interaction terms.
- Hyperparameter tuning for the random forest to reduce overfitting.
- Combining models for better accuracy.
- Developing a real-time prediction interface.

## Contributors

- Fung Sai Wa (1155194766)
- Kan Man Chung (1155181978)
- Leung Shing Yip (1155193525)

---

For more details, visit the [GitHub repository](https://github.com/tonykan1122/SEEM3650_Project.git).