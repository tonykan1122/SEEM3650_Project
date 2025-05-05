import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Input data
data = {
    'Year': [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021,
             2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4',
                'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4'],
    'GDP': [702601, 688922, 714029, 739470, 654083, 627597, 687240, 706873, 705090, 678833,
            730502, 753548, 684815, 674632, 717535, 731987, 718986, 704211, 766343, 794051,
            766387, 757073, 817028, 836505],
    'Deflator': [95.9, 96.9, 97.7, 97.7, 98.5, 97.4, 97.9, 96.9, 98.1, 98.0, 98.6, 98.6,
                 99.2, 98.8, 101.8, 100.1, 101.4, 101.5, 104.4, 104.1, 105.2, 105.9, 109.2, 107.1]
}

df = pd.DataFrame(data)

# Normalize deflator for 2022 base year (average deflator for 2022 = 100)
deflator_2022 = df[df['Year'] == 2022]['Deflator'].mean()  # Average: 99.975
df['gdp_deflator_index'] = (df['Deflator'] / deflator_2022) * 100

# Generate date range
start_date = datetime(2019, 4, 4)
end_date = datetime(2025, 4, 22)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]


# Function to map date to quarter
def get_quarter(date):
    month = date.month
    if 1 <= month <= 3:
        return 'Q1'
    elif 4 <= month <= 6:
        return 'Q2'
    elif 7 <= month <= 9:
        return 'Q3'
    else:
        return 'Q4'


# Create output DataFrame
output_data = []
for date in date_range:
    year = date.year
    quarter = get_quarter(date)

    # Find matching row in input data
    match = df[(df['Year'] == year) & (df['Quarter'] == quarter)]

    if not match.empty:
        gdp = match['GDP'].iloc[0]
        deflator_index = match['gdp_deflator_index'].iloc[0]
    else:
        # No data for 2025
        gdp = np.nan
        deflator_index = np.nan

    output_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'gdp($)': gdp,
        'gdp_deflator_index': deflator_index
    })

# Create and save output DataFrame
output_df = pd.DataFrame(output_data)
output_df.to_csv('processed_gdp_data.csv', index=False)

# Optional: Print first few rows for verification
print(output_df.head())