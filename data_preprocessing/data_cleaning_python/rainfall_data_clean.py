import pandas as pd
from datetime import datetime

# Read data from CSV file
data_file = 'rainfall.csv'  # Update with correct path if needed

# Read CSV, skipping the first two metadata lines
df = pd.read_csv(data_file, skiprows=2, encoding='utf-8')

# Explicitly set column names to avoid bilingual header issues
df.columns = ['Year', 'Month', 'Day', 'Value', 'data Completeness']

# Debug: Print column names and first few rows
print("Column names:", df.columns.tolist())
print("\nFirst few rows of DataFrame:")
print(df.head())

# Replace 'Trace' with 0.049 in the Value column
df['Value'] = df['Value'].replace('Trace', 0.049)

# Convert columns to numeric, handling errors
for col in ['Year', 'Month', 'Day', 'Value']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with invalid dates or missing values
df = df.dropna(subset=['Year', 'Month', 'Day'])

# Create date column, handling invalid dates
df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
df = df.dropna(subset=['date'])  # Remove rows with invalid dates

# Debug: Check for data on or after 2019/04/04
cutoff_date = datetime(2019, 4, 4)
print("\nRows after 2019/04/04 before filtering:")
print(df[df['date'] >= cutoff_date][['Year', 'Month', 'Day', 'Value', 'data Completeness']].head())

# Filter data on or after 2019-04-04
df_filtered = df[df['date'] >= cutoff_date].copy()  # Explicit copy to avoid warnings

# Create date(without year) column, ensuring no decimals (e.g., 4/4)
df_filtered.loc[:, 'date(without year)'] = df_filtered.apply(
    lambda x: f"{int(x['Month'])}/{int(x['Day'])}", axis=1
)

# Select and rename columns
result = df_filtered[['date', 'date(without year)', 'Value', 'data Completeness']].copy()
result.columns = ['date', 'date(without year)', 'Rainfall Value', 'data Completeness']

# Replace missing values with '?'
result = result.fillna('?')

# Format date column to YYYY/MM/DD
result['date'] = result['date'].dt.strftime('%Y/%m/%d')

# Print result
print("\nProcessed Data:")
print(result)

# Save to CSV
result.to_csv('processed_rainfall_data.csv', index=False)