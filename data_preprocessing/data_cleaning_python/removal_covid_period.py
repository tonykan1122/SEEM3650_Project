import pandas as pd

# Load the CSV file
df = pd.read_csv('realy.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Define the date range to remove
start_date = pd.to_datetime('23/01/2020', format='%d/%m/%Y')
end_date = pd.to_datetime('15/03/2023', format='%d/%m/%Y')

# Filter out rows between start_date and end_date
df_filtered = df[(df['Date'] < start_date) | (df['Date'] > end_date)]

# Save the filtered DataFrame back to a CSV
df_filtered.to_csv('realy_filtered.csv', index=False)

print("Rows between 23/01/2020 and 15/03/2023 have been removed. Saved to 'realy_filtered.csv'.")