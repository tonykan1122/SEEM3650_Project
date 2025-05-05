import pandas as pd
from datetime import datetime

# Read oil.csv
try:
    df = pd.read_csv('oil.csv')
except FileNotFoundError:
    print("Error: oil.csv not found.")
    exit(1)
except Exception as e:
    print(f"Error reading oil.csv: {e}")
    exit(1)

# Define date parsing function
def parse_oil_dates(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    date_str = str(date_str).strip()
    parts = date_str.split('/')
    if len(parts) != 3:
        return pd.NaT
    if len(parts[2]) == 2:
        date_str = f"{parts[0]}/{parts[1]}/20{parts[2]}"
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
    except (ValueError, TypeError):
        return pd.NaT

# Parse dates
df['Date'] = df['Date'].apply(parse_oil_dates)

# Log unparseable dates before dropping
unparseable = df[df['Date'].isna()]
if not unparseable.empty:
    print(f"Warning: {len(unparseable)} dates could not be parsed or are missing. See unparseable_dates.csv.")
    unparseable.to_csv('unparseable_dates.csv', index=False)

# Calculate daily averages
def calculate_average(row):
    values = [row['Open'], row['Close'], row['High'], row['Low']]
    non_zero_values = [v for v in values if pd.notna(v) and v != 0]
    return sum(non_zero_values) / len(non_zero_values) if non_zero_values else None

df['average oil price'] = df.apply(calculate_average, axis=1)

# Create full date range (reverse order to match CSV)
start_date = datetime(2019, 4, 4)
end_date = datetime(2025, 4, 22)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
full_df = pd.DataFrame({'Date': date_range})

# Assign dates to rows in reverse chronological order
df = df.reset_index(drop=True)
valid_dates = df['Date'].dropna()
missing_count = df['Date'].isna().sum()

if missing_count > 0:
    print(f"Found {missing_count} rows with missing dates. Attempting to infer dates.")
    # Reverse date range to match CSV order (latest to earliest)
    reversed_dates = date_range[::-1]
    # Assign dates to rows, prioritizing rows with parsed dates
    assigned_dates = []
    date_idx = 0
    for idx, row in df.iterrows():
        if pd.isna(row['Date']):
            # Assign next available date from reversed range, skipping used dates
            while date_idx < len(reversed_dates) and reversed_dates[date_idx] in assigned_dates:
                date_idx += 1
            if date_idx < len(reversed_dates):
                assigned_dates.append(reversed_dates[date_idx])
                df.at[idx, 'Date'] = reversed_dates[date_idx]
            else:
                print("Ran out of dates to assign!")
                break
        else:
            assigned_dates.append(row['Date'])

# Drop rows where average couldn't be calculated
df = df[['Date', 'average oil price']].dropna(subset=['average oil price'])

# Read monthly data
try:
    df_month = pd.read_csv('oil_month.csv')
    df_month['Date'] = pd.to_datetime(df_month['Date'], format='%b-%Y')
    df_month['month_year'] = df_month['Date'].dt.to_period('M')
    month_price_map = dict(zip(df_month['month_year'], df_month['U.S. Crude Oil First Purchase Price (Dollars per Barrel)']))
except FileNotFoundError:
    print("Error: oil_month.csv not found.")
    exit(1)
except Exception as e:
    print(f"Error processing oil_month.csv: {e}")
    exit(1)

# Merge with full date range
full_df = full_df.merge(df, on='Date', how='left')

# Fill missing values with monthly averages
def fill_missing_price(row):
    if pd.isna(row['average oil price']):
        month_year = row['Date'].to_period('M')
        return month_price_map.get(month_year, None)
    return row['average oil price']

full_df['average oil price'] = full_df.apply(fill_missing_price, axis=1)

# Check for unfillable gaps
if full_df['average oil price'].isna().any():
    print("Warning: Some dates lack both daily and monthly data.")
    print(full_df[full_df['average oil price'].isna()][['Date']])

# Format and save
full_df['Date'] = full_df['Date'].dt.strftime('%m/%d/%Y')
full_df.to_csv('processed_oil_data.csv', index=False)

# Display samples
print(full_df.head(20))
print(full_df.tail(10))
print("\nCSV saved as 'processed_oil_data.csv'")