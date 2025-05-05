import pandas as pd
from datetime import datetime

# Data from the provided table
data = [
    {"Year": 2019, "Month": "Apr", "Overall Crime": 4225},
    {"Year": 2019, "Month": "May", "Overall Crime": 4529},
    {"Year": 2019, "Month": "Jun", "Overall Crime": 3808},
    {"Year": 2019, "Month": "Jul", "Overall Crime": 4333},
    {"Year": 2019, "Month": "Aug", "Overall Crime": 4466},
    {"Year": 2019, "Month": "Sep", "Overall Crime": 4825},
    {"Year": 2019, "Month": "Oct", "Overall Crime": 6437},
    {"Year": 2019, "Month": "Nov", "Overall Crime": 6916},
    {"Year": 2019, "Month": "Dec", "Overall Crime": 7065},
    {"Year": 2020, "Month": "Jan", "Overall Crime": 5354},
    {"Year": 2023, "Month": "Mar", "Overall Crime": 7678},
    {"Year": 2023, "Month": "Apr", "Overall Crime": 7327},
    {"Year": 2023, "Month": "May", "Overall Crime": 7952},
    {"Year": 2023, "Month": "Jun", "Overall Crime": 7127},
    {"Year": 2023, "Month": "Jul", "Overall Crime": 7308},
    {"Year": 2023, "Month": "Aug", "Overall Crime": 8222},
    {"Year": 2023, "Month": "Sep", "Overall Crime": 8365},
    {"Year": 2023, "Month": "Oct", "Overall Crime": 8876},
    {"Year": 2023, "Month": "Nov", "Overall Crime": 7880},
    {"Year": 2023, "Month": "Dec", "Overall Crime": 7122},
    {"Year": 2024, "Month": "Jan", "Overall Crime": 7681},
    {"Year": 2024, "Month": "Feb", "Overall Crime": 5689},
    {"Year": 2024, "Month": "Mar", "Overall Crime": 8051},
    {"Year": 2024, "Month": "Apr", "Overall Crime": 7391},
    {"Year": 2024, "Month": "May", "Overall Crime": 8154},
    {"Year": 2024, "Month": "Jun", "Overall Crime": 8393},
    {"Year": 2024, "Month": "Jul", "Overall Crime": 8879},
    {"Year": 2024, "Month": "Aug", "Overall Crime": 8398},
    {"Year": 2024, "Month": "Sep", "Overall Crime": 7446},
    {"Year": 2024, "Month": "Oct", "Overall Crime": 8641},
    {"Year": 2024, "Month": "Nov", "Overall Crime": 8118},
    {"Year": 2024, "Month": "Dec", "Overall Crime": 8107},
]

# Create DataFrame
df = pd.DataFrame(data)

# Convert Month to numeric
month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
df["Month"] = df["Month"].map(month_map)

# Create a dictionary for crime data lookup
crime_dict = {(row["Year"], row["Month"]): row["Overall Crime"] for _, row in df.iterrows()}

# Generate daily dates
start_date = pd.to_datetime("2019-04-04")
end_date = pd.to_datetime("2025-04-22")
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# Create daily DataFrame
daily_data = []
for date in dates:
    year = date.year
    month = date.month
    # Get crime data if available, otherwise use "?"
    crime = crime_dict.get((year, month), "?")
    daily_data.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Month": month,
        "Overall Crime": crime
    })

# Create final DataFrame
daily_df = pd.DataFrame(daily_data)

# Save to CSV
daily_df.to_csv("processed_crime_data.csv", index=False)