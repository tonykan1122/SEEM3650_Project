import pandas as pd
from datetime import timedelta

# Set pandas option to avoid FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Step 1: Clean the Typhoon Dataset
def clean_typhoon_data(typhoon_file):
    # Load typhoon data, skipping the format descriptor row
    typhoon_df = pd.read_csv(typhoon_file, delimiter=',', skiprows=[1], skipinitialspace=True)

    # Strip whitespace from column names
    typhoon_df.columns = typhoon_df.columns.str.strip()

    # Debug: Print column names and first few rows
    print("Column names:", typhoon_df.columns.tolist())
    print("First 5 rows:\n", typhoon_df.head())

    # Check for required columns
    if 'Start Time' not in typhoon_df.columns or 'End Time' not in typhoon_df.columns:
        raise KeyError("Required columns 'Start Time' or 'End Time' not found in the CSV. Please check the file.")

    # Combine time and date columns
    # 'Start Time' has time (e.g., '16:15'), 'Unnamed: 4' has date (e.g., '2-Jul-19')
    typhoon_df['Start_Time'] = pd.to_datetime(
        typhoon_df['Start Time'] + ' ' + typhoon_df['Unnamed: 4'],
        format='%H:%M %d-%b-%y',
        errors='coerce'
    )
    typhoon_df['End_Time'] = pd.to_datetime(
        typhoon_df['End Time'] + ' ' + typhoon_df['Unnamed: 6'],
        format='%H:%M %d-%b-%y',
        errors='coerce'
    )

    # Debug: Print parsed datetimes
    print("Parsed Start_Time sample:\n", typhoon_df['Start_Time'].head())
    print("Parsed End_Time sample:\n", typhoon_df['End_Time'].head())

    # Drop rows with invalid dates
    pre_drop_len = len(typhoon_df)
    typhoon_df = typhoon_df.dropna(subset=['Start_Time', 'End_Time'])
    print(f"Dropped {pre_drop_len - len(typhoon_df)} rows due to invalid dates.")

    # Clean signal values
    def clean_signal(signal):
        if pd.isna(signal):
            return 0
        signal_str = str(signal)
        if '8' in signal_str:
            return 8
        try:
            return int(signal)
        except:
            return 0

    typhoon_df['Signal'] = typhoon_df['Signal'].apply(clean_signal)

    # Filter for relevant period (2019-04-04 to 2025-04-22)
    pre_filter_len = len(typhoon_df)
    typhoon_df = typhoon_df[(typhoon_df['Start_Time'] >= '2019-04-04') & (typhoon_df['Start_Time'] <= '2025-04-22')]
    print(f"Dropped {pre_filter_len - len(typhoon_df)} rows outside date range 2019-04-04 to 2025-04-22.")

    # Parse Duration (e.g., '13 25' -> 13.4167 hours)
    def parse_duration(duration):
        if pd.isna(duration):
            return 0
        try:
            hours, minutes = map(int, duration.split())
            return hours + minutes / 60
        except:
            return 0

    typhoon_df['Duration_Hours'] = typhoon_df['Duration'].apply(parse_duration)

    # Verify computed duration against End_Time - Start_Time
    typhoon_df['Computed_Duration'] = (typhoon_df['End_Time'] - typhoon_df['Start_Time']).dt.total_seconds() / 3600
    typhoon_df['Duration_Hours'] = typhoon_df['Duration_Hours'].where(
        typhoon_df['Duration_Hours'] > 0, typhoon_df['Computed_Duration']
    )

    return typhoon_df

# Step 2: Aggregate Typhoon Data to Daily Level
def aggregate_typhoon_daily(typhoon_df, start_date, end_date):
    daily_typhoon = []
    for _, row in typhoon_df.iterrows():
        start = row['Start_Time']
        end = row['End_Time']
        signal = row['Signal']
        current = start

        # Iterate through each day the typhoon spans
        while current <= end:
            day = current.date()
            # Ensure the day is within the desired date range
            if start_date.date() <= day <= end_date.date():
                # Check if signal was active in morning (00:00–12:00)
                morning_start = pd.Timestamp(day).replace(hour=0, minute=0)
                morning_end = pd.Timestamp(day).replace(hour=12, minute=0)
                is_morning = (start <= morning_end) and (end >= morning_start)

                # Calculate hours active on this day
                day_start = pd.Timestamp(day)
                day_end = day_start + timedelta(days=1)
                active_start = max(start, day_start)
                active_end = min(end, day_end)
                hours = (active_end - active_start).total_seconds() / 3600

                daily_typhoon.append({
                    'Date': day,
                    'Signal': signal,
                    'Hours': hours,
                    'Morning_Disruption': 1 if is_morning else 0
                })
            current += timedelta(days=1)

    # Create DataFrame and aggregate by date
    daily_typhoon_df = pd.DataFrame(daily_typhoon)
    if daily_typhoon_df.empty:
        daily_typhoon_agg = pd.DataFrame(columns=['Date', 'Max_Signal', 'Typhoon_Duration_Hours', 'Morning_Disruption', 'Typhoon_Present', 'High_Signal_Flag'])
    else:
        daily_typhoon_agg = daily_typhoon_df.groupby('Date').agg({
            'Signal': 'max',  # Max_Signal
            'Hours': 'sum',   # Typhoon_Duration_Hours
            'Morning_Disruption': 'max'  # Any morning disruption
        }).reset_index()

        # Add additional columns
        daily_typhoon_agg['Typhoon_Present'] = 1
        daily_typhoon_agg['High_Signal_Flag'] = (daily_typhoon_agg['Signal'] >= 8).astype(int)
        daily_typhoon_agg.rename(columns={'Signal': 'Max_Signal', 'Hours': 'Typhoon_Duration_Hours'}, inplace=True)

    # Create a full date range DataFrame
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    full_dates_df = pd.DataFrame({'Date': date_range})
    full_dates_df['Date'] = full_dates_df['Date'].dt.date

    # Merge with aggregated typhoon data
    daily_typhoon_full = full_dates_df.merge(daily_typhoon_agg, on='Date', how='left')

    # Fill missing values for days with no typhoon activity
    daily_typhoon_full['Typhoon_Present'] = daily_typhoon_full['Typhoon_Present'].fillna(0).astype(int)
    daily_typhoon_full['Max_Signal'] = daily_typhoon_full['Max_Signal'].fillna(0).astype(int)
    daily_typhoon_full['Typhoon_Duration_Hours'] = daily_typhoon_full['Typhoon_Duration_Hours'].fillna(0)
    daily_typhoon_full['Morning_Disruption'] = daily_typhoon_full['Morning_Disruption'].fillna(0).astype(int)
    daily_typhoon_full['High_Signal_Flag'] = daily_typhoon_full['High_Signal_Flag'].fillna(0).astype(int)

    return daily_typhoon_full

# Main function to run the pipeline
def main(typhoon_file, output_file):
    # Define date range
    start_date = pd.to_datetime('2019-04-04')
    end_date = pd.to_datetime('2025-04-22')

    # Clean typhoon data
    typhoon_df = clean_typhoon_data(typhoon_file)

    # Aggregate to daily level
    daily_typhoon_df = aggregate_typhoon_daily(typhoon_df, start_date, end_date)

    # Save the cleaned dataset
    daily_typhoon_df.to_csv(output_file, index=False)
    print(f"Cleaned typhoon dataset saved to {output_file}")

# Example usage
if __name__ == "__main__":
    typhoon_file = "typhoon.csv"
    output_file = "processed_typhoon_data.csv"
    main(typhoon_file, output_file)