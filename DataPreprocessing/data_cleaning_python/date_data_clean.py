import pandas as pd

# Load the CSV file
file_path = 'date.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Add a new column with the day of the week
df['Day of Week'] = df['Date'].dt.day_name()

# Save the updated DataFrame to a new CSV file
output_path = 'processed_days_data.csv'  # Replace with your desired output file name
df.to_csv(output_path, index=False)

print(f"Updated file saved as {output_path}")