import pandas as pd

data_raw = pd.read_csv('../../data/text_data/calls911/calls911_month_full.txt', header=None)

# Transpose the table
df_transposed = data_raw.transpose()

# Set the first row as column names
df_transposed.columns = df_transposed.iloc[0]

# Drop the first row (original column names)
df_transposed = df_transposed[1:]

# Set the first column as 'date' and format the date
df_transposed.insert(0, 'date', pd.date_range(start='2015-12-01', end='2020-07-01', freq='MS').strftime('%Y-%m-%d'))

# Convert type of columns
columns_to_convert = df_transposed.columns[1:]
df_transposed[columns_to_convert] = df_transposed[columns_to_convert].astype(float)

df_transposed.to_csv('../../data/text_data/calls911/calls911_benchmarks.csv', index=False)