# Python & Pandas Advanced Data Engineering Commands

Este repositório contém comandos avançados de Python e Pandas úteis para tarefas de engenharia de dados. Estes comandos cobrem importação, manipulação, limpeza e otimização de dados.

## Table of Contents
- [Import and Configuration](#import-and-configuration)
- [Data Reading](#data-reading)
- [Data Manipulation](#data-manipulation)
- [Data Cleaning](#data-cleaning)
- [Date and Time Operations](#date-and-time-operations)
- [Optimization and Performance](#optimization-and-performance)
- [Saving Data](#saving-data)

## Import and Configuration
```python
import pandas as pd
import numpy as np

# Display all columns
pd.set_option('display.max_columns', None)

# Data Reading
# Read a large CSV file in chunks
chunk_size = 100000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)

# Read Parquet files
df = pd.read_parquet('data.parquet')

# Read JSON files
df = pd.read_json('data.json', lines=True)

# Data Manipulation
# Apply complex functions with apply
df['new_column'] = df['existing_column'].apply(lambda x: complex_function(x))

# Transform columns with map and lambdas
df['new_column'] = df['existing_column'].map(lambda x: transformation_function(x))

# Group operations and aggregations
grouped = df.groupby('group_column').agg({
    'col1': 'sum',
    'col2': 'mean',
    'col3': lambda x: x.unique().tolist()
})

# Pivot and unpivot
pivot_df = df.pivot_table(index='index_column', columns='columns_column', values='values_column', aggfunc='sum')
melted_df = pd.melt(df, id_vars=['id_column'], value_vars=['value_column1', 'value_column2'])

# Operations with multiple DataFrames
merged_df = pd.merge(df1, df2, on='common_column', how='inner')

# Complex transformations with groupby and apply
def custom_agg(group):
    return pd.Series({
        'metric1': group['column1'].sum(),
        'metric2': group['column2'].mean(),
    })

agg_df = df.groupby('group_column').apply(custom_agg)

# Data Cleaning
# Handle missing values
df['column'].fillna('default_value', inplace=True)
df.dropna(subset=['important_column'], inplace=True)

# Remove duplicates
df.drop_duplicates(subset=['key_column'], keep='first', inplace=True)

# Remove outliers using IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['column'] < (Q1 - 1.5 * IQR)) | (df['column'] > (Q3 + 1.5 * IQR)))]

# Date and Time Operations
# Convert columns to datetime
df['date_column'] = pd.to_datetime(df['date_column'])

# Extract date components
df['year'] = df['date_column'].dt.year
df['month'] = df['date_column'].dt.month
df['day'] = df['date_column'].dt.day

# Filter by date range
filtered_df = df[(df['date_column'] >= '2023-01-01') & (df['date_column'] <= '2023-12-31')]

# Optimization and Performance
# Use optimized data types
df['int_column'] = df['int_column'].astype('int32')
df['float_column'] = df['float_column'].astype('float32')

# Vectorized operations instead of loops
df['new_column'] = df['column1'] + df['column2']

# Use Dask for large datasets
import dask.dataframe as dd
ddf = dd.read_csv('large_file.csv')
result = ddf.groupby('group_column').agg({'column': 'mean'}).compute()

# Saving Data
# Save to CSV
df.to_csv('output.csv', index=False)

# Save to Parquet
df.to_parquet('output.parquet', index=False)

# Save to Excel
df.to_excel('output.xlsx', index=False, sheet_name='Sheet1')
