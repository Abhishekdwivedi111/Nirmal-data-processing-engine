"""
Script to create sample data for testing the Nirmal data processing engine.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Create sample dataset with various data quality issues
np.random.seed(42)

# Generate sample data
n_samples = 100

data = {
    'id': range(1, n_samples + 1),
    'name': [f'Person_{i}' for i in range(1, n_samples + 1)],
    'age': np.random.randint(18, 80, n_samples),
    'salary': np.random.randint(30000, 150000, n_samples),
    'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Marketing'], n_samples),
    'join_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
    'email': [f'user{i}@example.com' for i in range(1, n_samples + 1)]
}

df = pd.DataFrame(data)

# Introduce data quality issues for testing
# 1. Add duplicates
df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # Add 5 duplicate rows

# 2. Add missing values
df.loc[10:15, 'age'] = np.nan
df.loc[20:25, 'salary'] = np.nan
df.loc[30, 'department'] = np.nan

# 3. Add outliers
df.loc[40, 'age'] = 200  # Invalid age
df.loc[41, 'salary'] = 5000000  # Extreme salary
df.loc[42, 'salary'] = -50000  # Negative salary

# 4. Add inconsistent text
df.loc[50:52, 'name'] = ['  JOHN DOE  ', 'jane smith', 'Bob JOHNSON']

# Save to CSV
output_path = 'data/sample_input.csv'
df.to_csv(output_path, index=False)
print(f"Sample data created: {output_path}")
print(f"Dataset shape: {df.shape}")
print(f"\nData quality issues introduced:")
print(f"  - Duplicate rows: {df.duplicated().sum()}")
print(f"  - Missing values:\n{df.isnull().sum()}")
print(f"\nFirst few rows:")
print(df.head(10))
