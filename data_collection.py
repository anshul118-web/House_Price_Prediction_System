
import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

# Dataset load
print("Loading House Price Dataset...")

housing = fetch_california_housing()

# Convert into DataFrame
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add target column
df["Price"] = housing.target

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Save CSV file
df.to_csv("dataset/house_data.csv", index=False)

print("Dataset successfully saved!")
print("Dataset Shape:", df.shape)

# Show first 5 rows
print(df.head())
