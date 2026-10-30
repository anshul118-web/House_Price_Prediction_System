
# House Price Prediction
# File 2: Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Dataset load
df = pd.read_csv("dataset/house_data.csv")

print("\n===== DATASET INFORMATION =====")

# Step 2: First 5 rows
print(df.head())

# Step 3: Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Step 4: Dataset information
print("\nDataset Info:")
df.info()

# Step 5: Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Step 6: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 7: Price distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["Price"], kde=True)

plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.show()

# Step 8: Correlation heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.show()

print("\nData Analysis Completed!")
