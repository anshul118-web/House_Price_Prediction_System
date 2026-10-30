
# House Price Prediction
# File 3: Data Preprocessing

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Step 1: Load dataset
print("Loading Dataset...")

df = pd.read_csv("dataset/house_data.csv")

# Step 2: Check dataset shape
print("\nOriginal Dataset Shape:")
print(df.shape)

# Step 3: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Remove duplicate rows
df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df.shape)

# Step 5: Handle missing values
# Fill missing numerical values with median

for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Step 6: Separate features and target

X = df.drop("Price", axis=1)

y = df["Price"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Step 7: Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 8: Display final shapes

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 9: Save processed data

X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)

y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("\nData Preprocessing Completed Successfully!")
