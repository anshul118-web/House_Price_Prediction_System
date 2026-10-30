
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

print("Loading Dataset...")

df = pd.read_csv("dataset/house_data.csv")


print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df.shape)
for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
X = df.drop("Price", axis=1)

y = df["Price"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)

y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("\nData Preprocessing Completed Successfully!")
