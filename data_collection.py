
import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

print("Loading House Price Dataset...")

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target


os.makedirs("dataset", exist_ok=True)


df.to_csv("dataset/house_data.csv", index=False)

print("Dataset successfully saved!")
print("Dataset Shape:", df.shape)

print(df.head())
