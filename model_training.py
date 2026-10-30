
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

print("Loading Training Data...")

X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv")

print("Training Data Loaded Successfully!")


model = LinearRegression()

print("\nCreating Linear Regression Model...")


model.fit(X_train, y_train.values.ravel())

print("Model Training Completed!")


print("\nModel Coefficients:")
print(model.coef_)

print("\nModel Intercept:")
print(model.intercept_)


joblib.dump(model, "house_price_model.pkl")

print("\nModel Saved Successfully!")

print("File Name: house_price_model.pkl")
