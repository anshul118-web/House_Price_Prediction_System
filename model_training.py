
# House Price Prediction
# File 4: Model Training

import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

print("Loading Training Data...")

# Step 1: Load training data

X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv")

print("Training Data Loaded Successfully!")

# Step 2: Create Linear Regression Model

model = LinearRegression()

print("\nCreating Linear Regression Model...")

# Step 3: Train the Model

model.fit(X_train, y_train.values.ravel())

print("Model Training Completed!")

# Step 4: Display Model Coefficients

print("\nModel Coefficients:")
print(model.coef_)

print("\nModel Intercept:")
print(model.intercept_)

# Step 5: Save Trained Model

joblib.dump(model, "house_price_model.pkl")

print("\nModel Saved Successfully!")

print("File Name: house_price_model.pkl")
