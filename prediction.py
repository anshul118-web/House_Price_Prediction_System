
# House Price Prediction
# File 6: Prediction

import pandas as pd
import joblib

print("===== HOUSE PRICE PREDICTION SYSTEM =====")

# Step 1: Load trained model
model = joblib.load("house_price_model.pkl")

print("\nModel Loaded Successfully!")

# Step 2: Take house details from user

print("\nEnter House Details:")

MedInc = float(input("Median Income: "))

HouseAge = float(input("House Age: "))

AveRooms = float(input("Average Rooms: "))

AveBedrms = float(input("Average Bedrooms: "))

Population = float(input("Population: "))

AveOccup = float(input("Average Occupancy: "))

Latitude = float(input("Latitude: "))

Longitude = float(input("Longitude: "))

# Step 3: Create DataFrame

new_house = pd.DataFrame([{
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
}])

# Step 4: Predict house price

prediction = model.predict(new_house)

# Step 5: Convert price into dollars

price = prediction[0] * 100000

# Step 6: Display result

print("\n===== PREDICTION RESULT =====")

print("Estimated House Price: $", round(price, 2))

print("\nPrediction Completed Successfully!")
