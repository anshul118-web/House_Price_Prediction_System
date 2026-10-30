
import pandas as pd
import joblib

print("===== HOUSE PRICE PREDICTION SYSTEM =====")

model = joblib.load("house_price_model.pkl")

print("\nModel Loaded Successfully!")


print("\nEnter House Details:")

MedInc = float(input("Median Income: "))

HouseAge = float(input("House Age: "))

AveRooms = float(input("Average Rooms: "))

AveBedrms = float(input("Average Bedrooms: "))

Population = float(input("Population: "))

AveOccup = float(input("Average Occupancy: "))

Latitude = float(input("Latitude: "))

Longitude = float(input("Longitude: "))



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

prediction = model.predict(new_house)

price = prediction[0] * 100000


print("\n===== PREDICTION RESULT =====")

print("Estimated House Price: $", round(price, 2))

print("\nPrediction Completed Successfully!")
