
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# Website heading
st.title("🏠 House Price Prediction")
st.write("Enter your house details to predict its estimated price.")

st.divider()

# Input fields
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input(
        "Median Income",
        min_value=0.1,
        value=5.0
    )

    HouseAge = st.number_input(
        "House Age",
        min_value=1.0,
        value=20.0
    )

    AveRooms = st.number_input(
        "Average Rooms",
        min_value=0.1,
        value=5.5
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.1,
        value=1.0
    )

with col2:
    Population = st.number_input(
        "Population",
        min_value=1.0,
        value=1000.0
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        min_value=0.1,
        value=3.0
    )

    Latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    Longitude = st.number_input(
        "Longitude",
        value=-118.0
    )

# Prediction button
if st.button("Predict House Price", type="primary"):

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

    # Predict
    prediction = model.predict(new_house)[0]

    # Convert into dollars
    price = prediction * 100000

    # Display result
    st.success("Prediction Completed!")

    st.metric(
        label="Estimated House Price",
        value=f"${price:,.2f}"
    )

    st.info(
        "This is an estimated price based on the "
        "California Housing dataset."
    )

st.divider()

st.caption("Machine Learning Project | Linear Regression")
