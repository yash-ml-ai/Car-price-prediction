import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# Load the saved model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "used_car_price_model.pkl"
)
model = joblib.load(model_path)

st.title("🚗 Used Car Price Predictor")
st.write("Enter the details of your car to estimate its selling price.")

with st.form("car_form"):
    name = st.text_input("Car name", "Maruti Swift Dzire VDI")
    car_age = st.number_input("Car age (years)", 0, 50, 6)
    km_driven = st.number_input("Kilometers driven", 0, 1000000, 50000, step=1000)

    fuel = st.selectbox("Fuel", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [
        "First Owner", "Second Owner", "Third Owner",
        "Fourth & Above Owner", "Test Drive Car"
    ])

    mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, value=23.0)
    engine = st.number_input("Engine (cc)", min_value=0.0, value=1248.0)
    max_power = st.number_input("Max power", min_value=0.0, value=74.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    new_car = pd.DataFrame([{
        "name": name,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage(km/ltr/kg)": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": float(seats),
        "car_age": car_age
    }])

    try:
        price = model.predict(new_car)[0]
        st.success(f"Estimated selling price: ₹{price:,.0f}")
        st.caption("This is an estimate, not a guaranteed market price.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
