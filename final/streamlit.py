import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load the trained model
model = load_model("predictions.keras")  # Ensure the model file is in the working directory

# Streamlit UI Header
st.title("📊 5-Day Rainfall & River Discharge Forecast")
st.write("Enter location coordinates to get the latest predictions.")

# User Input: Latitude & Longitude
latitude = st.number_input("Enter Latitude", value=59.91, format="%.2f")
longitude = st.number_input("Enter Longitude", value=10.75, format="%.2f")

# Dates for fetching historical data
yesterday = datetime.today().date() - timedelta(days=1)
start_date = yesterday - timedelta(days=60)

st.write(f"🔄 Fetching data from **{start_date}** to **{yesterday}**...")

# Function to fetch historical weather data
def fetch_weather_data(lat, lon, start, end):
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": ["precipitation_sum", "rain_sum", "precipitation_hours"],
        "timezone": "GMT"
    }
    response = requests.get(weather_url, params=params)
    
    if response.status_code == 200:
        data = response.json().get("daily", {})
        if "time" in data:
            data["date"] = pd.to_datetime(data["time"])
            del data["time"]
        return pd.DataFrame(data)
    else:
        st.error(f"❌ Failed to fetch weather data: {response.status_code}")
        return None

# Function to fetch historical river discharge data
def fetch_discharge_data(lat, lon, start, end):
    discharge_url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["river_discharge"],
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }
    response = requests.get(discharge_url, params=params)
    
    if response.status_code == 200:
        data = response.json().get("daily", {})
        if "time" in data:
            data["date"] = pd.to_datetime(data["time"])
            del data["time"]
        return pd.DataFrame(data)
    else:
        st.error(f"❌ Failed to fetch river discharge data: {response.status_code}")
        return None

# Prepare input data for prediction
def prepare_input_data(weather_df, discharge_df):
    combined_df = pd.merge(weather_df, discharge_df, on="date", how="inner")
    combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.tz_localize(None)

    # Define model-relevant features
    features = ["precipitation_sum", "rain_sum", "precipitation_hours", "river_discharge"]
    
    # Add missing dummy features
    combined_df["dummy_feature_1"] = 0  
    combined_df["dummy_feature_2"] = 0  

    # Ensure all features exist and convert to numeric
    for feature in features + ["dummy_feature_1", "dummy_feature_2"]:
        combined_df[feature] = pd.to_numeric(combined_df[feature], errors="coerce").fillna(0)

    # Select the last 60 days
    if len(combined_df) < 60:
        st.error("❌ Not enough historical data. Ensure at least 60 days of data is available.")
        return None, None

    recent_data = combined_df.iloc[-60:].copy()

    # Prepare input data (shape: (1, 60, 6))
    input_data = recent_data[features + ["dummy_feature_1", "dummy_feature_2"]].values  
    input_data = np.expand_dims(input_data, axis=0)  # Shape (1, 60, 6)

    return input_data, recent_data["date"].tolist()

def compute_flood_label(predictions):
    """
    Compute flood label dynamically based on predicted values.
    """
    precipitation_threshold = 5.0
    rain_threshold = 10.0
    precipitation_hours_threshold = 15.0
    river_discharge_threshold = 12.0

    flood_labels = (
        ((predictions[:, :, 0] >= precipitation_threshold) & 
         (predictions[:, :, 1] >= rain_threshold)) |
        (predictions[:, :, 2] >= precipitation_hours_threshold) |
        (predictions[:, :, 3] >= river_discharge_threshold)
    )
    return flood_labels.astype(bool)

# Predict next 5 days
def predict_weather(input_data, recent_dates):
    predictions = []
    new_dates = recent_dates.copy()

    for _ in range(5):
        pred = model.predict(input_data)  # Expected shape: (1, 1, 4)
        pred_values = pred[0, 0].tolist()  # Extract predicted values

        # Ensure 6 features (add two dummy values)
        new_row = pred_values + [0, 0]  

        # Predict next date
        new_date = new_dates[-1] + timedelta(days=1)
        new_dates.append(new_date)

        # Shift input data for next prediction
        input_data = np.roll(input_data, shift=-1, axis=1)  # Shift window
        input_data[0, -1, :] = new_row  # Add new prediction

        predictions.append(pred_values)

    return predictions, new_dates

# Fetch data when user clicks
if st.button("🔍 Get Forecast"):
    weather_data = fetch_weather_data(latitude, longitude, start_date, yesterday)
    discharge_data = fetch_discharge_data(latitude, longitude, start_date, yesterday)

    if weather_data is not None and discharge_data is not None:
        input_data, dates = prepare_input_data(weather_data, discharge_data)
        if input_data is not None:
            predictions, flood_risks, future_dates = predict_weather(input_data, dates)

            # Convert predictions into a DataFrame
            pred_df = pd.DataFrame(predictions, columns=["Precipitation Sum", "Rain Sum", "Precipitation Hours", "River Discharge"])
            pred_df["Date"] = future_dates
            pred_df["Flood Risk"] = ["⚠️ High" if risk else "✅ Low" for risk in flood_risks]

            # Display results
            st.subheader("📅 Predicted Data for the Next 5 Days")
            st.dataframe(pred_df.style.format({"Precipitation Sum": "{:.2f}", "Rain Sum": "{:.2f}", "Precipitation Hours": "{:.2f}", "River Discharge": "{:.2f}"}))

            # Show Line Chart
            st.subheader("📈 Forecast Visualization")
            st.line_chart(pred_df.set_index("Date").drop(columns=["Flood Risk"]))
