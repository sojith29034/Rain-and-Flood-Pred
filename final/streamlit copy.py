import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model("predictions.keras")

# Function to fetch weather data
def fetch_weather_data(latitude, longitude, start_date, end_date):
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["precipitation_sum", "rain_sum", "precipitation_hours"],
        "timezone": "GMT"
    }
    response = requests.get(weather_url, params=weather_params)
    if response.status_code == 200:
        data = response.json().get('daily', {})
        if 'time' in data:
            data['date'] = pd.to_datetime(data['time'])
            del data['time']
        return pd.DataFrame(data)
    else:
        st.error("Failed to fetch weather data")
        return None

# Function to fetch river discharge data
def fetch_discharge_data(latitude, longitude, start_date, end_date):
    discharge_url = "https://flood-api.open-meteo.com/v1/flood"
    discharge_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["river_discharge"], 
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    response = requests.get(discharge_url, params=discharge_params)
    if response.status_code == 200:
        data = response.json().get('daily', {})
        if 'time' in data:
            data['date'] = pd.to_datetime(data['time'])
            del data['time']
        return pd.DataFrame(data)
    else:
        st.error("Failed to fetch river discharge data")
        return None

# Function to create flood probability based on thresholds
def calculate_flood_probability(df):
    flood_conditions = (
        (df["precipitation_sum"] > 20) | 
        (df["rain_sum"] > 15) | 
        (df["river_discharge"] > 500)
    )
    df["flood_probability"] = flood_conditions.astype(int)
    return df

# Prepare input data for prediction
def prepare_input_data(weather_data, discharge_data):
    combined_df = pd.merge(weather_data, discharge_data, on="date", how="inner")
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None)

    # Calculate flood probability
    combined_df = calculate_flood_probability(combined_df)

    # Ensure correct feature order
    features = ["precipitation_sum", "rain_sum", "precipitation_hours", "river_discharge", "flood_probability"]
    
    # Convert to numeric and handle missing values
    for feature in features:
        if feature in combined_df:
            combined_df[feature] = pd.to_numeric(combined_df[feature], errors="coerce")
            combined_df[feature] = combined_df[feature].fillna(0)

    # Apply log transformation for river discharge if needed
    combined_df["river_discharge"] = np.log1p(combined_df["river_discharge"])

    # Standardize input data
    mean = combined_df[features].mean()
    std = combined_df[features].std()
    std.replace(0, 1, inplace=True)  # Prevent division by zero
    scaled_data = (combined_df[features] - mean) / std

    # Select the last 60 days for model input
    recent_data = scaled_data[-60:].values
    if recent_data.shape[0] < 60:
        st.error("Insufficient data for prediction (need at least 60 days).")
        return None, None

    return np.expand_dims(recent_data, axis=0), combined_df["date"].tolist()

# Predict next 5 days
def predict_weather(input_data, recent_dates):
    predictions = []
    new_dates = recent_dates.copy()

    for _ in range(5):
        pred = model.predict(input_data)
        predictions.append(pred[0])  # Extract prediction

        # Reverse standardization using last known mean & std
        mean, std = input_data[0].mean(axis=0), input_data[0].std(axis=0)
        new_row = (pred[0] * std) + mean  # Re-scale

        # Convert back from log scale for river discharge
        new_row[3] = np.expm1(new_row[3])

        new_date = new_dates[-1] + timedelta(days=1)
        new_dates.append(new_date)

        # Update input window
        recent_data = np.vstack([input_data[0, 1:, :], new_row])
        input_data = np.expand_dims(recent_data, axis=0)

    return predictions, new_dates

# Streamlit UI
st.title("5-Day Weather & Flood Prediction")

# Define location and date range
latitude = 59.91   # Coordinates of Oslo, Norway
longitude = 10.75
yesterday = datetime.today().date() - timedelta(days=1)
start_date = yesterday - timedelta(days=60)

st.write(f"Start Date: {start_date}")
st.write(f"End Date: {yesterday}")

# Fetch data automatically
weather_data = fetch_weather_data(latitude, longitude, start_date, yesterday)
discharge_data = fetch_discharge_data(latitude, longitude, start_date, yesterday)

if weather_data is not None and discharge_data is not None:
    input_data, dates = prepare_input_data(weather_data, discharge_data)
    if input_data is not None:
        predictions, new_dates = predict_weather(input_data, dates)

        # Display predictions
        st.subheader("Predicted Weather for the Next 5 Days:")
        st.write("____________________________________________________")
        for i, pred in enumerate(predictions, start=1):
            st.write(f"📅 {new_dates[i].date()} 🌧️")
            st.write(f" - **Precipitation Sum:** {pred[0]:.2f} mm")
            st.write(f" - **Rain Sum:** {pred[1]:.2f} mm")
            st.write(f" - **Precipitation Hours:** {pred[2]:.2f} hours")
            st.write(f" - **River Discharge:** {pred[3]:.2f} m³/s")
            st.write(f" - **Flood Probability:** {pred[4]}")
            st.write("____________________________________________________")
