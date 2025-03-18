import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model("lstm_weather_model.keras")

# Define features based on the trained model
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", 
            "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max"]

sequence_length = 15  # Past 15 days used for prediction

# Function to fetch weather data
def fetch_weather_data(latitude, longitude, start_date, end_date):
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": features,
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

# Prepare input data for prediction
def prepare_input_data(weather_data):
    weather_data['date'] = pd.to_datetime(weather_data['date']).dt.tz_localize(None)
    for feature in features:
        if feature in weather_data:
            weather_data[feature] = pd.to_numeric(weather_data[feature], errors='coerce')
            weather_data[feature] = weather_data[feature].fillna(0)
    
    # Use only the last 15 days
    recent_data = weather_data[-sequence_length:].copy()
    input_data = recent_data[features].values  # Extract feature values
    
    # Scale the input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    return np.expand_dims(input_data_scaled, axis=0), scaler

# Predict next 5 days
def predict_weather(input_data, scaler):
    predictions = []
    for _ in range(5):
        pred_scaled = model.predict(input_data)
        pred = scaler.inverse_transform(pred_scaled)[0]  # Convert back to original scale
        predictions.append(pred)
        
        # Update input sequence with new prediction
        recent_data = np.vstack([input_data[0, 1:, :], pred])
        input_data = np.expand_dims(recent_data, axis=0)
    
    return predictions

# Streamlit UI
st.title("5-Day Weather Forecast")

# Define location and date range
latitude = 52.52   # Mumbai, India
longitude = 13.41

yesterday = datetime.today().date() - timedelta(days=1)
start_date = yesterday - timedelta(days=sequence_length)

st.write(f"Fetching weather data from {start_date} to {yesterday}...")

weather_data = fetch_weather_data(latitude, longitude, start_date, yesterday)

if weather_data is not None:
    input_data, scaler = prepare_input_data(weather_data)
    predictions = predict_weather(input_data, scaler)
    
    # Display predictions
    st.write("Predicted Weather Data for the Next 5 Days:")
    st.write("_________________________________________________________________")
    for i, pred in enumerate(predictions, start=1):
        st.write(f"**Day {i}:**")
        for j, feature in enumerate(features):
            st.write(f"{feature.replace('_', ' ').title()}: {pred[j]:.2f}")
        st.write("_________________________________________________________________")