import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model("training_model.h5")

# Function to fetch weather data
def fetch_weather_data(latitude, longitude, start_date, end_date):
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max"],
        "timezone": "GMT"
    }
    response = requests.get(weather_url, params=weather_params)
    if response.status_code == 200:
        data = response.json().get('daily', {})
        if 'time' in data:
            data['date'] = pd.to_datetime(data['time'])  # Convert time to datetime
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
        "daily": "river_discharge",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    response = requests.get(discharge_url, params=discharge_params)
    if response.status_code == 200:
        data = response.json().get('daily', {})
        if 'time' in data:
            data['date'] = pd.to_datetime(data['time'])  # Convert time to datetime
            del data['time']
        return pd.DataFrame(data)
    else:
        st.error("Failed to fetch river discharge data")
        return None

# Prepare input data for prediction
def prepare_input_data(weather_data, discharge_data):
    # Merge on 'date' and keep the 'date' column in the combined dataframe
    combined_df = pd.merge(weather_data, discharge_data, on="date", how="inner")
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None)
    
    # Use the last 60 days of data to meet the model's input requirements
    recent_data = combined_df[-60:].copy()
    input_data = np.array([
        recent_data['temperature_2m_max'],
        recent_data['temperature_2m_min'],
        recent_data['precipitation_hours'],
        recent_data['wind_speed_10m_max'],
        recent_data['wind_gusts_10m_max'],
        recent_data['river_discharge']
    ])
    
    # Reshape to (1, timesteps, features) as expected by the model
    return np.expand_dims(input_data.T, axis=0), recent_data['date'].tolist()

# Predict the next 5 days
def predict_weather(input_data):
    predictions = []
    for _ in range(5):
        pred = model.predict(input_data)
        predictions.append(pred[0])
        
        # Roll data for the next prediction
        input_data = np.roll(input_data, shift=-1, axis=1)
        input_data[0, -1, :] = pred  # Update with new prediction

    return np.array(predictions)

# Streamlit UI
st.title("5-Day Weather Forecast using Open-Meteo Data")

# Define location and date range
latitude = 59.91   # Example coordinates, e.g., Oslo, Norway
longitude = 10.75
yesterday = pd.to_datetime("today").date() - timedelta(days=1)
start_date = yesterday - timedelta(days=60)

# Fetch data automatically
weather_data = fetch_weather_data(latitude, longitude, start_date, yesterday)
discharge_data = fetch_discharge_data(latitude, longitude, start_date, yesterday)

if weather_data is not None and discharge_data is not None:
    # Prepare the data for prediction
    input_data, dates = prepare_input_data(weather_data, discharge_data)
    
    # Predict next 5 days
    predictions = predict_weather(input_data)
    
    # Display predictions for each day with dates
    st.write("Predicted data for the next five days:")
    for i, pred in enumerate(predictions, start=1):
        prediction_date = dates[-1] + timedelta(days=i)
        
        # Assuming pred contains multiple outputs, such as [temperature, rain, river discharge]
        pred_temp, pred_rain, pred_discharge = pred[0], pred[1], pred[2]
        
        st.write(f"**{prediction_date.strftime('%Y-%m-%d')}**")
        st.write(f"Temperature: {pred_temp:.2f} °C")
        st.write(f"Rain: {pred_rain:.2f} mm")
        st.write(f"River Discharge: {pred_discharge:.2f} m³/s")
