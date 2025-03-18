import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model("best_lstm_model.keras")

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

# Prepare input data for prediction
def prepare_input_data(weather_data, discharge_data):
    combined_df = pd.merge(weather_data, discharge_data, on="date", how="inner")
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None)
    
    features = ['precipitation_sum', 'rain_sum', 'precipitation_hours', 'river_discharge']
    for feature in features:
        if feature in combined_df:
            combined_df[feature] = pd.to_numeric(combined_df[feature], errors='coerce')
            combined_df[feature] = combined_df[feature].fillna(0)
    
    # Use only the last 60 days
    recent_data = combined_df[-60:].copy()
    input_data = recent_data[features].values 
    
    # Get dates for recent data
    recent_dates = recent_data['date'].tolist()

    return np.expand_dims(input_data, axis=0), recent_dates

# Predict next 5 days
def predict_weather(input_data, recent_dates):
    predictions = []
    new_dates = recent_dates.copy()
    
    for _ in range(5):
        pred = model.predict(input_data)
        predictions.append(pred[0])  # Extract prediction
        
        new_row = [float(pred[0][i]) for i in range(4)]
        new_date = new_dates[-1] + timedelta(days=1)
        new_dates.append(new_date)
        
        # Update input window
        recent_data = np.vstack([input_data[0, 1:, :], new_row])
        input_data = np.expand_dims(recent_data, axis=0)
    
    return predictions, new_dates

# Streamlit UI
st.title("5-Day Weather Forecast")

# Define location and date range
latitude = 59.91   # coordinates of Oslo, Norway
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
    predictions, new_dates = predict_weather(input_data, dates)

    # Display predictions
    st.write("Predicted data for the next five days:")
    st.write("_____________________________________________________________________")
    for i, pred in enumerate(predictions, start=1):
        st.write(f"{new_dates[i].date()}: Precipitation Sum: {pred[0]:.2f}")
        st.write(f"{new_dates[i].date()}: Rain Sum: {pred[1]:.2f}")
        st.write(f"{new_dates[i].date()}: Precipitation Hours: {pred[2]:.2f}")
        st.write(f"{new_dates[i].date()}: River Discharge: {pred[3]:.2f}")
        st.write("_____________________________________________________________________")
