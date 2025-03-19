import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model("predictions.keras")

# Streamlit UI
st.title("5-Day Weather Forecast")

# User input for location
latitude = st.number_input("Enter Latitude:", value=22)
longitude = st.number_input("Enter Longitude:", value=79)

yesterday = datetime.today().date() - timedelta(days=1)
start_date = yesterday - timedelta(days=60)

st.write(f"Fetching data from {start_date} to {yesterday}")

@st.cache_data
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

@st.cache_data
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

def prepare_input_data(weather_data, discharge_data):
    combined_df = pd.merge(weather_data, discharge_data, on="date", how="inner")
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None)

    features = ['precipitation_sum', 'rain_sum', 'precipitation_hours', 'river_discharge']
    for feature in features:
        if feature in combined_df:
            combined_df[feature] = pd.to_numeric(combined_df[feature], errors='coerce')
            combined_df[feature] = combined_df[feature].fillna(0)

    recent_data = combined_df[-1:].copy()
    input_data = recent_data[features].values
    recent_dates = recent_data['date'].tolist()
    return np.expand_dims(input_data, axis=0), recent_dates

def predict_weather(input_data, recent_dates):
    predictions = []
    new_dates = recent_dates.copy()

    for _ in range(5):
        pred = model.predict(input_data)
        pred = np.maximum(pred, 0)
        predictions.append(pred[0])

        new_row = [float(pred[0][i]) for i in range(4)]
        new_date = new_dates[-1] + timedelta(days=1)
        new_dates.append(new_date)

        recent_data = np.vstack([input_data[0, 1:, :], new_row])
        input_data = np.expand_dims(recent_data, axis=0)
    return predictions, new_dates

weather_data = fetch_weather_data(latitude, longitude, start_date, yesterday)
discharge_data = fetch_discharge_data(latitude, longitude, start_date, yesterday)

if weather_data is not None and discharge_data is not None:
    input_data, dates = prepare_input_data(weather_data, discharge_data)
    predictions, new_dates = predict_weather(input_data, dates)

    st.write("### 📊 Predicted Weather Data for the Next 5 Days:")
    st.write("_________________________________________")
    flood_predictions = []

    for i, (date, pred) in enumerate(zip(new_dates[1:], predictions), start=1):
        flood = (
            (pred[0] >= 5.0 and pred[1] >= 10.0) or
            (pred[2] >= 15.0) or
            (pred[3] >= 12.0)
        )
        flood_predictions.append(flood)

        st.write(f"📅 **Date:** {date.date()}")
        st.write(f"🌧️ **Precipitation (mm):** {pred[0]:.2f}")
        st.write(f"☔ **Rain (mm):** {pred[1]:.2f}")
        st.write(f"🕒 **Precipitation Hours:** {pred[2]:.2f}")
        st.write(f"🌊 **River Discharge (m³/s):** {pred[3]:.2f}")
        st.write(f"🚨 **Flood Risk:** {'Yes' if flood else 'No'}")
        st.write("_________________________________________")

    # DataFrame for Display
    df_predictions = pd.DataFrame({
        "Date": [d.date() for d in new_dates[1:]],
        "Precipitation (mm)": [p[0] for p in predictions],
        "Rain (mm)": [p[1] for p in predictions],
        "Precipitation Hours": [p[2] for p in predictions],
        "River Discharge (m³/s)": [p[3] for p in predictions],
        "Flood Risk": flood_predictions
    })
    
    # Display Table
    st.write("### 📋 Tabular Data View:")
    st.dataframe(df_predictions)

    # Line Chart Visualization
    st.write("### 📈 Data Trends Over 5 Days:")
    st.line_chart(df_predictions.set_index("Date")[['Precipitation (mm)', 'Rain (mm)', 'River Discharge (m³/s)']])