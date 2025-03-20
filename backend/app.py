from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows Streamlit to communicate with this API

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
    return None

# Function to prepare input data
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

# Function to predict weather for 5 days
def predict_weather(input_data, recent_dates):
    predictions = []
    new_dates = recent_dates.copy()

    for _ in range(7):
        pred = model.predict(input_data)
        pred = np.maximum(pred, 0)
        predictions.append(pred[0])

        new_row = [float(pred[0][i]) for i in range(4)]
        new_date = new_dates[-1] + timedelta(days=1)
        new_dates.append(new_date)

        recent_data = np.vstack([input_data[0, 1:, :], new_row])
        input_data = np.expand_dims(recent_data, axis=0)

    return predictions, new_dates

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        latitude = data["latitude"]
        longitude = data["longitude"]

        yesterday = datetime.today().date() - timedelta(days=1)
        start_date = yesterday - timedelta(days=60)

        weather_data = fetch_weather_data(latitude, longitude, start_date, yesterday)
        discharge_data = fetch_discharge_data(latitude, longitude, start_date, yesterday)

        if weather_data is None or discharge_data is None:
            return jsonify({"error": "Failed to fetch weather or discharge data"}), 500

        input_data, dates = prepare_input_data(weather_data, discharge_data)
        predictions, new_dates = predict_weather(input_data, dates)

        return jsonify({
            "predictions": [p.tolist() for p in predictions],
            "dates": [str(d.date()) for d in new_dates[1:]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)