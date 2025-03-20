import streamlit as st
import requests
import pandas as pd

st.title("5-Day Weather Forecast")

# User input for location
latitude = st.number_input("Enter Latitude:", value=22)
longitude = st.number_input("Enter Longitude:", value=79)

if st.button("Predict Weather"):
    api_url = "http://127.0.0.1:5000/predict"
    response = requests.post(api_url, json={"latitude": latitude, "longitude": longitude})

    if response.status_code == 200:
        result = response.json()
        predictions = result["predictions"]
        new_dates = result["dates"]

        st.write("### 📊 Predicted Weather Data for the Next 7 Days:")
        flood_predictions = []

        for i, (date, pred) in enumerate(zip(new_dates, predictions)):
            flood = (
                (pred[0] >= 5.0 and pred[1] >= 10.0) or
                (pred[2] >= 15.0) or
                (pred[3] >= 12.0)
            )
            flood_predictions.append(flood)

            st.write(f"📅 **Date:** {date}")
            st.write(f"🌧️ **Precipitation (mm):** {pred[0]:.2f}")
            st.write(f"☔ **Rain (mm):** {pred[1]:.2f}")
            st.write(f"🕒 **Precipitation Hours:** {pred[2]:.2f}")
            st.write(f"🌊 **River Discharge (m³/s):** {pred[3]:.2f}")
            st.write(f"🚨 **Flood Risk:** {'Yes' if flood else 'No'}")
            st.write("_________________________________________")

        # DataFrame for Display
        df_predictions = pd.DataFrame({
            "Date": new_dates,
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
        st.write("### 📈 Data Trends Over 7 Days:")
        st.line_chart(df_predictions.set_index("Date")[['Precipitation (mm)', 'Rain (mm)', 'River Discharge (m³/s)']])
    else:
        st.error("Failed to get predictions from API.")