from flask import Flask, render_template, request
import pandas as pd
import folium
from geopy.geocoders import Nominatim
from joblib import load
import os

app = Flask(__name__)

# Load the saved models
reg_model = load("tuned_xgb_regressor.joblib")
clf_model = load("tuned_xgb_classifier.joblib")

# Define options for dropdowns
month_options = ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weather_options = ["Sunny", "Rainy", "Snowy"]

# Mapping for months (for model input)
month_mapping = {name: idx+1 for idx, name in enumerate(month_options)}

# Initialize geocoder (using Nominatim)
geolocator = Nominatim(user_agent="traffic_app")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form inputs
        location_input = request.form.get("location")
        temperature = request.form.get("temperature")
        wind_speed = request.form.get("wind_speed")
        hour = request.form.get("hour")
        day_of_week = request.form.get("day_of_week")
        month_str = request.form.get("month")
        weather = request.form.get("weather")
        
        # Convert numeric inputs
        try:
            temperature = float(temperature)
        except:
            temperature = 0.0
        try:
            wind_speed = float(wind_speed)
        except:
            wind_speed = 0.0
        try:
            hour = int(hour)
        except:
            hour = 0
        month = month_mapping.get(month_str, 0)
        
        # Geocode the location (append ", Toronto, ON" for accuracy)
        loc = geolocator.geocode(f"{location_input}, Toronto, ON")
        if loc:
            lat, lon = loc.latitude, loc.longitude
        else:
            lat, lon = 43.65, -79.38  # default to downtown Toronto

        # Format weather summary as "Weather, temp°C"
        weather_summary = f"{weather}, {temperature:.1f}°C"
        location_name = location_input  # use the user-entered location

        # Create input DataFrame matching model features
        input_data = pd.DataFrame({
            "lat": [lat],
            "lon": [lon],
            "temp_c": [temperature],
            "wind_speed": [wind_speed],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "weather_summary": [weather_summary],
            "location_name": [location_name]
        })

        # Make predictions using the loaded models
        pred_volume = reg_model.predict(input_data)[0]
        pred_clf_encoded = clf_model.predict(input_data)[0]
        # (Assuming the LabelEncoder mapping from training: 0:"High", 1:"Low", 2:"Medium")
        congestion_mapping = {0: "High", 1: "Low", 2: "Medium"}
        pred_congestion = congestion_mapping.get(pred_clf_encoded, "Unknown")

        # Create a Folium map centered at the given location
        m = folium.Map(location=[lat, lon], zoom_start=13)
        marker_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(pred_congestion, "blue")
        popup_text = f"<strong>Traffic Volume:</strong> {pred_volume:.2f}<br><strong>Congestion:</strong> {pred_congestion}"
        folium.Marker([lat, lon], popup=popup_text, icon=folium.Icon(color=marker_color)).add_to(m)
        map_html = m._repr_html_()

        # Render the results page
        return render_template("result.html",
                               pred_volume=pred_volume,
                               pred_congestion=pred_congestion,
                               map_html=map_html,
                               input_data=input_data.to_html(classes="table table-striped", index=False))
    return render_template("index.html", month_options=month_options, days_of_week=days_of_week, weather_options=weather_options)

if __name__ == "__main__":
    app.run(debug=True)
