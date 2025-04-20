#!/usr/bin/env python3
"""
app.py – interactive GIS+ML demo
────────────────────────────────
• Single‑point traffic prediction
• Hour‑selectable traffic heat‑map
• Collision hot‑spots with user‑chosen radius / severity
"""

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import folium
from folium import plugins
from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
from joblib import load

# ───── paths & one‑time loads ────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
MODEL_DIR = ROOT  / "models"

# load our models
reg_model = load(MODEL_DIR / "xgb_regressor.joblib")
clf_model = load(MODEL_DIR / "xgb_classifier.joblib")

# load city‑wide table for heat‑map
df_city = pd.read_csv(
    DATA_DIR / "combined_final_dataset.csv",
    parse_dates=["timestamp"], low_memory=False
)
df_city["hour"] = df_city["timestamp"].dt.hour

# raw collisions for hot‑spots
df_coll = pd.read_csv(
    DATA_DIR / "Traffic_Collisions_Toronto_data.csv",
    parse_dates=["OccurrenceDate"], low_memory=False
).rename(columns={"Latitude":"lat","Longitude":"lon"})
# ── remove tz so comparisons work ────────────────────────────────
df_coll["OccurrenceDate"] = df_coll["OccurrenceDate"].dt.tz_convert(None)

# ───── Flask setup ───────────────────────────────────────────────
app      = Flask(__name__)
geocoder = Nominatim(user_agent="traffic_app")

MONTHS  = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
DAYS    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEATHER = ["Sunny","Rainy","Snowy"]
MONTH2I = {m: i+1 for i, m in enumerate(MONTHS)}
COLOUR  = {"Low":"green","Medium":"orange","High":"red"}

def build_collision_heat(center, radius_m, fatal_only):
    cutoff = datetime.utcnow() - timedelta(days=365)
    sub = df_coll[df_coll["OccurrenceDate"] >= cutoff]
    if fatal_only:
        sub = sub[sub["Fatalities"] > 0]
    lat0, lon0 = center
    ddeg = radius_m / 1000.0 / 111.0
    box = sub[
        sub["lat"].between(lat0-ddeg, lat0+ddeg) &
        sub["lon"].between(lon0-ddeg, lon0+ddeg)
    ]
    return box[["lat","lon"]].dropna().values

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        # ── parse inputs ─────────────────────────
        loc_raw   = request.form["location"]
        temp_c    = float(request.form["temperature"])
        wind_sp   = float(request.form["wind_speed"])
        hour      = int(request.form["hour"])
        hour_hmap = int(request.form.get("hour_heatmap", hour))
        dow       = request.form["day_of_week"]
        month     = MONTH2I[request.form["month"]]
        weather   = request.form["weather"]
        hmap_on   = request.form.get("show_heatmap")=="on"
        rad_m     = int(request.form.get("coll_radius",150))
        fatal_on  = request.form.get("fatal_only")=="on"
        kpi_on    = request.form.get("show_collisions")=="on"

        # ── geocode & feature row ─────────────────
        geo = geocoder.geocode(f"{loc_raw}, Toronto, ON")
        lat, lon = (geo.latitude, geo.longitude) if geo else (43.65, -79.38)

        X = pd.DataFrame([{
            "lat":lat, "lon":lon,
            "temp_c":temp_c, "wind_speed":wind_sp,
            "hour":hour, "day_of_week":dow,
            "month":month,
            "weather_summary":f"{weather}, {temp_c:.1f}°C",
            "location_name":loc_raw,
            "precip_flag": 0 if weather=="Sunny" else 1
        }])

        # cyclical features
        X["hour_sin"]  = np.sin(2*np.pi*X["hour"]/24)
        X["hour_cos"]  = np.cos(2*np.pi*X["hour"]/24)
        X["month_sin"] = np.sin(2*np.pi*(X["month"]-1)/12)
        X["month_cos"] = np.cos(2*np.pi*(X["month"]-1)/12)

        # ── predict ───────────────────────────────
        vol     = float(reg_model.predict(X)[0])
        cls_enc = int(clf_model.predict(X)[0])
        cls     = {0:"High",1:"Low",2:"Medium"}.get(cls_enc,"Unknown")

        # ── build map ──────────────────────────────
        m = folium.Map([lat,lon], zoom_start=13, tiles="cartodbpositron")
        folium.Marker(
            [lat,lon],
            popup=f"<b>Vol:</b> {vol:.0f}<br><b>Cong:</b> {cls}",
            icon=folium.Icon(color=COLOUR.get(cls,"blue"), icon="car", prefix="fa")
        ).add_to(m)

        if hmap_on:
            pts = df_city[df_city["hour"]==hour_hmap][["lat","lon","total_traffic_volume"]].values
            plugins.HeatMap(pts, radius=8, blur=12, min_opacity=0.3,
                            name=f"Heat‑map {hour_hmap:02d}:00").add_to(m)

        if kpi_on:
            coll_pts = build_collision_heat((lat,lon), rad_m, fatal_on)
            if len(coll_pts):
                plugins.HeatMap(
                    coll_pts, gradient={0.4:"yellow",0.8:"red"},
                    radius=6, blur=10, min_opacity=0.3,
                    name=f"Collisions ≤{rad_m}m"
                ).add_to(m)

        folium.LayerControl().add_to(m)

        return render_template(
            "result.html",
            pred_volume=vol,
            pred_congestion=cls,
            map_html=m._repr_html_(),
            input_data=X.to_html(classes="table table-striped table-sm", index=False)
        )

    # GET form
    return render_template(
        "index.html",
        month_options=MONTHS,
        days_of_week=DAYS,
        weather_options=WEATHER
    )

if __name__=="__main__":
    app.run(debug=True, threaded=True)
