# Imports
import os
import shutil

import requests
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# Constants
LATITUDE  = 41.9028
LONGITUDE = 12.4964

FEATURE_GROUP_NAME = "air_quality_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME     = "air_quality_feature_view"
FEATURE_VIEW_VERSION  = 1
MODEL_NAME            = "air_quality_classifier"
MODEL_VERSION         = 1
MODEL_DIR             = "downloaded_model"

FEATURE_COLUMNS = [
    "co_rolling24h_mean",
    "no2_rolling24h_mean",
    "pm25_rolling24h_mean",
    "relative_humidity",
    "wind_speed",
]

# Fetch the live data
def fetch_live_rt_features(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
) -> dict:
    """
    Fetch the most recent hourly relative_humidity and wind_speed
    from the Open-Meteo forecast endpoint (the current hour is the
    RT feature — it is only known at inference time).
    """
    print("Fetching live RT features from Open-Meteo")

    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":   latitude,
            "longitude":  longitude,
            "current":     [
                "relative_humidity_2m",
                "wind_speed_10m",
            ],
            "timezone":   "Europe/Rome",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(data)

    times    = data["current"]["time"]
    humidity = data["current"]["relative_humidity_2m"]
    wind     = data["current"]["wind_speed_10m"]

    rt_features = {
        "timestamp": times,
        "relative_humidity": humidity,
        "wind_speed": wind,
    }

    print(f"Live RT features @ {rt_features['timestamp']}:")
    print(f"  relative_humidity = {rt_features['relative_humidity']} %")
    print(f"  wind_speed        = {rt_features['wind_speed']} km/h")
    return rt_features


# Load the aggregated features

def fetch_aggregated_features(fs) -> dict:
    """
    Retrieve the most recently stored aggregated features (rolling 24h
    means) from the Hopsworks Feature Store via the Feature View.
    """
    print("Loading aggregated features from Hopsworks Feature Store")

    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    latest = df.sort_values("event_time", ascending=False).iloc[0]

    agg_features = {
        "co_rolling24h_mean": latest["co_rolling24h_mean"],
        "no2_rolling24h_mean": latest["no2_rolling24h_mean"],
        "pm25_rolling24h_mean": latest["pm25_rolling24h_mean"],
    }

    print("  Aggregated features loaded:")
    for k, v in agg_features.items():
        print(f"    {k} = {v}")

    return agg_features


# ── 3. Download Model from Registry ──────────────────────────────────────────

def download_model(project) -> object:
    """Download the latest model artifact from the Hopsworks Model Registry."""
    print("Downloading model from Hopsworks Model Registry")
    mr = project.get_model_registry()
    hw_model = mr.get_model(
        name=MODEL_NAME,
        version=MODEL_VERSION,
    )

    # Remove previously downloaded model and metrics
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    model_dir = hw_model.download(local_path=MODEL_DIR)
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


# ── 4. Predict ────────────────────────────────────────────────────────────────

def predict(model, agg_features: dict, rt_features: dict) -> None:
    """
    Combine aggregated + RT features, run prediction, and print result.
    """
    input_row = {
        "co_rolling24h_mean":   agg_features["co_rolling24h_mean"],
        "no2_rolling24h_mean":  agg_features["no2_rolling24h_mean"],
        "pm25_rolling24h_mean": agg_features["pm25_rolling24h_mean"],
        "relative_humidity":    rt_features["relative_humidity"],
        "wind_speed":           rt_features["wind_speed"],
    }

    X = pd.DataFrame([input_row])[FEATURE_COLUMNS]

    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    print("\n" + "="*50)
    print("AIR QUALITY PREDICTION")
    print("="*50)
    print(f"Timestamp      : {rt_features['timestamp']}")
    print(f"Location       : Rome, Italy ({LATITUDE}°N, {LONGITUDE}°E)")
    print()
    print("Input Features:")
    for col in FEATURE_COLUMNS:
        print(f"  {col:<26} = {input_row[col]}")
    print()
    print(f"Prediction     : {'⚠️  HIGH AQI (>= 50)' if prediction == 1 else '✅  NORMAL AQI (< 50)'}")
    print(f"Probability    : P(normal)={probability[0]:.3f}  P(high)={probability[1]:.3f}")
    print("="*50)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Fetch live RT features (current conditions, only known now)
    rt_features = fetch_live_rt_features()

    # Step 2: Connect to Hopsworks
    print("\nConnecting to Hopsworks")
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()

    # Step 3: Load aggregated features from Feature Store
    agg_features = fetch_aggregated_features(fs)

    # Step 4: Download model from registry
    model = download_model(project)

    # Step 5: Predict
    predict(model, agg_features, rt_features)


if __name__ == "__main__":
    main()
