# Imports
import os
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Constants
# Looking at Rome, so latitude and longitude have to be set
LATITUDE = 41.9028
LONGITUDE = 12.4964

# Define the feature group name
# https://docs.hopsworks.ai/feature-store-api/2.5.17/generated/feature_group/
# https://docs.hopsworks.ai/latest/user_guides/fs/feature_group/create/
FEATURE_GROUP_NAME = "air_quality_features"
FEATURE_GROUP_VERSION = 1

# Define the amount of days that should be used for backfilling
HISTORY_DAYS = 365


# Fetch the data
# https://open-meteo.com/en/docs/air-quality-api?time_mode=time_interval&start_date=2026-05-05&end_date=2026-05-17&timezone=Europe%2FBerlin&hourly=pm2_5,carbon_monoxide,nitrogen_dioxide,european_aqi
# The API allows to select the features wanted in the UI and basically the code necessary gets generated below
# Though it has to be adapted for easier usage here

def fetch_historical_data(
        start_date: str,
        end_date: str,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
) -> pd.DataFrame:
    """
    Fetch historical data from Air Quality API

    :param start_date: The start date of the historical data
    :param end_date: The end date of the historical data
    :param latitude: The latitude of the location, defaults to latitude constant set
    :param longitude: The longitude of the location, defaults to longitude constant set
    :return: dataframe with the data
    """
    print(f"Fetching historical data from Air Quality API from {start_date} to {end_date}")

    # Air quality variables
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "carbon_monoxide",
                "nitrogen_dioxide",
                "pm2_5",
                "european_aqi",
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Europe/Rome",
        },
        timeout=30,
    )
    aq_resp.raise_for_status()
    aq_json = aq_resp.json()

    # Get additional historical data -> weather variables
    wx_resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "relative_humidity_2m",
                "wind_speed_10m",
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Europe/Rome",
        },
        timeout=30,
    )
    wx_resp.raise_for_status()
    wx_json = wx_resp.json()

    # Build DataFrames
    aq_df = pd.DataFrame({
        "datetime": pd.to_datetime(aq_json["hourly"]["time"]),
        "carbon_monoxide": aq_json["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": aq_json["hourly"]["nitrogen_dioxide"],
        "pm2_5": aq_json["hourly"]["pm2_5"],
        "european_aqi": aq_json["hourly"]["european_aqi"],
    })

    wx_df = pd.DataFrame({
        "datetime": pd.to_datetime(wx_json["hourly"]["time"]),
        "relative_humidity": wx_json["hourly"]["relative_humidity_2m"],
        "wind_speed": wx_json["hourly"]["wind_speed_10m"],
    })

    df = pd.merge(aq_df, wx_df, on="datetime", how="inner")
    print(f"  Raw rows fetched: {len(df)}")
    return df


# Clean the data set
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing target or key sensor values.

    :param df: dataframe with the data
    :return: dataframe with the cleaned data
    """
    print("Cleaning data")
    df = df.copy()
    df = df.dropna(subset=["european_aqi", "carbon_monoxide"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")
    return df


# Engineer features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the aggregated features

    :param df: dataframe with the data
    :return: dataframe with the cleaned data
    """
    print("Engineering features...")
    df = df.copy()

    # Rolling 24h means (aggregated / batch features)
    # Add the new features to the dataset
    for col, new_col in [
        ("carbon_monoxide", "co_rolling24h_mean"),
        ("nitrogen_dioxide", "no2_rolling24h_mean"),
        ("pm2_5", "pm25_rolling24h_mean"),
    ]:
        df[new_col] = (
            df[col]
            .rolling(window=24, min_periods=6)
            .mean()
            .round(4)
        )

    # True/false target: European AQI >= 50 → "Poor" or worse -> integer representation
    df["high_aqi"] = (df["european_aqi"] >= 50).astype(int)

    # Round RT features
    df["relative_humidity"] = df["relative_humidity"].round(2)
    df["wind_speed"] = df["wind_speed"].round(2)

    # Drop rows where rolling features are still NaN
    df = df.dropna(subset=["co_rolling24h_mean"])

    # Create a unique row id from the timestamp
    df["row_id"] = df["datetime"].dt.strftime("%Y%m%d%H").astype(int)

    # Keep only the columns we need
    df = df[[
        "row_id",
        "datetime",
        "co_rolling24h_mean",
        "no2_rolling24h_mean",
        "pm25_rolling24h_mean",
        "relative_humidity",
        "wind_speed",
        "high_aqi",
    ]].rename(columns={"datetime": "event_time"})

    print(f"  Rows after feature engineering: {len(df)}")
    return df


# Add the dataframe to Hopswork
def write_to_feature_store(df: pd.DataFrame) -> None:
    """Connect to Hopsworks and upsert data into the Feature Group.

    :param df: dataframe with the data
    """
    print("Connecting to Hopsworks")
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"],
                              project=os.environ["HOPSWORKS_PROJECT_ID"])
    fs = project.get_feature_store()

    # Create a feature group or get the existing feature group
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["row_id"],
        event_time="event_time",
        description=(
            "Hourly air quality features for Rome (Open-Meteo). "
            "Includes 24h rolling means (aggregated) and current "
            "humidity/wind (RT features)."
        ),
        online_enabled=True,
    )

    print(f"Inserting {len(df)} rows into feature group '{FEATURE_GROUP_NAME}'")
    fg.insert(df, write_options={"wait_for_job": True})
    print("  Done.")


# Main
def main():
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    raw_df = fetch_historical_data(str(start_date), str(end_date))
    clean_df = clean_data(raw_df)
    features_df = engineer_features(clean_df)

    print(features_df.head())
    print(features_df.describe())

    write_to_feature_store(features_df)


if __name__ == "__main__":
    main()
