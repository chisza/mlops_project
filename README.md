# CAS AI Operations - MLOps Project

## Problem Statement
In the MLOps part of the CAS AI Operation, we work on small projects. 
This project predicts whether the **European Air Quality Index (AQI) will be high
(≥ 50 = "Poor" or worse)** for Rome, Italy, using live data from the
[Open-Meteo Air Quality API](https://open-meteo.com/).
The city of Rome was arbitrarily chosen, as air quality is interesting to look at
in places where there is high pollution.

## Data Source

| Property      | Value                                            |
|---------------|--------------------------------------------------|
| Provider      | [Open-Meteo](https://open-meteo.com/)            |
| Endpoints     | Air Quality API + Historical Weather Archive API |
| Location      | Rome, Italy (41.9028°N, 12.4964°E)               |
| Granularity   | Hourly                                           |
| History used  | Last 365 days                                    |
| Auth required | None (free, no API key)                          |
| License       | CC BY 4.0                                        |

**Historical vs. live split:** The feature pipeline backfills the last
365 days of data for training. At inference time, the rolling aggregated
features come from the Feature Store (pre-computed), while the RT
features (humidity, wind speed) are fetched live from the Open-Meteo
forecast endpoint.

**Documentation**: [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs)

## Features

The following features will be used in the model:

| Feature                | Type           | Description                                                 |
|------------------------|----------------|-------------------------------------------------------------|
| `co_rolling24h_mean`   | **Aggregated** | 24h rolling mean of CO concentration (µg/m³)                |
| `no2_rolling24h_mean`  | **Aggregated** | 24h rolling mean of NO₂ concentration (µg/m³)               |
| `pm25_rolling24h_mean` | **Aggregated** | 24h rolling mean of PM2.5 (µg/m³)                           |
| `relative_humidity`    | **RT (live)**  | Current relative humidity (%), only known at inference time |
| `wind_speed`           | **RT (live)**  | Current wind speed (km/h), only known at inference time     |

### Target

The following features is the target that should be predicted:

| Field      | Description                                      |
|------------|--------------------------------------------------|
| `high_aqi` | 1 if European AQI ≥ 50 ("Poor" or worse), else 0 |

## Model

- **Type:** Random Forest Classifier (scikit-learn)
- **Preprocessing:** StandardScaler (in a sklearn Pipeline)
- **Parameters:** 100 estimators, max depth 10
- **Note:** Model performance (accuracy, F1) is not the focus of this
  project. The goal is a functioning FTI pipeline.

## Setup

### Prerequisites

- Python **3.13** (Hopsworks requires `< 3.14`)
- A free [Hopsworks account](https://app.hopsworks.ai)
- Your Hopsworks API key

### Installation

```bash
git clone https://github.com/chisza/mlops_project.git
cd mlops_project
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```dotenv
HOPSWORKS_API_KEY="<your-api-key>"
HOPSWORKS_PROJECT_ID="<your-project-id>"
```

## Pipelines

Run the pipelines in order.

### Step 1 — Feature Pipeline

Fetches 365 days of historical data, engineers features, and writes
them to the Hopsworks Feature Store.

```bash
python feature_pipeline.py
```
### Step 2 — Training Pipeline

Loads features from the Feature Store, trains the model, and uploads
it to the Model Registry.

```bash
python training_pipeline.py
```

### Step 3 — Inference Pipeline

Fetches live RT features, combines them with aggregated features from
the store, and predicts whether AQI will be high.

```bash
python inference_pipeline.py
```

## Reflexion & Limitations

- **No scheduled runs**: Currently, the pipeline has to be triggered manually. This could be improved by creating a schedule that automatically triggers the pipelines. Suitable would be an hourly trigger, as the data granularity is hourly.
- **Rolling feature aging**: The aggregated features are calculated and stored. The run of the inference pipeline does not trigger the feature pipeline or check for the age of the features. Depending on the time of the last run of the feature pipeline (and with it the training pipeline), the features (and the model) might be obsolete.
- **True / false target**: The target is calculated as true / false value, not as an actual value.
- 