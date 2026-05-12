# CAS AI Operations - MLOps Project

## Problem Statement
Predict whether the **hourly CO concentration** at a road-level sensor station 
in an Italian city is **above the historical median** (binary classification: 
`high_co = 1` -> high pollution, `high_co = 0` -> normal).

## Data
The selected data is the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality).

The dataset contains the following variables:

| Variable Name | Role    | Type        | Description                                                                                           | Units      | Missing Values |
|---------------|---------|-------------|-------------------------------------------------------------------------------------------------------|------------|----------------|
| Date          | Feature | Date        |                                                                                                       |            | no             |
| Time          | Feature | Categorical |                                                                                                       |            | no             |
| CO(GT)        | Feature | Integer     | True hourly averaged concentration CO in mg/m³ (reference analyzer)                                   | mg/m³      | no             |
| PT08.S1(CO)   | Feature | Categorical | Hourly averaged sensor response (nominally CO targeted)                                               |            | no             |
| NMHC(GT)      | Feature | Integer     | True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m³ (reference analyzer) | microg/m³  | no             |
| C6H6(GT)      | Feature | Continuous  | True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)                         | microg/m^3 | no             |
| PT08.S2(NMHC) | Feature | Categorical | Hourly averaged sensor response (nominally NMHC targeted)                                             |            | no             |
| NOx(GT)       | Feature | Integer     | True hourly averaged NOx concentration in ppb (reference analyzer)                                    | ppb        | no             |
| PT08.S3(NOx)  | Feature | Categorical | hourly averaged sensor response (nominally NOx targeted)                                              |            | no             |
| NO2(GT)       | Feature | Integer     | True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)                             | microg/m^3 | no             |
| PT08.S4(NO2)  | Feature | Categorical | hourly averaged sensor response (nominally NO2 targeted)                                              |            | no             |
| PT08.S5(O3)   | Feature | Categorical | hourly averaged sensor response (nominally O3 targeted)                                               |            | no             |
| T             | Feature | Continous   | Temperature                                                                                           | °C         | no             |
| RH            | Feature | Continuous  | Relative Humidity                                                                                     | %          | no             |
| AH            | Feature | Continous   | Absolute Humidity                                                                                     |            | no             |

## Features
| Feature                      | Type                   | Description                                                                                   |
|------------------------------|------------------------|-----------------------------------------------------------------------------------------------|
| `pt08_s1_co_rolling24h_mean` | **Aggregated (Batch)** | Rolling 24-hour mean of the PT08.S1 tin-oxide CO sensor. Captures the recent pollution trend. |
| `rh_current`                 | **Real-Time (RT)**     | Current relative humidity (%). Only known at the moment of inference.                         |
| `temperature`                | Context                | Ambient temperature in °C                                                                     |
| `pt08_s2_nmhc`               | Context                | NMHC sensor response                                                                          |
| `pt08_s3_nox`                | Context                | NOx sensor response                                                                           |
| `pt08_s4_no2`                | Context                | NO₂ sensor response                                                                           |
| `ah`                         | Context                | Absolute humidity                                                                             |

**Target:** `high_co` — 1 if CO(GT) > median(CO(GT)), else 0