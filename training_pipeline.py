# Imports
import os
import joblib
import json
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from configparser import ConfigParser

constants = ConfigParser()
constants.read("constants.ini")


load_dotenv()

# Constants
# FEATURE_GROUP_NAME = "air_quality_features"
# FEATURE_GROUP_VERSION = 1
# FEATURE_VIEW_NAME = "air_quality_feature_view"
# FEATURE_VIEW_VERSION = 1
#
# FEATURE_COLUMNS = [
#     "co_rolling24h_mean",
#     "no2_rolling24h_mean",
#     "pm25_rolling24h_mean",
#     "relative_humidity",
#     "wind_speed"
# ]
# TARGET_COLUMN = "high_aqi"
#
# MODEL_NAME = "air_quality_classifier"
# MODEL_VERSION = 1
# MODEL_DIR = "models"

# Load the feature group and create a feature view
# To make sure that the script can be executed several times, the feature view is
# either loaded or created in a try-except block

def get_feature_view(feature_store):
    """
    Get or create a Feature View from the air quality Feature Group.

    :param feature_store: The Hopsworks feature store
    :return: The Feature View
    """
    print("Loading Feature Group")
    feature_group = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    print("Creating / loading Feature View")
    # try: todo
    #     feature_view = feature_store.get_feature_view(
    #         name=FEATURE_VIEW_NAME,
    #         version=FEATURE_VIEW_VERSION,
    #     )
    #     print("Existing Feature View found.")
    # except Exception:
    print("No existing Feature View found, creating new one")

    # Get the Features and the target
    query = feature_group.select(FEATURE_COLUMNS + [TARGET_COLUMN])

    # Create the Feature View
    feature_view = feature_store.create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        query=query,
        labels=[TARGET_COLUMN],
        description="Feature view for air quality AQI classifier.",
    )
    print("Feature View created.")

    return feature_view


# Build training dataset from the feature view
def get_training_data(feature_view) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pull training data from the Feature View and return X, y.

    :param feature_view: The Hopsworks Feature View
    :return: A tuple of (X, y)
    """
    print("Creating training dataset from Feature View")
    X, y = feature_view.training_data(
        description="Air quality training dataset",
    )
    print(f"Training dataset shape: X={X.shape}, y={y.shape}")
    return X, y


# Train the model
def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a Random Forest classifier on the training data.

    :param X: Training features
    :param y: Training labels
    :return: A tuple of (model, metrics)
    """
    print("Splitting into train / test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, # Define the size of the test set
        random_state=42, # Set a random state to make sure that results are reproducible
        stratify=y, # Stratify the data to make sure that the test set has the same distribution as the training set (https://medium.com/@aymuosmukherjee/why-do-we-use-stratify-in-train-test-split-e3eb296a5494)
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Training Random Forest classifier")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Train the model
    model.fit(X_train, y_train)

    print("Evaluating model")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1_score": round(report["weighted avg"]["f1-score"], 4),
    }
    return model, metrics

# Save the model locally and to the registry
def save_model(project, model, metrics: dict) -> None:
    """
    Persist the model locally with joblib, then upload it to the
    Hopsworks Model Registry.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved locally to {model_path}")

    # Save metrics alongside the model artifact
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Uploading model to Hopsworks Model Registry")
    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        metrics=metrics,
        description=(
            "Random Forest classifier predicting high AQI (>= 50) "
            "in Rome based on Open-Meteo air quality data."
        ),
    )
    hw_model.save(MODEL_DIR)
    print(f"Model '{MODEL_NAME}' v{MODEL_VERSION} uploaded successfully.")

def main():
    print("Connecting to Hopsworks")
    hw_project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = hw_project.get_feature_store()

    fv = get_feature_view(fs)
    X, y = get_training_data(fv)
    model, metrics = train_model(X, y)
    save_model(hw_project, model, metrics)

    print("\nTraining pipeline complete.")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()