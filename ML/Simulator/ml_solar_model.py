import json
from pathlib import Path

import numpy as np
import pandas as pd

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=5000):
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = 0.0

        self.x_mean = None
        self.x_std = None

        self.cost_history = []

    def normalize_features(self, X):
        return (X - self.x_mean) / self.x_std

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)

        self.x_std[self.x_std == 0] = 1

        X_norm = self.normalize_features(X)

        total_rows, total_features = X_norm.shape

        self.weights = np.zeros(total_features)
        self.bias = 0.0

        for i in range(self.iterations):
            predictions = np.dot(X_norm, self.weights) + self.bias
            errors = predictions - y

            weight_gradient = (1 / total_rows) * np.dot(X_norm.T, errors)
            bias_gradient = (1 / total_rows) * np.sum(errors)

            self.weights = self.weights - self.learning_rate * weight_gradient
            self.bias = self.bias - self.learning_rate * bias_gradient

            cost = (1 / (2 * total_rows)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_norm = self.normalize_features(X)
        predictions = np.dot(X_norm, self.weights) + self.bias

        return np.maximum(predictions, 0)

def load_actual_data(actual_path):
    df = pd.read_csv(actual_path)

    df["datetime"] = pd.to_datetime(
        df["LocalTime"],
        format="%m/%d/%y %H:%M"
    )

    df = df.rename(columns={
        "Power(MW)": "actual_power_mw"
    })

    df = df[["datetime", "actual_power_mw"]]

    df = (
        df.set_index("datetime")
        .resample("30min")
        .mean()
        .reset_index()
    )

    return df


def load_weather_data(weather_path):
    df = pd.read_csv(weather_path, skiprows=2)

    df["datetime"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=df["Day"],
            hour=df["Hour"],
            minute=df["Minute"]
        )
    )

    selected_columns = [
        "datetime",
        "Temperature",
        "Relative Humidity",
        "GHI",
        "DHI",
        "DNI",
        "Solar Zenith Angle",
        "Cloud Type",
        "Wind Speed"
    ]

    df = df[selected_columns]

    df = df.rename(columns={
        "Temperature": "temperature",
        "Relative Humidity": "relative_humidity",
        "GHI": "ghi",
        "DHI": "dhi",
        "DNI": "dni",
        "Solar Zenith Angle": "solar_zenith_angle",
        "Cloud Type": "cloud_type",
        "Wind Speed": "wind_speed"
    })

    return df


def load_forecast_data(forecast_path, column_name):
    df = pd.read_csv(forecast_path)

    df["datetime"] = pd.to_datetime(
        df["LocalTime"],
        format="%m/%d/%y %H:%M"
    )

    df = df.rename(columns={
        "Power(MW)": column_name
    })

    return df[["datetime", column_name]]

def prepare_training_dataset(actual_path, weather_path):
    actual_df = load_actual_data(actual_path)
    weather_df = load_weather_data(weather_path)

    df = pd.merge(
        weather_df,
        actual_df,
        on="datetime",
        how="inner"
    )

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month

    df["is_daytime"] = np.where((df["hour"] >= 6) & (df["hour"] <= 18), 1, 0)
    df["ghi_temperature"] = df["ghi"] * df["temperature"]
    df["ghi_cloud"] = df["ghi"] * df["cloud_type"]
    df["zenith_ghi"] = df["solar_zenith_angle"] * df["ghi"]

    numeric_columns = [
        "temperature",
        "relative_humidity",
        "ghi",
        "dhi",
        "dni",
        "solar_zenith_angle",
        "cloud_type",
        "wind_speed",
        "hour",
        "month",
        "is_daytime",
        "ghi_temperature",
        "ghi_cloud",
        "zenith_ghi",
        "actual_power_mw"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    df = df[df["actual_power_mw"] >= 0]

    return df


def train_test_split_manual(df, train_ratio=0.8, random_seed=42):
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    total_rows = len(shuffled_df)
    train_size = int(total_rows * train_ratio)

    train_df = shuffled_df.iloc[:train_size].copy()
    test_df = shuffled_df.iloc[train_size:].copy()

    return train_df, test_df

def evaluate_model(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    errors = y_pred - y_true

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

def export_model_outputs(model, df, feature_columns, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    sample_df = df.head(500).copy()
    sample_predictions = model.predict(sample_df[feature_columns].values)

    sample_df["predicted_power_mw"] = sample_predictions
    sample_df["actual_power_kw"] = sample_df["actual_power_mw"] * 1000
    sample_df["predicted_power_kw"] = sample_df["predicted_power_mw"] * 1000

    export_columns = [
        "datetime",
        "temperature",
        "relative_humidity",
        "ghi",
        "dhi",
        "dni",
        "solar_zenith_angle",
        "cloud_type",
        "wind_speed",
        "actual_power_mw",
        "predicted_power_mw",
        "actual_power_kw",
        "predicted_power_kw"
    ]

    sample_df[export_columns].to_csv(
        output_folder / "ml_prediction_sample.csv",
        index=False
    )

    sample_df[export_columns].to_json(
        output_folder / "ml_prediction_sample.json",
        orient="records",
        indent=2,
        date_format="iso"
    )


def export_residual_error_dataset(model, df, feature_columns, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    sample_df = df.head(500).copy()
    predictions_mw = model.predict(sample_df[feature_columns].values)

    sample_df["actual_power_kw"] = sample_df["actual_power_mw"] * 1000
    sample_df["predicted_power_kw"] = predictions_mw * 1000
    sample_df["error_kw"] = sample_df["actual_power_kw"] - sample_df["predicted_power_kw"]
    sample_df["absolute_error_kw"] = np.abs(sample_df["error_kw"])

    export_columns = [
        "datetime",
        "actual_power_kw",
        "predicted_power_kw",
        "error_kw",
        "absolute_error_kw"
    ]

    sample_df[export_columns].to_json(
        output_folder / "ml_residual_error.json",
        orient="records",
        indent=2,
        date_format="iso"
    )


def export_ideal_vs_ml_dataset(model, df, feature_columns, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    sample_df = df.head(500).copy()
    predictions_mw = model.predict(sample_df[feature_columns].values)

    sample_df["predicted_power_kw"] = predictions_mw * 1000

    dataset_capacity_kw = 13000

    sample_df["hour_float"] = (
        sample_df["datetime"].dt.hour +
        sample_df["datetime"].dt.minute / 60
    )

    sample_df["ideal_power_kw"] = sample_df["hour_float"].apply(
        lambda hour: dataset_capacity_kw * max(0.0, np.sin(hour * np.pi / 12))
    )

    export_columns = [
        "datetime",
        "hour_float",
        "ideal_power_kw",
        "predicted_power_kw"
    ]

    sample_df[export_columns].to_json(
        output_folder / "ideal_vs_ml_solar.json",
        orient="records",
        indent=2,
        date_format="iso"
    )


def export_forecast_comparison_dataset(model, df, feature_columns, output_folder, base_dir):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    da_path = base_dir / "data" / "222628_DA_DPV_13MW_60m.csv"
    ha4_path = base_dir / "data" / "222628_HA4_DPV_13MW_60m.csv"

    da_df = load_forecast_data(da_path, "day_ahead_power_mw")
    ha4_df = load_forecast_data(ha4_path, "four_hours_ahead_power_mw")

    ml_df = df.copy()
    ml_df["ml_power_mw"] = model.predict(ml_df[feature_columns].values)

    comparison_df = pd.merge(
        ml_df[["datetime", "actual_power_mw", "ml_power_mw"]],
        da_df,
        on="datetime",
        how="inner"
    )

    comparison_df = pd.merge(
        comparison_df,
        ha4_df,
        on="datetime",
        how="inner"
    )

    comparison_df = comparison_df.head(500).copy()

    comparison_df["actual_power_kw"] = comparison_df["actual_power_mw"] * 1000
    comparison_df["ml_power_kw"] = comparison_df["ml_power_mw"] * 1000
    comparison_df["day_ahead_power_kw"] = comparison_df["day_ahead_power_mw"] * 1000
    comparison_df["four_hours_ahead_power_kw"] = comparison_df["four_hours_ahead_power_mw"] * 1000

    export_columns = [
        "datetime",
        "actual_power_kw",
        "ml_power_kw",
        "day_ahead_power_kw",
        "four_hours_ahead_power_kw"
    ]

    comparison_df[export_columns].to_json(
        output_folder / "forecast_comparison.json",
        orient="records",
        indent=2,
        date_format="iso"
    )

def train_solar_model():
    base_dir = Path(__file__).resolve().parent

    actual_path = base_dir / "data" / "222628_Actual_DPV_13MW_5m.csv"
    weather_path = base_dir / "data" / "222628_Weather_30m.csv"

    dashboard_output = base_dir / "output" / "dashboard_data"

    df = prepare_training_dataset(actual_path, weather_path)

    feature_columns = [
        "temperature",
        "relative_humidity",
        "ghi",
        "dhi",
        "dni",
        "solar_zenith_angle",
        "cloud_type",
        "wind_speed",
        "hour",
        "month",
        "is_daytime",
        "ghi_temperature",
        "ghi_cloud",
        "zenith_ghi"
    ]

    target_column = "actual_power_mw"

    train_df, test_df = train_test_split_manual(
        df,
        train_ratio=0.8,
        random_seed=42
    )

    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values

    X_test = test_df[feature_columns].values
    y_test = test_df[target_column].values

    model = LinearRegressionFromScratch(
        learning_rate=0.01,
        iterations=5000
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_metrics = evaluate_model(y_train, train_predictions)
    test_metrics = evaluate_model(y_test, test_predictions)

    model_report = {
        "model_type": "Linear Regression from Scratch",
        "learning_rate": model.learning_rate,
        "iterations": model.iterations,
        "features": feature_columns,
        "target": target_column,
        "total_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
        "feature_mean": model.x_mean.tolist(),
        "feature_std": model.x_std.tolist()
    }

    dashboard_output.mkdir(parents=True, exist_ok=True)

    with open(dashboard_output / "ml_model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(model_report, file, indent=2)

    export_model_outputs(
        model=model,
        df=df,
        feature_columns=feature_columns,
        output_folder=dashboard_output
    )

    export_residual_error_dataset(
        model=model,
        df=df,
        feature_columns=feature_columns,
        output_folder=dashboard_output
    )

    export_ideal_vs_ml_dataset(
        model=model,
        df=df,
        feature_columns=feature_columns,
        output_folder=dashboard_output
    )

    export_forecast_comparison_dataset(
        model=model,
        df=df,
        feature_columns=feature_columns,
        output_folder=dashboard_output,
        base_dir=base_dir
    )

    print("\n===== ML SOLAR MODEL TRAINED =====")
    print("Total rows:", len(df))
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    print("\nTrain metrics:")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nFiles generated:")
    print(dashboard_output / "ml_model_metrics.json")
    print(dashboard_output / "ml_prediction_sample.csv")
    print(dashboard_output / "ml_prediction_sample.json")
    print(dashboard_output / "ml_residual_error.json")
    print(dashboard_output / "ideal_vs_ml_solar.json")
    print(dashboard_output / "forecast_comparison.json")

    return model, df, feature_columns

class SolarMLPredictor:
    def __init__(self):
        self.model = None
        self.weather_data = None
        self.feature_columns = None
        self.max_training_power_mw = None

        self.train()

    def train(self):
        self.model, df, self.feature_columns = train_solar_model()

        self.weather_data = df.copy().reset_index(drop=True)
        self.max_training_power_mw = df["actual_power_mw"].max()

    def get_weather_row_by_sim_time(self, sim_time_hours):
        if self.weather_data is None or len(self.weather_data) == 0:
            raise ValueError("Weather data is not loaded.")

        row_index = int((sim_time_hours * 2) % len(self.weather_data))

        return self.weather_data.iloc[row_index]

    def predict_generation_kw(self, sim_time_hours, household_peak_kw=5.0):
        row = self.get_weather_row_by_sim_time(sim_time_hours)

        feature_values = row[self.feature_columns].values
        predicted_mw = self.model.predict(feature_values)[0]

        predicted_kw_total_plant = predicted_mw * 1000

        dataset_capacity_kw = 13000
        scaled_prediction_kw = (predicted_kw_total_plant / dataset_capacity_kw) * household_peak_kw

        scaled_prediction_kw = max(0.0, min(scaled_prediction_kw, household_peak_kw))

        return scaled_prediction_kw


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    train_solar_model()