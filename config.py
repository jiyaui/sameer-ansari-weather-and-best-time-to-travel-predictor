RANDOM_STATE = 42

# City name: (latitude, longitude)
CITIES = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Bengaluru": (12.97, 77.59),
    "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27),
    "Jaipur": (26.91, 75.79),
    "Shimla": (31.10, 77.17),
    "Goa": (15.50, 73.83),
    "Leh": (34.15, 77.58),
}

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
TRAIN_END_YEAR = 2022          # 2015-2022 train, 2023-2024 test

RAW_PATH = "data/raw/weather_daily.csv"
MONTHLY_PATH = "data/processed/monthly.csv"
CLUSTER_PATH = "data/processed/climatology_clusters.csv"
MODEL_PATH = "models/classifiers.pkl"
METRICS_PATH = "reports/metrics.csv"

# Features: sirf PICHHLE saalon ke history + location + season
FEATURES = [
    "tavg_hist", "rain_total_hist", "rainy_days_hist", "wind_hist",
    "m_sin", "m_cos", "lat", "lon",
]

# preference: (min avg temp, max avg temp, max rainy days in a month)
PREFERENCES = {
    "cool": (14, 24, 6),
    "warm": (24, 32, 6),
    "dry": (15, 32, 2),
}
