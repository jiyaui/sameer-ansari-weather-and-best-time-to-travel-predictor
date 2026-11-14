# Multi-City Weather Comparison + Best Time to Travel Predictor

A data science project based on 10 years (2015-2024) of historical weather data from 9 Indian cities.
It helps you find which cities are good to visit in which months, based on your preference (cool / warm / dry).

## Features

- **Data collection:** Daily weather from the Open-Meteo Archive API (free, no API key)
- **Clustering:** K-Means groups city-months with similar weather patterns
- **Classification:** Comparison of Random Forest, Logistic Regression, and a Baseline (time-based split)
- **Travel Comfort Index:** a 0-100 score based on temperature, rain, and wind, with custom weights
- **Trip Planner:** heatmaps and the top 3 months for each city, based on your preference (interactive)

## Project structure

```
weather-travel-advisor/
│── weather_travel_advisor.ipynb   # full project (in cells)
│── data/
│   ├── raw/                       # daily data from the API
│   └── processed/                 # monthly and clustered data
│── models/                        # trained classifiers (.pkl)
│── reports/                       # metrics and plots
│── README.md
```

## Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab (or the VS Code notebook editor)
- Internet connection (for data collection)

Install:

```bash
pip install pandas numpy requests scikit-learn joblib matplotlib seaborn ipywidgets jupyter
```

## How to run

1. Open `weather_travel_advisor.ipynb` in Jupyter.
2. Click **Kernel → Restart & Run All**, or run the cells from top to bottom.
3. Cell 6 downloads the data (this can take 1-2 minutes).
4. Cell 22 shows the trip planner. Cell 23 provides interactive sliders.

## Output files

| File | Description |
|---|---|
| `data/raw/weather_daily.csv` | Raw daily data |
| `data/processed/monthly.csv` | City-month-year summary |
| `data/processed/climatology_clusters.csv` | City-month clusters and comfort data |
| `models/classifiers.pkl` | Trained Random Forest models (cool, warm, dry) |
| `reports/metrics.csv` | Model comparison (accuracy, precision, recall, F1) |
| `reports/k_selection.png` | Elbow and silhouette plots |

## Methodology

- **Leakage prevention:** features use only the averages of past years.
- **Time-based split:** 2015-2022 for training, 2023-2024 for testing.
- **Label:** a rule based on the temperature range and rainy days for each preference.

## Limitations

- Open-Meteo data is reanalysis (ERA5), not exact station data.
- Labels are rule-based, not survey-based.
- Only 9 cities are included.
- Crowd, price, and festival data are not included.

## Future scope

- Live forecast integration
- Monsoon onset analysis
- Deployment as a Streamlit or web app

## Author

**SAMEER ANSARI**, Summer Training Project, Data Science
