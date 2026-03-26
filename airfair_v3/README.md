# AirFair Vista: Flight Price Prediction

AirFair Vista is an end-to-end machine learning project for predicting airline ticket prices with a clean two-service architecture. The project combines synthetic data generation, feature engineering, exploratory data analysis, model comparison, SHAP explainability, and an interactive Streamlit dashboard.

This version, `airfair_v3`, is the main project folder for GitHub and portfolio use.

## Highlights

- Predicts flight prices across 15 airlines and 15 cities
- Trains and compares 6 regression models
- Uses `TimeSeriesSplit` for chronological validation
- Includes SHAP explainability and feature-importance reports
- Generates EDA charts and business insights automatically
- Runs locally, with PowerShell helpers, or with Docker

## Project Structure

```text
airfair_v3/
|-- ml_pipeline/
|   |-- train.py
|   |-- src/
|   |   |-- config.py
|   |   |-- data_generator.py
|   |   |-- data_loader.py
|   |   |-- eda.py
|   |   |-- features.py
|   |   `-- trainer.py
|   |-- models/
|   `-- reports/
|-- streamlit_app/
|   |-- app.py
|   |-- pages/
|   `-- utils/
|-- docker-compose.yml
|-- Makefile
|-- tasks.ps1
`-- README.md
```

## Architecture

The project is split into two parts:

- `ml_pipeline/` handles data generation, preprocessing, EDA, feature engineering, training, evaluation, and artifact export
- `streamlit_app/` loads trained artifacts and serves the frontend dashboard

This keeps training and inference cleanly separated while preserving feature parity between both stages.

## Streamlit Pages

| Page | Purpose |
|---|---|
| Predict | Live ticket price prediction |
| EDA | Visual analysis of the dataset and trends |
| Features | Explanation of engineered model features |
| Models | Model comparison, metrics, and SHAP plots |
| About | Project overview and usage guide |

## Model Outputs

After training, the pipeline saves:

- `model.pkl`
- `encoders.pkl`
- `features.pkl`
- `model_meta.json`
- EDA plots in `ml_pipeline/reports/`
- Model comparison, residual, and SHAP plots

## Quick Start

### Local

```bash
cd airfair_v3
make install
make train
make app
```

Open:

```text
http://localhost:8501
```

### Windows PowerShell

```powershell
cd airfair_v3
.\tasks.ps1 install
.\tasks.ps1 train
.\tasks.ps1 app
```

### Docker

```bash
cd airfair_v3
docker-compose up --build
```

Or:

```bash
cd airfair_v3
make up
```

## Re-Train Options

```bash
make train-large
make train-force
make train-skip-eda
make retrain
```

PowerShell:

```powershell
.\tasks.ps1 train-large
.\tasks.ps1 train-force
.\tasks.ps1 train-skip-eda
.\tasks.ps1 retrain
```

## Tech Stack

- Python
- Pandas and NumPy
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- Matplotlib and Seaborn
- Streamlit
- Plotly
- Docker and Docker Compose

## Why This Project

This project was built to demonstrate a full machine learning workflow for a real-world pricing problem, from raw data and feature engineering to model deployment and a user-facing application.

## GitHub Push

Your GitHub remote is already connected:

```text
https://github.com/jay2592004/flight-price-prediction.git
```

To upload only this README update:

```bash
git add airfair_v3/README.md
git commit -m "Improve airfair_v3 README for GitHub"
git push origin main
```

If your branch is not `main`, check it with:

```bash
git branch --show-current
```
