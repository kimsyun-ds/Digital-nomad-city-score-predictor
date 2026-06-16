# 🌏 Digital Nomad City Score Predictor

A bilingual (English / Korean) machine learning tool that predicts a city's suitability score for digital nomads based on 19 urban features.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)

---

## What It Does

Enter values for 19 city features (internet speed, safety, cost of living, etc.) and the program predicts a **nomad suitability score (0–5)** using a trained Random Forest model. It also shows a Top 10 city ranking and exports visualization charts.

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/nomad-predictor.git
cd nomad-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python predictor.py
```

> Make sure `cities_predict.csv` is in the same folder as `predictor.py`.

---

## Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 1.5.0 | Data loading and manipulation |
| numpy | ≥ 1.23.0 | Numerical operations |
| scikit-learn | ≥ 1.2.0 | ML models and preprocessing |
| matplotlib | ≥ 3.6.0 | Chart generation |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Usage

On startup, select your language:
```
=== Digital Nomad City Score Predictor ===
=== 디지털 노마드 도시 점수 예측기 ===
Select language / 언어를 선택하세요  [en / ko]: en
```

Then choose from the menu:
```
[1] Predict score for a new city
[2] Show top 10 recommended cities
[3] Show feature importance chart
[4] Show model comparison chart
[q] Quit
```

### Predicting a new city

Enter values for each feature when prompted. Press **Enter** to use the dataset average automatically.

```
  Enter Internet Speed (Mbps) (avg 47.82): 80
  Enter Safety (avg 3.71): 4.5
  Enter Startup Score (avg 3.45):       ← uses average
  ...

🏙  Predicted Nomad Score: 4.123
```

### Top 10 cities (from dataset)

```
── Top 10 Cities for Digital Nomads ──
  City                           Country              Score
  ────────────────────────────────────────────────────────
  Chiang Mai Thailand            Thailand              4.970
  Budapest Hungary               Hungary               4.930
  Bangkok Thailand               Thailand              4.900
  Ho Chi Minh City Vietnam       Vietnam               4.830
  Canggu Bali Indonesia          Indonesia             4.800
  ...
```

---

## Project Structure

```
nomad_predictor/
├── predictor.py            # Main script — bilingual CLI + ML pipeline
├── cities_predict.csv      # Dataset: 730 cities × 32 features
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── short_report.pdf        # Project report
├── feature_importance.png  # Auto-generated (menu option 3)
└── model_comparison.png    # Auto-generated (menu option 4)
```

---

## Model Performance

Three models were trained and compared on an 80/20 train/test split:

| Model | R² Score | MSE |
|-------|----------|-----|
| Linear Regression | 0.3853 | 0.1344 |
| **Random Forest ✔** | **0.4274** | **0.1252** |
| MLP Regressor | 0.2295 | 0.1685 |

Random Forest was selected automatically as the best model.

---

## Dataset

- **Source**: Kaggle — *World Regions and Nomadscore* (based on Nomads.com / Nomad List)
- **Size**: 730 cities × 32 features
- **Target**: `nomad_score` (0–5 scale, community-rated)
- **Features used**: 19 (selected by feature importance ≥ 0.01)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this project with attribution.

---

*Sejong University · Introduction to Open Source Software · 2026*
```
