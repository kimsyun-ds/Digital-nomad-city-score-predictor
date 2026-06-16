"""
Digital Nomad City Score Predictor
====================================
Predicts a city's nomad suitability score using machine learning.
디지털 노마드 도시 점수 예측기 - 머신러닝 기반 도시 적합도 예측

Author  : OSS Final Project
License : MIT
"""

# ── External libraries ──────────────────────────────────────────────────────
import pandas as pd          # Data loading and manipulation (CSV → DataFrame)
import numpy as np           # Numerical operations (array sorting, linspace)
import matplotlib            # Base matplotlib package; backend must be set before pyplot
matplotlib.use("Agg")        # Use non-interactive backend — safe for all OS environments
                             # (prevents GUI window errors when saving charts)
import matplotlib.pyplot as plt          # Chart drawing: bar charts, subplots, labels
import matplotlib.font_manager as fm     # Font management for chart text rendering
import warnings              # Suppress non-critical warnings from sklearn/matplotlib
import os                    # File path operations: os.path.join, os.path.dirname
import sys                   # System-level access (reserved for future use / error handling)

warnings.filterwarnings("ignore")   # Hide deprecation/convergence warnings during model training

# ── scikit-learn: machine learning utilities ────────────────────────────────
from sklearn.model_selection import train_test_split   # Split dataset into train/test sets
from sklearn.preprocessing import StandardScaler       # Normalize features (mean=0, std=1)
from sklearn.linear_model import LinearRegression      # Baseline regression model
from sklearn.ensemble import RandomForestRegressor     # Best-performing ensemble model
from sklearn.neural_network import MLPRegressor        # Multi-layer perceptron regression model
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics


# ─────────────────────────────────────────────
# Language / 언어 설정
# ─────────────────────────────────────────────

LANG = "en"  # Global language setting; changed to "ko" if user selects Korean at startup

# Bilingual message dictionary
# Structure: MESSAGES[message_id][language_code] → string
# Supports "en" (English) and "ko" (Korean)
MESSAGES = {
    "welcome": {
        "en": "\n=== Digital Nomad City Score Predictor ===",
        "ko": "\n=== 디지털 노마드 도시 점수 예측기 ===",
    },
    "select_lang": {
        # Shown in both languages since user hasn't selected one yet
        "en": "Select language / 언어를 선택하세요  [en / ko]: ",
        "ko": "Select language / 언어를 선택하세요  [en / ko]: ",
    },
    "loading": {
        "en": "Loading dataset...",
        "ko": "데이터셋을 불러오는 중...",
    },
    "training": {
        "en": "Training models (this may take a moment)...",
        "ko": "모델 학습 중 (잠시 기다려 주세요)...",
    },
    "model_results": {
        "en": "\n── Model Performance ──",
        "ko": "\n── 모델 성능 비교 ──",
    },
    "best_model": {
        # {name} and {r2} are format placeholders filled in by t()
        "en": "\n✔  Best model: {name}  (R² = {r2:.4f})",
        "ko": "\n✔  최적 모델: {name}  (R² = {r2:.4f})",
    },
    "menu": {
        "en": (
            "\n[1] Predict score for a new city"
            "\n[2] Show top 10 recommended cities"
            "\n[3] Show feature importance chart"
            "\n[4] Show model comparison chart"
            "\n[q] Quit"
            "\nChoice: "
        ),
        "ko": (
            "\n[1] 새 도시 점수 예측"
            "\n[2] 추천 도시 TOP 10 보기"
            "\n[3] 변수 중요도 차트 보기"
            "\n[4] 모델 성능 비교 차트 보기"
            "\n[q] 종료"
            "\n선택: "
        ),
    },
    "enter_feature": {
        # {feat} = feature name, {avg} = dataset average for that feature
        "en": "  Enter {feat} (avg {avg:.2f}): ",
        "ko": "  {feat} 입력 (평균 {avg:.2f}): ",
    },
    "predicted": {
        # {score} = predicted nomad score (float)
        "en": "\n🏙  Predicted Nomad Score: {score:.3f}",
        "ko": "\n🏙  예측된 노마드 점수: {score:.3f}",
    },
    "top10_header": {
        "en": "\n── Top 10 Cities for Digital Nomads ──",
        "ko": "\n── 디지털 노마드 추천 도시 TOP 10 ──",
    },
    "chart_saved": {
        # {path} = file path where the chart PNG was saved
        "en": "Chart saved → {path}",
        "ko": "차트 저장됨 → {path}",
    },
    "invalid": {
        "en": "Invalid input, please try again.",
        "ko": "잘못된 입력입니다. 다시 시도해 주세요.",
    },
    "bye": {
        "en": "Goodbye! 👋",
        "ko": "종료합니다! 👋",
    },
}


def t(key, **kwargs):
    """
    Translation helper — returns the message string for the current LANG.
    Supports format-string substitution via keyword arguments.
    Example: t("best_model", name="Random Forest", r2=0.4274)
    """
    msg = MESSAGES[key][LANG]
    return msg.format(**kwargs) if kwargs else msg


# ─────────────────────────────────────────────
# Data & Feature config
# ─────────────────────────────────────────────

# Build absolute path to the CSV so the script works from any working directory
DATA_FILE = os.path.join(os.path.dirname(__file__), "cities_predict.csv")

# 19 features selected from the original 32 columns
# Selection criterion: Random Forest feature importance >= 0.01
# Covers 4 categories: economic, infrastructure, social, lifestyle
SELECTED_FEATURES = [
    "startup_score", "female_friendly", "air_quality_(year-round)",
    "coca-cola", "internet", "coffee", "nightlife",
    "1br_studio_rent_in_center", "airbnb_(monthly)",
    "cost_of_living_for_local", "cost_of_living_for_expat",
    "fun", "friendly_to_foreigners", "safety", "racial_tolerance",
    "cashless_society", "peace", "healthcare", "happiness",
]

# Bilingual display labels for each feature
# Used in CLI prompts and chart axis labels
FEATURE_LABELS = {
    "startup_score":               {"en": "Startup Score",              "ko": "스타트업 점수"},
    "female_friendly":             {"en": "Female Friendly",            "ko": "여성 친화도"},
    "air_quality_(year-round)":    {"en": "Air Quality",                "ko": "공기질"},
    "coca-cola":                   {"en": "Coca-Cola Price (USD)",      "ko": "콜라 가격 (USD)"},
    "internet":                    {"en": "Internet Speed (Mbps)",      "ko": "인터넷 속도 (Mbps)"},
    "coffee":                      {"en": "Coffee Price (USD)",         "ko": "커피 가격 (USD)"},
    "nightlife":                   {"en": "Nightlife",                  "ko": "나이트라이프"},
    "1br_studio_rent_in_center":   {"en": "1BR Rent in Center (USD)",   "ko": "시내 1룸 월세 (USD)"},
    "airbnb_(monthly)":            {"en": "Airbnb Monthly (USD)",       "ko": "에어비앤비 월세 (USD)"},
    "cost_of_living_for_local":    {"en": "Cost of Living (Local)",     "ko": "현지인 생활비"},
    "cost_of_living_for_expat":    {"en": "Cost of Living (Expat)",     "ko": "외국인 생활비"},
    "fun":                         {"en": "Fun Score",                  "ko": "즐거움 점수"},
    "friendly_to_foreigners":      {"en": "Friendly to Foreigners",     "ko": "외국인 친화도"},
    "safety":                      {"en": "Safety",                     "ko": "안전도"},
    "racial_tolerance":            {"en": "Racial Tolerance",           "ko": "인종 관용도"},
    "cashless_society":            {"en": "Cashless Society",           "ko": "무현금 사회"},
    "peace":                       {"en": "Peace",                      "ko": "평화도"},
    "healthcare":                  {"en": "Healthcare",                 "ko": "의료 수준"},
    "happiness":                   {"en": "Happiness",                  "ko": "행복도"},
}


def label(feat):
    """Return the bilingual display name for a given feature key."""
    return FEATURE_LABELS[feat][LANG]


# ─────────────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────────────

def load_data():
    """
    Load the dataset from CSV and extract features, target, and metadata.
    Returns:
        X    : DataFrame of 19 selected input features
        y    : Series of nomad_score (target variable, range 0–5)
        meta : DataFrame with city name, country, and region columns
        df   : Full original DataFrame (used for Top 10 display)
    """
    df = pd.read_csv(DATA_FILE)
    X = df[SELECTED_FEATURES]                                   # Input features only
    y = df["nomad_score"]                                       # Target variable
    meta = df[["place_slug", "country", "region"]].copy()      # City metadata
    return X, y, meta, df


# ─────────────────────────────────────────────
# Train models
# ─────────────────────────────────────────────

def train_models(X, y):
    """
    Standardize features, split data, and train 3 regression models.
    Evaluates each model on the test set using R² and MSE.
    Returns:
        results : dict — {model_name: {model, mse, r2, y_test, y_pred}}
        scaler  : fitted StandardScaler (reused during prediction)
        X_train, X_test, y_train, y_test : split arrays
    """
    # Normalize features: mean=0, std=1 (required for fair model comparison)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80% training, 20% testing; fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Define the three models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "MLP Regressor":     MLPRegressor(
                                 hidden_layer_sizes=(100,), max_iter=1000,
                                 activation="relu", random_state=42
                             ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)          # Train on training set
        y_pred = model.predict(X_test)       # Predict on unseen test set
        results[name] = {
            "model":  model,
            "mse":    mean_squared_error(y_test, y_pred),   # Lower is better
            "r2":     r2_score(y_test, y_pred),             # Higher is better (max 1.0)
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results, scaler, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────

def chart_feature_importance(rf_model, output_dir="."):
    """
    Generate a horizontal bar chart of Random Forest feature importances.
    Bars are sorted ascending so the most important feature appears at the top.
    Chart title and axis label switch language based on LANG.
    Saves the chart as 'feature_importance.png' in output_dir.
    """
    importances = rf_model.feature_importances_         # Array of importance scores (sum=1)
    feat_labels = [label(f) for f in SELECTED_FEATURES] # Bilingual feature names
    idx = np.argsort(importances)                        # Sort indices low → high

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        [feat_labels[i] for i in idx],                          # Feature names (sorted)
        importances[idx],                                        # Importance values (sorted)
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(idx))), # Gradient color scheme
        edgecolor="white",
    )
    ax.set_xlabel("Importance" if LANG == "en" else "중요도", fontsize=11)
    ax.set_title(
        "Feature Importance (Random Forest)" if LANG == "en"
        else "변수 중요도 (랜덤 포레스트)",
        fontsize=13, fontweight="bold"
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)  # Show values on each bar
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)   # Save at 150 DPI for clear print/presentation quality
    plt.close()                  # Release memory — important when Agg backend is used
    return path


def chart_model_comparison(results, output_dir="."):
    """
    Generate a side-by-side bar chart comparing R² and MSE across all 3 models.
    Left subplot: R² Score (higher = better)
    Right subplot: MSE — Mean Squared Error (lower = better)
    Saves the chart as 'model_comparison.png' in output_dir.
    """
    names = list(results.keys())
    r2_vals  = [results[n]["r2"]  for n in names]
    mse_vals = [results[n]["mse"] for n in names]

    x = np.arange(len(names))   # x positions for bar groups
    width = 0.35                 # Bar width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Left: R² Score ──
    bars1 = ax1.bar(x, r2_vals, width, color=["#4C72B0", "#DD8452", "#55A868"],
                    edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=12, ha="right")
    ax1.set_ylabel("R² Score")
    ax1.set_title("R² Score by Model" if LANG == "en" else "모델별 R² 점수",
                  fontweight="bold")
    ax1.set_ylim(0, 1)                                      # Fix y-axis 0–1 for R²
    ax1.bar_label(bars1, fmt="%.4f", padding=3)             # Show exact values on bars

    # ── Right: MSE ──
    bars2 = ax2.bar(x, mse_vals, width, color=["#4C72B0", "#DD8452", "#55A868"],
                    edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=12, ha="right")
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE by Model" if LANG == "en" else "모델별 MSE",
                  fontweight="bold")
    ax2.bar_label(bars2, fmt="%.4f", padding=3)

    plt.suptitle(
        "Model Performance Comparison" if LANG == "en" else "모델 성능 비교",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")  # bbox_inches="tight" prevents label clipping
    plt.close()
    return path


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict_new_city(model, scaler, mean_values):
    """
    Interactively prompt the user to enter values for each of the 19 features.
    Empty input is automatically filled with the dataset average for that feature.
    The input is standardized using the same scaler fitted during training,
    then passed to the best model to produce a predicted nomad score.
    """
    print()
    user_vals = {}
    for feat in SELECTED_FEATURES:
        while True:
            try:
                # Show feature name (bilingual) and dataset average as a hint
                raw = input(t("enter_feature", feat=label(feat), avg=mean_values[feat]))
                # If user presses Enter without typing, use dataset average
                user_vals[feat] = float(raw) if raw.strip() else mean_values[feat]
                break
            except ValueError:
                # Re-prompt if input cannot be converted to float
                print(t("invalid"))

    # Build a single-row DataFrame in the correct feature order
    row = pd.DataFrame([user_vals])[SELECTED_FEATURES]
    # Apply the same StandardScaler used during training (not fit again)
    row_scaled = scaler.transform(row)
    score = model.predict(row_scaled)[0]   # Extract scalar from 1-element array
    print(t("predicted", score=score))
    return score


# ─────────────────────────────────────────────
# Top-10 cities
# ─────────────────────────────────────────────

def show_top10(df):
    """
    Display the top 10 cities ranked by actual nomad_score from the dataset.
    Column headers switch language based on LANG.
    """
    top = df.nlargest(10, "nomad_score")[["place_slug", "country", "nomad_score"]]
    print(t("top10_header"))
    # Print column header row (language-aware)
    print(f"  {'City':<30} {'Country':<20} {'Score':>6}" if LANG == "en"
          else f"  {'도시':<30} {'국가':<20} {'점수':>6}")
    print("  " + "─" * 56)
    for _, row in top.iterrows():
        # Convert slug format (e.g. "chiang-mai-thailand") to title case
        city = row["place_slug"].replace("-", " ").title()
        print(f"  {city:<30} {row['country']:<20} {row['nomad_score']:>6.3f}")


# ─────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────

def main():
    """
    Entry point for the bilingual command-line interface.
    Flow:
      1. Ask user to select language (en / ko)
      2. Load dataset and compute feature means
      3. Train all 3 models and select the best by R²
      4. Display interactive menu and loop until user quits
    """
    global LANG   # Allow inner functions to reference the updated language setting

    # Display welcome message in both languages before language is selected
    print(MESSAGES["welcome"]["en"] + "\n" + MESSAGES["welcome"]["ko"])
    lang_input = input(MESSAGES["select_lang"]["en"]).strip().lower()
    if lang_input in ("ko", "en"):
        LANG = lang_input   # Set global language; defaults to "en" if invalid input

    print(t("loading"))
    X, y, meta, df = load_data()
    mean_values = X.mean()   # Precompute feature averages for prediction auto-fill

    print(t("training"))
    results, scaler, *_ = train_models(X, y)

    # Automatically select the model with the highest R² on the test set
    best_name  = max(results, key=lambda n: results[n]["r2"])
    best_model = results[best_name]["model"]

    # Print performance summary for all 3 models
    print(t("model_results"))
    for name, res in results.items():
        print(f"  {name:<22}  R²={res['r2']:.4f}  MSE={res['mse']:.4f}")
    print(t("best_model", name=best_name, r2=results[best_name]["r2"]))

    # Charts and output files are saved to the same directory as predictor.py
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Interactive menu loop ──
    while True:
        choice = input(t("menu")).strip().lower()

        if choice == "1":
            # Prompt user for feature values → predict nomad score
            predict_new_city(best_model, scaler, mean_values)

        elif choice == "2":
            # Show top 10 cities ranked by actual nomad_score
            show_top10(df)

        elif choice == "3":
            # Generate and save feature importance bar chart (PNG)
            rf = results["Random Forest"]["model"]
            path = chart_feature_importance(rf, output_dir)
            print(t("chart_saved", path=path))

        elif choice == "4":
            # Generate and save R²/MSE model comparison chart (PNG)
            path = chart_model_comparison(results, output_dir)
            print(t("chart_saved", path=path))

        elif choice == "q":
            print(t("bye"))
            break   # Exit the menu loop and end the program

        else:
            print(t("invalid"))   # Re-display menu on unrecognized input


if __name__ == "__main__":
    main()   # Only run when executed directly, not when imported as a module
