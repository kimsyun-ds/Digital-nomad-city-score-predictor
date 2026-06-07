# 🌏 Digital Nomad City Score Predictor

> **English** | [한국어](#한국어)

A machine learning project that predicts a city's suitability score for digital nomads based on 19 urban features such as internet speed, safety, cost of living, and healthcare quality.

---

## Features

- 🤖 Trains and compares **3 ML models**: Linear Regression, Random Forest, MLP
- 🌐 **Bilingual CLI** — runs in English or Korean (`en` / `ko`)
- 🏙️ Predicts nomad score for any custom city input
- 📊 Generates feature importance and model comparison charts
- 🏆 Shows Top 10 recommended cities from the dataset

## Project Structure

```
nomad_predictor/
├── predictor.py          # Main script
├── cities_predict.csv    # Dataset (730 cities, 32 features)
├── README.md
├── requirements.txt
├── LICENSE
└── short_report.pdf
```

## Quickstart

```bash
# 1. Clone / download the project
git clone https://github.com/<your-username>/nomad-predictor.git
cd nomad-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python predictor.py
```

You will be prompted to select a language (`en` or `ko`). The program then trains the models and presents an interactive menu.

## Sample Output

```
=== Digital Nomad City Score Predictor ===
=== 디지털 노마드 도시 점수 예측기 ===
Select language / 언어를 선택하세요  [en / ko]: en

Loading dataset...
Training models (this may take a moment)...

── Model Performance ──
  Linear Regression       R²=0.3853  MSE=0.1344
  Random Forest           R²=0.4274  MSE=0.1252
  MLP Regressor           R²=0.2295  MSE=0.1685

✔  Best model: Random Forest  (R² = 0.4274)
```

## Dataset

- **Source**: Nomad List (city quality-of-life data)
- **Size**: 730 cities × 32 features
- **Target variable**: `nomad_score` (0–5 scale)
- **Selected features (19)**: startup score, safety, internet speed, healthcare, happiness, cost of living, and more.

## Tools & Libraries

| Library | Purpose | License |
|---------|---------|---------|
| pandas | Data loading & manipulation | BSD-3 |
| numpy | Numerical computation | BSD-3 |
| scikit-learn | ML models, preprocessing | BSD-3 |
| matplotlib | Chart generation | PSF |

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 한국어

# 🌏 디지털 노마드 도시 점수 예측기

인터넷 속도, 안전도, 생활비, 의료 수준 등 19가지 도시 특성을 기반으로, 머신러닝을 활용해 해당 도시의 디지털 노마드 적합도 점수를 예측하는 프로젝트입니다.

## 주요 기능

- 🤖 **3가지 ML 모델** 학습 및 성능 비교 (선형 회귀 / 랜덤 포레스트 / MLP)
- 🌐 **영한 이중언어 CLI** 지원 (`en` / `ko` 선택)
- 🏙️ 사용자 정의 도시 값 입력 → 노마드 점수 예측
- 📊 변수 중요도 및 모델 성능 비교 차트 자동 생성
- 🏆 데이터셋 내 추천 도시 TOP 10 출력

## 실행 방법

```bash
pip install -r requirements.txt
python predictor.py
```

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
