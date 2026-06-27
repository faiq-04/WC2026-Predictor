# ⚽ FIFA World Cup 2026 Predictor

> A machine learning application that predicts FIFA World Cup 2026 match outcomes using a Random Forest classifier trained on **49,433 international football matches** spanning 1872–2026.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wc2026-predictor.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

##  Live Demo

 **[Try the app here](https://wc2026-predictor.streamlit.app)**

---

##  App Preview

| Tournament Simulator | Head-to-Head Predictor |
|---|---|
| Pick 8 teams, simulate the full knockout bracket | Any two teams → win probabilities |

| Elo Rankings | Model Performance |
|---|---|
| Live Elo leaderboard for all WC 2026 teams | Confusion matrix, feature importances |

---

##  How It Works

### Dataset
- **Source:** [Kaggle — International Football Results 1872–2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **Size:** 49,433 matches after cleaning (training uses 1998–2026)
- **Features used:** Match date, home/away teams, scores, tournament type

### Feature Engineering
For every match, the following features are computed using **only historical data before that match** (no data leakage):

| Feature | Description |
|---|---|
| `elo_diff` | Difference in Elo ratings between teams |
| `win_rate_diff` | Rolling win rate gap (last 10 matches) |
| `attack_diff` | Avg goals scored difference (last 10 matches) |
| `defense_diff` | Avg goals conceded difference (last 10 matches) |
| `h2h_home_win_rate` | Historical head-to-head win rate |
| `rest_diff` | Days since last match (fatigue proxy) |
| `is_world_cup` | Whether the match is a World Cup fixture |

### Elo Rating System
- Initial rating: **1500** for all teams
- K-factor: **20** (moderate sensitivity to results)
- Updated after every match, capturing long-term team strength

### Model
```
RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 8,
    class_weight = 'balanced',   # handles Home Win / Draw / Away Win imbalance
    random_state = 42
)
```

**3-class prediction:** Home Win · Draw · Away Win

---

##  Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **54.2%** |
| CV Mean Accuracy | **53.1% ± 3.5%** |
| Home Win F1 | 0.66 |
| Away Win F1 | 0.55 |
| Draw F1 | 0.31 |

> 54% accuracy is competitive for football prediction — the sport is inherently unpredictable. Commercial models rarely exceed 58–60%.

### Feature Importances
```
Elo difference       ████████████████████  48.0%
Defense difference   ████████              18.8%
Win rate difference  ████                  10.3%
Attack difference    ████                   9.1%
H2H win rate         ███                    7.5%
Rest difference      ██                     5.9%
Is World Cup         ░                      0.4%
```

---

## Predicted Results (Default QF Bracket)

```
QUARTER-FINALS
  Argentina    vs Croatia      →  Argentina  (85.3%)
  France       vs England      →  France     (61.7%)
  Spain        vs Netherlands  →  Spain      (66.2%)
  Brazil       vs Portugal     →  Brazil     (55.6%)

SEMI-FINALS
  Argentina    vs France       →  Argentina  (59.6%)
  Spain        vs Brazil       →  Spain      (59.1%)

FINAL
  Argentina    vs Spain        →  Argentina  (58.6%)

🏆 CHAMPION: ARGENTINA
```

---

##  Project Structure

```
WC2026-Predictor/
│
├── app.py                  # Streamlit application (4 pages)
├── predictor_V2.ipynb      # Full Jupyter notebook with analysis
├── results_cleaned.csv     # Cleaned dataset (49,433 matches)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/faiq-04/WC2026-Predictor.git
cd WC2026-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## 🔍 App Pages

### 🏆 Tournament Simulator
- Select any 8 teams for the quarter-finals
- Simulates QF → SF → Final automatically
- Draw probability redistributed proportionally in knockout rounds
- Shows win confidence for every match

### ⚔️ Head-to-Head Predictor
- Pick any two teams from 48 WC 2026 nations
- Returns Home Win / Draw / Away Win probabilities
- Visual probability bar + Elo and form comparison table

### 📊 Model Performance
- Interactive confusion matrix
- Feature importance chart
- Full classification report per class

### 📈 Elo Rankings
- Live Elo leaderboard for all WC 2026 teams
- Filterable bar chart with top-N slider
- Full sortable rankings table

---

## 📝 Data Cleaning Notes

The raw Kaggle dataset had a **mixed date format** issue:
- Most rows: `YYYY-MM-DD`
- Some rows: `DD-MM-YY` (legacy format)

A custom `clean_dates()` function detects the actual rollover point around 1999/2000 rather than trusting pandas' generic 2-digit year inference, which would have mislabelled ~7,400 rows.

The 44 rows with missing scores (upcoming 2026 fixtures) were preserved as a prediction set rather than dropped.

---

## 🙋 Author

**Faiq Ahmed**

- Built as a personal ML project for FIFA World Cup 2026
- Dataset: Kaggle — International Football Results 1872–2026
- Model: scikit-learn Random Forest
- Deployed via: Streamlit Community Cloud

---

## ⚠️ Disclaimer

This is a machine learning project for educational purposes. Predictions are based on historical data and statistical patterns — they are not financial or betting advice. Football is inherently unpredictable!
