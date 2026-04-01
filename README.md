# 🛒 Store Sales Time-Series Forecasting — ANN Regression Pipeline

## 🎯 Overview

This project implements a **Deep Learning regression pipeline** using Artificial Neural Networks (ANN) to forecast grocery sales for **Corporación Favorita**, a large Ecuadorian-based retailer. The model processes millions of transactions across **54 stores** and **33 product families**, integrating external factors like oil prices, holiday events, and store metadata to minimize **RMSLE** (Root Mean Squared Logarithmic Error).

The solution moves beyond simple time-series averages by utilizing a multi-layered perceptron to capture complex non-linear relationships between inventory promotions, regional economic indicators (oil), and localized holiday effects.

- **Competition:** [Store Sales - Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Target Metric:** RMSLE (Root Mean Squared Logarithmic Error)
- **Approach:** Multi-source Data Merging → Feature Engineering → Deep ANN Regression

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Core ML & Metrics | scikit-learn (StandardScaler, LabelEncoder, MSLE) |
| Data Engineering | pandas, numpy, kagglehub |
| Visualization | matplotlib, seaborn |
| Hardware | NVIDIA Tesla T4 GPU (Kaggle Accelerator) |

---

## 📈 Pipeline Phases

### 1. Multi-Source Data Integration

The dataset was distributed across seven CSV files. A master training set was built by performing complex joins:

- **Relational Merging:** Joined `train.csv` (3M+ rows) with `stores.csv`, `oil.csv`, and `transactions.csv` on `date` and `store_nbr` keys.
- **Holiday Hierarchy:** Implemented a priority-based merge for National, Regional, and Local holidays to ensure store-specific event accuracy.
- **Temporal Alignment:** Converted date strings to datetime objects and synchronized the 2013–2017 timeline.

### 2. Feature Engineering & Cleaning

- **Missing Value Recovery:** Used `ffill()` and `bfill()` on oil prices (`dcoilwtico`) to handle weekends and market holidays.
- **Boolean Indicators:** Created an `is_holiday` flag by consolidating three different holiday granularity levels.
- **Encoding Strategy:** Applied `pd.get_dummies` for categorical variables like `family`, `city`, `state`, and `type` to prepare data for neural network consumption.
- **Feature Scaling:** Utilized `StandardScaler` on numerical inputs (`onpromotion`, `transactions`, `dcoilwtico`) to accelerate ANN convergence.

### 3. Model Architecture (ANN)

The model is a sequential Artificial Neural Network built for high-dimensional regression:

| Layer | Details |
|---|---|
| Input Layer | Handles wide feature set from one-hot encoding |
| Hidden Layers | Multiple dense layers with **ReLU** activation |
| Output Layer | Single neuron with **Linear** activation |
| Optimizer | Adam — Loss: Mean Squared Error |

---

## 🏆 Key Results

### Performance by Product Category

| Category | RMSLE | Status |
|---|---|---|
| LAWN AND GARDEN | 0.1458 | Highest Accuracy ⭐ |
| LADIESWEAR | 0.2142 | Very Predictable |
| PRODUCE | 0.7348 | High Volatility ⚠️ |
| LIQUOR, WINE, BEER | 0.7760 | Hardest to Predict |

> **Insight:** Categories like "Lawn and Garden" show steady seasonal trends, while "Liquor" and "Produce" are subject to high-frequency spikes and perishability factors that challenge the model.

---

## 📂 Repository Structure

```
store-sales-forecasting/
│
├── ann-impletetaion-on-dataset.ipynb   # Core ANN pipeline & analysis
├── README.md                           # Project documentation
│
└── data/  (Auto-fetched via KaggleHub)
    ├── train.csv                       # 3,000,888 training samples
    ├── oil.csv                         # Daily crude oil prices
    ├── holidays_events.csv             # National & Local events
    └── stores.csv                      # Store metadata (city, state, type)
```

---

## 🚀 Future Improvements

- [ ] **Lag Features:** Implement t-7 and t-30 sales lags to capture weekly and monthly seasonality.
- [ ] **Log Transformation:** Apply `log1p` to the target `sales` variable before training to better align with the RMSLE metric.
- [ ] **LSTM Integration:** Transition from a standard ANN to a Long Short-Term Memory (LSTM) network to better process temporal sequences.
- [ ] **Rolling Statistics:** Add 7-day rolling means for oil prices to smooth out daily volatility.

---

## 📧 Contact

**Prashant Shukla**

- 📧 Email: prashantshukla8851@gmail.com
- 💼 LinkedIn: [Prashant Shukla](https://www.linkedin.com/in/prashant-shukla)
- 🔗 GitHub: [@pr4sh4nt-shukla](https://github.com/pr4sh4nt-shukla)

---

⭐ *If this forecasting approach helped your research, please give it a star!* ⭐
