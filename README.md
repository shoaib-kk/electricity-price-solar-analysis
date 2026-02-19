# Battery Arbitrage using Conformal Prediction and Multi-Horizon Forecasting

This project implements an end to end battery arbitrage trading strategy, starting from collection data from the Australian National Electricity Market (NEM), cleaning, feature engineering, and ultimately using quantile regression and conformal prediction to forecast electricity prices and make uncertainty-aware charging and discharging decisions to yield a profit.

The system is evaluated against a strong baseline threshold policy using real electricity price data from the NEM.

---

## Overview

Electricity prices in the NEM are extremely volatile and often exhibit extreme spikes. Battery arbitrage involves charging when prices are low and discharging when prices are high to capture profit.

This project explores whether forecast-based strategies using uncertainty-aware conformal prediction can outperform simple heuristic threshold policies.

The pipeline includes:

- automated data ingestion from AEMO 

- cleaning, gap handling, feature engineering 

- exploratory time-series analysis and decomposition 

- baseline ARIMA forecasting and naive models 

- probabilistic forecasting using LightGBM quantile regression and conformal prediction 

- battery simulation with efficiency, capacity, and SOC constraints 

- backtesting and risk analysis including cumulative profit and drawdownn

---
## Sidenote of this project 
Although not the focus of this project, Data_collection and Data_cleaning provided the basis for this project to be completed and involved downloading the data from NEM as well as implementing different strategies to maximise memory efficiency. Further notes on this can be found in `Data_pipeline.md`.
## Strategies Implemented

### Baseline Threshold Policy
Charges when price is below the 30th percentile of the training distribution and discharges when above the 70th percentile.

This provides a simple but surprisingly effective strategy 

### Conformal Multi-Horizon Forecast Policy
Uses quantile regression and conformal prediction to generate calibrated prediction intervals across multiple forecast horizons.

Trading decisions are based on expected price movement relative to uncertainty-adjusted thresholds.

---

## Results Summary

| Policy | Total Profit | Max Drawdown | Equivalent Cycles | Profit per Cycle |
|------|-------------|-------------|-------------------|------------------|
| Baseline Threshold | $1639.88 | $89.08 | 127.7 | $12.84 |
| Conformal Multi-Horizon | $829.92 | $241.30 | 225.2 | $3.66 |

Key findings:

- Simple threshold policies remain extremely strong baselines
- Conformal policies introduce a trade-off between selectivity and profitability
- Uncertainty-aware trading improves decision control but does not guarantee higher raw profit
- Through changing paramters for conformal policies, profit per cycle can be brought up at the exchange of lowering equivalent cycles, introducing a tradeoff between the 2 

---

---

## Key Takeaways

This project demonstrates how uncertainty-aware forecasting can be integrated into decision systems, and highglights the significance of testing against simple strategies which while basic, often prove to be extremely effective.

As a personal takeaway, I also recognised the importance of good project design, having accidentally hyperfocusing on many modules which did not further my goals for this project and remain unused but are still included for user benefit. As such I find that it's important to build a general project pipeline before beginning a project to avoid getting off track.

---
## Things to add
AEMO API → data pipeline → trained model → prediction API → dashboard
model saving/loading

FastAPI prediction endpoint

dashboard (Streamlit)

scheduled retraining
---

## Author

Shoaib Kabir


