## Time Series Data Pipeline: Electricity Price & Demand (NEM – Victoria)

### Project Overview

This repository implements a full data pipeline for time-series modelling using Australian National Electricity Market (NEM) data. Modules were based on Victorian data so may require changes for other states.

The focus of this project is data engineering and modelling (soon to come). Data integrity and leakage preventage was a large portion of this project to make the data usable for forecasting later on 

### Data Source

Files are downloaded directly from AEMO, merged, de-duplicated, sorted, and exported into a single consolidated CSV before cleaning.

### Pipeline Structure

#### 1. Data Collection & Merging
 Downloads monthly AEMO CSV files with retry logic and outputs a single sorted CSV:
	- PRICE_AND_DEMAND_FULL_VIC1.csv

#### 2. Time Index Validation
e
Removes duplicate timestamps and reports unexpected gaps 

#### 3. Missing Data Handling
- RRP 
	- Short gaps are forward-filled only to avoid leakage.

- TOTALDEMAND
	- Short gaps are filled using time-based interpolation.

Remaining missing values are explicitly dropped.

#### 4. Unnatural values filtering

Rows outside of bounds (configurable) are removed, for example:

- implausible or invalid prices,
- implausible or invalid demand values.

This is conservative and intended to only remove obvious data errors and not actual market events.

#### 5. Time-Based Train / Test Split

- Splits data chronologically default: 80% train, 20% test
- Cleaning is applied independently to train and test sets

#### 6. Feature Engineering

Features are constructed to reflect time-series structure:

- Calendar features

- Cyclical encodings

- Lag features

- Rolling statistics (configurable)

Lagged features for the test set are generated using train history only ensuring no leakage.

#### 7. Final Outputs

The pipeline produces:

- CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv
- CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv

These datasets are:
- feature-complete,
- and ready for modelling.

### Modeling
#### Baseline 
- ARIMA is used as a baseline for this data to provide a method of comparison to other models that will be used later on. It  performs poorly on the high frequency price data and exhibits convergence issues due to strong intraday seasonality, price spikes, and the engineered features.

These limitations are the source of motivation for the use of feature-based machine learning models in later stages of the project






#### Quantile Forecast Calibration

- Initial LightGBM quantile models produced reasonable forecasts but exhibited systematic miscalibration:

	- 5% quantile covered only 2.9% of observations

	- 90% prediction intervals covered 93.8% 

- To address this, I implemented a validation-based quantile mapping calibration layer, which:

	- learned empirical bias corrections from a held-out validation set

	- adjusted the lower  and upper quantiles

	- preserved proper out-of-sample evaluation

	Results after calibration:

	- 5% quantile coverage improved from 2.9% → 4.0%

	- 90% interval coverage improved from 93.8% → 91.7%

	mean interval width reduced from 100.8 → 93.1 AUD/MWh

pinball loss remained stable
### Key highlights 

- No leakage: time order is respected everywhere.
- Explicit logging: gaps, drops, and assumptions are printed.
- Modularity: collection, cleaning, and feature engineering are separable.


I evaluated against proper baselines and discovered my ML point model does not beat persistence – which led me to focus on probabilistic forecasting instead.
### Scope & Limitations

This repository does not yet include forecasting models.

Feature choices are generic and meant as a strong baseline.

