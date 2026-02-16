# Data pipeline for battery arbitrage (Downloader, Merger, Cleaner and feature engineer-er?)

This is an overview of `Data_collection.py` and `Data_cleaning.py`, which downloads monthly AEMO NEM CSVs for a chosen state (defaulting to Victoria), merges them into a single dataset safely using chunks, and then cleans the time series and produces time based train/test datasets with leakage safe lag features.

This pipeline is designed to be: 
- Robust (retries and checks for correct CSVs downloaded)
- Memory safe (reading using chunks and a SQLite database focused merge)
- Time series correct(gap handling and time-based split)
- Model ready and leakage safe (feature engineering with train-history for test so lags exist without using future data)

---

## Overview

### Download monthly files from AEMO
Downloads AEMO files from https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data

Key behaviour:
- skips redownloading files if a complete file already exists 
- retries failed downloads 
- rejects suspicious files (too small or containing html)

### Merge Monthly files into one dataset
Merges all monthly files into  `PRICE_AND_DEMAND_FULL_VIC1.csv` using SQLite which allows deduplication, sorting my timestamp, and exporting in chunks rather than the whole thing at once 

### Clean the time series 
Cleans the dataset to prepare it for modelling:
- sort index
- remove duplicates 
- reindex onto 5 minute spaces 
- fix small gaps
- remove RRP and TOTALDEMAND Nans 
- remove improbably values 

# Time-based train/test split and feature engineering 
- split chronologically 
- add:
    - cycling features 
    - lag features 
    - rolling stats 
- leakage safe features:
    - building test features using the end of the train history so you don't have missing data for that section 

outputs `CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv` and `CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv`
