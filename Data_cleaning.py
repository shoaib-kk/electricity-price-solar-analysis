import pandas as pd
import numpy as np

def shorten_time_increment(df, time_freq: int = 60) -> pd.DataFrame:

    factor = time_freq // 5
    if time_freq % 5 != 0 or time_freq < 5:
        raise ValueError("time_freq must be a multiple of 5.")
    
    df = df.sort_index()
    df_reduced = df.resample(f"{factor * 5}T").mean()
    return df_reduced

def time_based_train_test_split(df, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def missing_data_percentage(df, column_name: str):
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' does not exist in the DataFrame.")
    total_points = len(df)
    if total_points == 0:
        return 0.0
    missing_points = df[column_name].isna().sum()
    percentage_missing = (missing_points / total_points) * 100
    return percentage_missing


def check_time_index(df, expected_freq_minutes=5):
    # check for gaps and duplicates in time index

    df = df.sort_index()
    initial_rows = len(df)
    df = df[~df.index.duplicated(keep="first")]
    duplicates_removed = initial_rows - len(df)

    if duplicates_removed > 0:
        print(
            f"Removed {duplicates_removed} duplicate timestamps ({duplicates_removed/initial_rows*100:.2f}%)."
        )

    # list of time gaps between rows 
    time_gaps = df.index.to_series().diff()

    mode_gap = time_gaps.mode()[0] if len(time_gaps) > 0 else None

    # need to convert to Timedelta for comparison
    expected_gap = pd.Timedelta(minutes=expected_freq_minutes)

    if mode_gap and mode_gap != expected_gap:
        print(
            f" expected {expected_freq_minutes}min intervals, but most common gap is {mode_gap}"
        )
    else:
        print(f"Frequency validated: {expected_freq_minutes}min intervals")

    return df


def check_gaps(df, rrp_ffill_limit: int=2, demand_interp_limit: int=4, expected_freq_minutes: int=5):
    initial_rows = len(df)
    missing_rrp_vals= df["RRP"].isna().sum()
    missing_demand_vals = df["TOTALDEMAND"].isna().sum()
    print(
        f"Before filling: RRP has {missing_rrp_vals} missing values, "
        f"  TOTALDEMAND has {missing_demand_vals} missing values"
    )

    max_rrp_gap_minutes = rrp_ffill_limit * expected_freq_minutes
    max_demand_gap_minutes = demand_interp_limit * expected_freq_minutes
    print(f"Filling RRP gaps up to {max_rrp_gap_minutes}min (forward fill)")
    print(f"  Filling TOTALDEMAND gaps up to {max_demand_gap_minutes}min (time interpolation)")

    # forward fill to prevent leakage, limit to rrp_ffill_limit, as RRP can be constant for short periods
    df["RRP"] = df["RRP"].ffill(limit=rrp_ffill_limit)

    # time-based interpolation, limit to demand_interp_limit
    df["TOTALDEMAND"] = df["TOTALDEMAND"].interpolate(method="time", limit=demand_interp_limit)

    missing_rrp_after = df["RRP"].isna().sum()
    missing_demand_after = df["TOTALDEMAND"].isna().sum()
    print(
        f"  After filling: RRP still has {missing_rrp_after} missing values, "
        f"  TOTALDEMAND still has {missing_demand_after} missing values"
    )

    df = df.dropna(subset=["RRP", "TOTALDEMAND"])
    rows_dropped = initial_rows - len(df)

    if rows_dropped > 0:
        print(
            f"Dropped {rows_dropped} rows with missing values ({rows_dropped/initial_rows*100:.2f}% of data)."
        )
    else:
        print(f"No rows dropped; all gaps filled successfully.")

    return df


def remove_improbable_values(df, rrp_bounds: tuple, demand_bounds: tuple):
    initial_rows = len(df)

    # create mask to find valid rows based on bounds
    rrp_mask = (df["RRP"] >= rrp_bounds[0]) & (df["RRP"] <= rrp_bounds[1])
    rrp_removed = (~rrp_mask).sum()
    if rrp_removed > 0:
        print(
            f"RRP: {rrp_removed} rows outside [{rrp_bounds[0]}, {rrp_bounds[1]}] range."
        )

    # create mask to find valid rows based on bounds
    demand_mask = (df["TOTALDEMAND"] >= demand_bounds[0]) & (
        df["TOTALDEMAND"] <= demand_bounds[1]
    )
    demand_removed = (~demand_mask).sum()
    if demand_removed > 0:
        print(
            f"TOTALDEMAND: {demand_removed} rows outside [{demand_bounds[0]}, {demand_bounds[1]}] range."
        )

    # combine masks to filter dataframe with valid rrp and demand vals 
    df = df[rrp_mask & demand_mask]
    total_removed = initial_rows - len(df)

    if total_removed > 0:
        print(
            f"Total removed: {total_removed} rows ({total_removed/initial_rows*100:.2f}% of data)."
        )
    else:
        print(f"No rows removed; all values within bounds.")

    return df


def feature_engineering(df, long_term_sin_cos_encoding: bool = True, rolling_stats: bool = True):
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # not as important for short-term forecasts, but may help with long-term seasonality
    if long_term_sin_cos_encoding:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    
    df["RRP_lag1"] = df["RRP"].shift(1)
    df["RRP_lag12"] = df["RRP"].shift(12)
    df["RRP_lag288"] = df["RRP"].shift(288)

    df["TOTALDEMAND_lag1"] = df["TOTALDEMAND"].shift(1)
    df["TOTALDEMAND_lag12"] = df["TOTALDEMAND"].shift(12)
    df["TOTALDEMAND_lag288"] = df["TOTALDEMAND"].shift(288)

    # could cause overfitting so make optional 
    if rolling_stats:
        df["RRP_roll_mean_12"] = df["RRP"].rolling(window=12, min_periods=1).mean()
        df["RRP_roll_std_12"] = df["RRP"].rolling(window=12, min_periods=1).std()
        df["RRP_roll_min_12"] = df["RRP"].rolling(window=12, min_periods=1).min()
        df["RRP_roll_max_12"] = df["RRP"].rolling(window=12, min_periods=1).max()

        df["TOTALDEMAND_roll_mean_12"] = (
            df["TOTALDEMAND"].rolling(window=12, min_periods=1).mean()
        )
        df["TOTALDEMAND_roll_std_12"] = (
            df["TOTALDEMAND"].rolling(window=12, min_periods=1).std()
        )

    return df


def clean_data(df, apply_feature_engineering=True, long_term_encoding=True, rolling_stats=True):
    df = df.copy()

    # make the datetime column the index if not already
    if "SETTLEMENTDATE" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("SETTLEMENTDATE")

    print(f"Starting shape: {df.shape}")

    print("\nStep 1: Checking time index...")

    # ensure no duplicates and expected frequency
    df = check_time_index(df, expected_freq_minutes=5)

    print("\nStep 2: Checking gaps and filling missing values...")
    df = check_gaps(
        df, rrp_ffill_limit=2, demand_interp_limit=4, expected_freq_minutes=5
    )

    print("\nStep 3: Removing improbable values...")
    df = remove_improbable_values(
        df, rrp_bounds=(-1000, 15500), demand_bounds=(0, 20000)
    )

    if apply_feature_engineering:
        print("\nStep 4: Engineering features...")
        df = feature_engineering(df, long_term_sin_cos_encoding=long_term_encoding, 
                                 rolling_stats=rolling_stats)
        
        # Drop rows with NaN values in lag/rolling features only (not entire dataframe)
        rows_before = len(df)
        lag_cols = ["RRP_lag1", "RRP_lag12", "RRP_lag288", 
                    "TOTALDEMAND_lag1", "TOTALDEMAND_lag12", "TOTALDEMAND_lag288"]
        if rolling_stats:
            lag_cols.extend(["RRP_roll_std_12", "TOTALDEMAND_roll_std_12"])
        
        df = df.dropna(subset=lag_cols)
        rows_dropped = rows_before - len(df)
        if rows_dropped > 0:
            print(f"  Dropped {rows_dropped} rows with NaN values from feature engineering")

    print(f"\nCleaning complete. Final shape: {df.shape}")
    return df


def prepare_train_test_features(train_cleaned, test_cleaned, long_term_encoding = True,
                                 rolling_stats = True, lag: int = 288):
    lag_cols = ["RRP_lag1", "RRP_lag12", "RRP_lag288", 
                "TOTALDEMAND_lag1", "TOTALDEMAND_lag12", "TOTALDEMAND_lag288"]
    if rolling_stats:
        lag_cols.extend(["RRP_roll_std_12", "TOTALDEMAND_roll_std_12"])
    
    print("\n" + "=" * 70)
    print("Engineering features for TRAIN...")
    print("=" * 70)
    train_final = feature_engineering(train_cleaned, long_term_sin_cos_encoding=long_term_encoding,
                                       rolling_stats=rolling_stats)
    
    rows_before_train = len(train_final)
    train_final = train_final.dropna(subset=lag_cols)
    print(f"  Dropped {rows_before_train - len(train_final)} rows with NaN values from train")

    print("\n" + "=" * 70)
    print("Engineering features for TEST (using train history)...")
    print("=" * 70)
    # Take last 288 rows of cleaned train as history for test lags
    train_history = train_cleaned.tail(lag)
    
    # Concatenate history + test
    combined = pd.concat([train_history, test_cleaned])
    
    # Compute features on combined data, this way test lags all have values
    combined_with_features = feature_engineering(combined, long_term_sin_cos_encoding=long_term_encoding,
                                                  rolling_stats=rolling_stats)
    
    # Slice back to just test rows
    test_final = combined_with_features.loc[test_cleaned.index]
    
    # Drop any remaining NaNs
    rows_before_test = len(test_final)
    test_final = test_final.dropna(subset=lag_cols)
    print(f"Dropped {rows_before_test - len(test_final)} rows with NaN values from test")
    
    return train_final, test_final

def final_data_info(train_final: pd.DataFrame, test_final: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Final data info:")
    print("=" * 70)
    print("TRAIN:")
    print(train_final.info())
    print("\nTEST:")
    print(test_final.info())

def load_data(file_path: str) -> pd.DataFrame:
    file_path = "PRICE_AND_DEMAND_FULL_VIC1.csv"
    df = pd.read_csv(
        file_path, parse_dates=["SETTLEMENTDATE"], index_col="SETTLEMENTDATE"
    )

    print("=" * 70)
    print("Initial data info:")
    print("=" * 70)
    print(df.info())
    return df
def split_and_clean_data(df):
    
    print("\n" + "=" * 70)
    print("Splitting data (80% train, 20% test)...")
    print("=" * 70)
    train, test = time_based_train_test_split(df, test_size=0.2)
    print(f"Train: {train.index[0]} to {train.index[-1]} ({len(train)} rows)")
    print(f"Test:  {test.index[0]} to {test.index[-1]} ({len(test)} rows)")

    print("\n" + "=" * 70)
    print("Cleaning TRAIN set (without features)...")
    print("=" * 70)
    train_cleaned = clean_data(train, apply_feature_engineering=False)

    print("\n" + "=" * 70)
    print("Cleaning TEST set (without features)...")
    print("=" * 70)
    test_cleaned = clean_data(test, apply_feature_engineering=False)

    # Apply feature engineering with train history for test
    train_final, test_final = prepare_train_test_features(train_cleaned, test_cleaned, 
                                                           long_term_encoding=True, rolling_stats=True)
    return train_final, test_final

def save_data(train_final: pd.DataFrame, test_final: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Saving cleaned datasets...")
    print("=" * 70)
    train_final.to_csv("CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv")
    test_final.to_csv("CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv")
    print("Saved: CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv")
    print("Saved: CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv")

def main():
    df = load_data("PRICE_AND_DEMAND_FULL_VIC1.csv")
    train_final, test_final = split_and_clean_data(df)
    save_data(train_final, test_final)
    final_data_info(train_final, test_final)



if __name__ == "__main__":
    main()

# main is getting a bit too long 