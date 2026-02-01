from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def on_key(event):
    # Allow users to close plots by pressing q instead of clicking the window
    if event.key == "q":
        plt.close(event.canvas.figure)


def plot_decomposition_component(ax, index, values, title, ylabel, color):
    ax.plot(index, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)


def read_data(file_path):
    # Load CSV with validation for required columns and datetime index.
    csv_path = Path(file_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["SETTLEMENTDATE"])


    required_cols = {"RRP", "TOTALDEMAND"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")

    df = df.sort_values("SETTLEMENTDATE")
    df = df.set_index("SETTLEMENTDATE")
    return df

def resample_hourly_mean(df):
    hourly = df[["RRP", "TOTALDEMAND"]].resample("h").mean()


    return hourly


def plot_hourly_average(df_hourly):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(df_hourly.index, df_hourly["RRP"], color="blue", label="Hourly RRP")
    axes[0].set_title("Hourly Average Price over Time")
    axes[0].set_ylabel("Price (RRP)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df_hourly.index, df_hourly["TOTALDEMAND"], color="green", label="Hourly demand")
    axes[1].set_title("Hourly Average Demand over Time")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Total Demand")
    axes[1].legend()
    axes[1].grid(True)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


def _prepare_series_for_decomposition(df_hourly, column: str, max_gap_hours: int = 6) -> pd.Series:
    """Return an hourly, gap-filled series suitable for seasonal_decompose.

    Interpolate short gaps (up to max_gap_hours) using time interpolation,
    then forward/backward fill anything left so the series is regular. These
    filled values are for decomposition stability in diagnostics, not for
    inference or forecasting.
    """

    series = df_hourly[column].asfreq("h")
    n_missing = series.isna().sum()

    if n_missing == 0:
        return series

    print(
        f"Decomposition: {n_missing} missing hourly values for {column} "
    )   

    series = series.interpolate(method="time", limit=max_gap_hours)
    remaining = series.isna().sum()

    if remaining > 0:
        print(
            f"Decomposition: {remaining} values still missing for {column}"
        )
        series = series.ffill().bfill()

    return series


def plot_decomposition_hourly(df_hourly):
    """ Weekly seasonality really matters in electricity because human routines
     (workdays vs weekends) drive demand, which then drives prices

     Use an additive weekly model (24*7 hours) 
     multiplicative could break because NEM prices can be zero/negative 
    """
    period = 24 * 7

    price_series = _prepare_series_for_decomposition(df_hourly, "RRP")
    if len(price_series) < 2 * period:
        print(f"Skipping price decomposition: need at least {2 * period} hours, have {len(price_series)}.")

    else:
        decomposition_price = seasonal_decompose(price_series, model="additive", period=period)

        figure, axes = plt.subplots(4, 1, figsize=(12, 12))
        # Plot the same (gap-filled) series that was decomposed so panels are consistent.
        plot_decomposition_component(axes[0], price_series.index, price_series,
                                    "Price - Hourly (filled for decomposition)", "Price (RRP)", "blue")
        plot_decomposition_component(axes[1], decomposition_price.trend.index, decomposition_price.trend,
                                    "Price - Trend", "Trend", "orange")
        plot_decomposition_component(axes[2], decomposition_price.seasonal.index, decomposition_price.seasonal,
                                    "Price - Seasonal (Weekly Pattern)", "Seasonal", "green")
        plot_decomposition_component(axes[3], decomposition_price.resid.index, decomposition_price.resid,
                                    "Price - Residual", "Residual", "red")
        axes[3].set_xlabel("Time")
        figure.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show()

    # Decompose Demand with the same weekly additive assumption for consistency.
    demand_series = _prepare_series_for_decomposition(df_hourly, "TOTALDEMAND")
    if len(demand_series) < 2 * period:
        print(f"Skipping demand decomposition: need at least {2 * period} hours, have {len(demand_series)}.")
        return

    decomposition_demand = seasonal_decompose(demand_series, model="additive", period=period)

    figure, axes = plt.subplots(4, 1, figsize=(12, 12))
    plot_decomposition_component(axes[0], demand_series.index, demand_series,
                                "Demand - Hourly (filled for decomposition)", "Total Demand", "green")
    plot_decomposition_component(axes[1], decomposition_demand.trend.index, decomposition_demand.trend,
                                "Demand - Trend", "Trend", "orange")
    plot_decomposition_component(axes[2], decomposition_demand.seasonal.index, decomposition_demand.seasonal,
                                "Demand - Seasonal (Weekly Pattern)", "Seasonal", "purple")
    plot_decomposition_component(axes[3], decomposition_demand.resid.index, decomposition_demand.resid,
                                "Demand - Residual", "Residual", "red")
    axes[3].set_xlabel("Time")
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()



def create_hour_day_heatmap(df_hourly, column, title, cmap, cbar_label, heatmap_kwargs=None):
    # Heatmap is used to surface hour-of-day vs day-of-week structure (intraday + weekly seasonality).
    df_temp = df_hourly.copy()
    df_temp["hour"] = df_temp.index.hour
    df_temp["day_of_week"] = df_temp.index.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_data = df_temp.pivot_table(values=column, index="day_of_week", columns="hour", aggfunc="mean")
    pivot_data = pivot_data.reindex(day_order)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        pivot_data,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        ax=ax,
        **(heatmap_kwargs or {}),  # allow caller to control scaling/centering for spikes/negatives
    )
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


def hourly_demand_heatmap(df_hourly):
    # scaling stops a few extreme peaks from washing out typical demand patterns.
    create_hour_day_heatmap(
        df_hourly,
        "TOTALDEMAND",
        "Average Demand by Hour and Day of Week",
        "YlOrRd",
        "Average Demand",
        heatmap_kwargs={"robust": True},
    )


def hourly_price_heatmap(df_hourly):
    # Centre the colour map on $0 to make negative prices visually obvious robust=True down-weights spikes.
    create_hour_day_heatmap(
        df_hourly,
        "RRP",
        "Average Price by Hour and Day of Week",
        "coolwarm",
        "Average Price (RRP)",
        heatmap_kwargs={"center": 0.0, "robust": True},
    )


def describe_price_demand_extremes(df_hourly):
    # Print a quick, human-readable summary of negative prices and extreme spikes.
    price = df_hourly["RRP"].dropna()
    demand = df_hourly["TOTALDEMAND"].dropna()

    if price.empty or demand.empty:
        raise ValueError("Price or demand series is empty after dropping NaNs cannot summarise extremes.")

    n = len(price)
    neg_count = (price < 0).sum()
    p_neg = 100 * neg_count / n

    p1, p99 = price.quantile([0.01, 0.99])
    d1, d99 = demand.quantile([0.01, 0.99])

    print("\n--- NEM price/demand behaviour summary (hourly) ---")
    print(f"Price min/max: {price.min():.2f} / {price.max():.2f} AUD/MWh")
    print(f"Share of hours with negative prices: {p_neg:.2f}% ({neg_count} of {n})")
    print(f"Price 1st/99th percentiles: {p1:.2f} / {p99:.2f} AUD/MWh (defines typical range vs spikes)")
    print(f"Demand 1st/99th percentiles: {d1:.0f} / {d99:.0f} MW")




def main():
    file_path = "PRICE_AND_DEMAND_FULL_VIC1.csv"
    df = read_data(file_path)
    df_hourly = resample_hourly_mean(df)

    describe_price_demand_extremes(df_hourly)

    print("1. Hourly Line Plot")
    plot_hourly_average(df_hourly)

    print("2. Hourly Demand Heatmap (Hour x Day of Week)")
    hourly_demand_heatmap(df_hourly)

    print("3. Hourly Price Heatmap (Hour x Day of Week)")
    hourly_price_heatmap(df_hourly)

    print("4. Seasonal Decomposition")
    plot_decomposition_hourly(df_hourly)


if __name__ == "__main__":
    main()
