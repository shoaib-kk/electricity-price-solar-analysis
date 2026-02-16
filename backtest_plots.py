from matplotlib import pyplot as plt
import pandas as pd 

def on_key(event):
    # Allow users to close plots by pressing q instead of clicking the window
    if event.key == "q":
        plt.close(event.canvas.figure)


def plot_decomposition_component(ax, index, values, title, ylabel, color):
    ax.plot(index, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

def plot_cumulative_profit(actions_df: pd.DataFrame, fee_rate: float):
    # Calculate cumulative profit
    actions_df["profit"] = actions_df["action_kwh"] * actions_df["price_kwh"] * (1 - fee_rate)
    actions_df["cumulative_profit"] = actions_df["profit"].cumsum()

    # Plot cumulative profit over time
    figure, ax = plt.subplots(figsize=(12, 6))
    plot_decomposition_component(
        ax,
        actions_df["timestamp"],
        actions_df["cumulative_profit"],
        "Cumulative Profit Over Time",
        "Cumulative Profit ($)",
        "blue"
    )
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


def plot_cumulative_profit_overlay(action_dfs: list[pd.DataFrame], labels: list[str], fee_rate: float):
    if len(action_dfs) != len(labels):
        raise ValueError("action_dfs and labels must have the same length.")

    figure, ax = plt.subplots(figsize=(12, 6))
    for df, label in zip(action_dfs, labels):
        work_df = df.copy()
        work_df["profit"] = work_df["action_kwh"] * work_df["price_kwh"] * (1 - fee_rate)
        work_df["cumulative_profit"] = work_df["profit"].cumsum()
        ax.plot(work_df["timestamp"], work_df["cumulative_profit"], label=label)

    ax.set_title("Cumulative Profit Over Time")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.grid(True)
    ax.legend()
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    
def plot_battery_soc(actions_df: pd.DataFrame):
    # Plot battery state of charge (SoC) over time
    figure, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actions_df["timestamp"], actions_df["soc_kwh"], label="Battery SoC (kWh)", color="orange")
    ax.set_title("Battery State of Charge Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("State of Charge (kWh)")
    ax.legend()
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_drawdown(actions_df: pd.DataFrame):
    # Calculate drawdown
    actions_df["cumulative_profit"] = actions_df["profit"].cumsum()
    actions_df["running_max"] = actions_df["cumulative_profit"].cummax()
    actions_df["drawdown"] = actions_df["running_max"] - actions_df["cumulative_profit"]

    # Plot drawdown over time
    figure, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actions_df["timestamp"], actions_df["drawdown"], label="Drawdown", color="red")
    ax.set_title("Drawdown Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown ($)")
    ax.legend()
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.grid(True)
    plt.tight_layout()
    plt.show(
    )


def plot_drawdown_overlay(action_dfs: list[pd.DataFrame], labels: list[str]):
    if len(action_dfs) != len(labels):
        raise ValueError("action_dfs and labels must have the same length.")

    figure, ax = plt.subplots(figsize=(12, 6))
    for df, label in zip(action_dfs, labels):
        work_df = df.copy()
        work_df["cumulative_profit"] = work_df["profit"].cumsum()
        work_df["running_max"] = work_df["cumulative_profit"].cummax()
        work_df["drawdown"] = work_df["running_max"] - work_df["cumulative_profit"]
        ax.plot(work_df["timestamp"], work_df["drawdown"], label=label)

    ax.set_title("Drawdown Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True)
    ax.legend()
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

def plot_forecast_window(actions_df: pd.DataFrame, forecast_horizon: int):
    # Plot actual vs. forecasted prices for a specific window
    figure, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actions_df["timestamp"], actions_df["price_kwh"], label="Actual Price", color="blue")
    ax.plot(actions_df["timestamp"] + pd.Timedelta(minutes=forecast_horizon), actions_df["forecast_price_kwh"], label="Forecasted Price", color="green")
    ax.set_title(f"Actual vs. Forecasted Prices (Horizon: {forecast_horizon} minutes)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price ($/kWh)")
    ax.legend()
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.grid(True)
    plt.tight_layout()
    plt.show()