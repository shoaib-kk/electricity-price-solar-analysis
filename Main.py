import Data_analysis
import Data_visualisation
import Data_cleaning
import Data_collection
import Point_forecast
import Quantile_regression
import time
import numpy as np
import pandas as pd


def main():
    # Collect data
    Data_collection.main()

    # Clean data
    Data_cleaning.main()
    # Visualize data

    Data_visualisation.main()

    # Analyze data
    Data_analysis.main()

    Point_forecast.main()

    Quantile_regression.main()
if __name__ == "__main__":
    main()
