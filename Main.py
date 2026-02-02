import Data_analysis
import Data_visualisation
import Data_cleaning
import Data_collection
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


if __name__ == "__main__":
    main()
