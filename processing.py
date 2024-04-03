import os
import pandas as pd


def load_data(filename: str):
    """
    Load data from data folder. 
    args: 
        filename : name of the file in "data/" folder
    """
    filepath = os.path.join("data", filename) 
    return pd.read_csv(filepath)