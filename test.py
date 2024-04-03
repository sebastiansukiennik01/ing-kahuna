from processing import load_data
import matplotlib.pyplot as plt
import numpy as np
from funkcje import count_special_values

if __name__ == "__main__":
    train = load_data("in_time.csv")
    test = load_data("out_of_time.csv")
    
    print(f"train shape: {train.shape}\ntest shape: {test.shape}")
    print(train['External_credit_card_balance'].describe())

    count_special_values()