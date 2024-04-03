from processing import load_data


if __name__ == "__main__":
    train = load_data("in_time.csv")
    test = load_data("out_of_time.csv")
    
    print(f"train shape: {train.shape}\ntest shape: {test.shape}")