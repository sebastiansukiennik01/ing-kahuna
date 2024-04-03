from processing import load_data, change_to_dates, add_custom_variables
from funkcje import count_special_values

if __name__ == "__main__":
    train = load_data("in_time.csv")
    print(train.info())

    count_special_values(train)
    print(train.shape)
    print(list(train.columns))
