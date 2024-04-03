from processing import load_data, change_to_dates, add_custom_variables
from funkcje import count_special_values

if __name__ == "__main__":
    train = load_data("in_time.csv")
    print(train.select_dtypes(include=['object']).head())

    train = change_to_dates(train)
    train = add_custom_variables(train)
    # print(train.info())
    print(train["months_left"])

    count_special_values(train)

