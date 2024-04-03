def count_special_values(df):
    for column in df.columns:
        count_9999 = (df[column] == -9999).sum()
        if count_9999 > 0:
            print(f"Column '{column}' has {count_9999} rows with value 9999.")