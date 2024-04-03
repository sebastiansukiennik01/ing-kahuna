from datetime import datetime

def count_special_values(df):
    for column in df.columns:
        count_9999 = (df[column] == -9999).sum()
        if count_9999 > 0:
            print(f"Column '{column}' has {count_9999} rows with value 9999.")

def calculate_customer_age(df):
    df['Age'] = df['Ref_month'] - df['Birth_date']
    df['Age'] = df['Age'].dt.days // 365
    return df

def calculate_relation_time(df):
    df['Relation_time'] = df['Ref_month'] - df['Oldest_account_date']
    df['Relation_time'] = df['Relation_time'].dt.days // 365
    return df

def calculate_limit_use(df):
    for i in range(1,13):
        df[f'Limit_use_H{i}'] = df[f'utilized_limit_in_revolving_loans_H{i}'] / df[f'limit_in_revolving_loans_H{i}'] * 100
    return df