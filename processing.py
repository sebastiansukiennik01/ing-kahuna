import os
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("processing")


def load_data(filename: str):
    """
    Load data from data folder. 
    args: 
        filename : name of the file in "data/" folder
    """
    filepath = os.path.join("data", filename) 
    
    data = pd.read_csv(filepath)
    
    data = change_to_dates(data)
    
    data = calculate_limit_use(data)
    data = add_custom_variables(data)
    data = drop_unnecessary_columns(data)
    data = fill_missing_values(data)
    data = calculate_customer_age(data)
    data = calculate_relation_time(data)
    
    return data


def change_to_dates(data: pd.DataFrame, columns: list = ["Ref_month",  "Birth_date", "Oldest_account_date", "Contract_origination_date", "Contract_end_date"]):
    """
    changes to dates
    """
    log.info("Changing dates to datetime..")
    data['Ref_month'] = pd.to_datetime(data['Ref_month'], format='%m-%Y')
    data['Birth_date'] = pd.to_datetime(data['Birth_date'], format='%d-%m-%Y')
    data['Oldest_account_date'] = pd.to_datetime(data['Oldest_account_date'], format='%d-%m-%Y')
    data['Contract_origination_date'] = pd.to_datetime(data['Contract_origination_date'], format='%d-%m-%Y')
    data['Contract_end_date'] = pd.to_datetime(data['Contract_end_date'], format='%d-%m-%Y')
    
    return data
    

def add_custom_variables(data):
    """
    Adds custom variables calcualated based on the provided data.
    args:
        data : pandas dataframe
    """
    log.info("Adding custom variables to dataset..")
    
    # contract origin date + contract end data ===> months_left
    data["Months_left"] = (data["Contract_end_date"] - data["Contract_origination_date"]) / pd.Timedelta(days=30)
    
    
    log.warn(data.shape)
    for k in range(0, 6):
        data = add_change_in_balance(data, k)
    for k in range(0, 1):
        data = add_change_in_debt(data, k)
    for k in range(0, 4):
        data = add_change_in_savings(data, k)
    for k in range(0, 4):
        data = add_change_in_limit_use(data, k)
    
    data = add_change_in_overdue(data, k)
        
    # drop columns that are now not needed (from change in balance, change in debt, etc.)
    log.warn(data.shape)
    data = drop_unnecessary_columns(data, columns=[f"inc_transactions_amt_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"out_transactions_amt_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Os_term_loan_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Os_credit_card_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Os_mortgage_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Savings_amount_balance_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Overdue_term_loan_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Overdue_credit_card_H{i}" for i in range(0, 13)])
    data = drop_unnecessary_columns(data, columns=[f"Overdue_mortgage_H{i}" for i in range(0, 13)])
    log.warn(data.shape)
        
    return data


def drop_unnecessary_columns(data, columns: list = []):
    """
    Drops provided columns (lack of significance or data) AND deault columns.
    args:
        data : pandas dataframe
        columns : columns names to be dropped
    """ 
    
    # TODO DROP CREDIT CARDS
    log.info("Dropping unnecesssary columns..")
    
    external = ["External_credit_card_balance", "External_term_loan_balance", "External_mortgage_balance"]
    incomes = ['Income_H0', 'Income_H1', 'Income_H2', 'Income_H3', 'Income_H4', 'Income_H5', 'Income_H6', 'Income_H7', 'Income_H8', 'Income_H9', 'Income_H10', 'Income_H11', 'Income_H12']
    transaction = ["inc_transactions_Hx", "out_transactions_Hx"]
    
    to_drop = columns + external + incomes + transaction
    
    return data.drop(columns=to_drop, errors='ignore')
    
    
def fill_missing_values(data):
    """
    Fill invalid data with more descriptive/appropriate values.
    """
    log.info("Filling missing values..")
    data["Active_mortgages"] = data["Active_mortgages"].replace({-9999: 0})
    
    return data
    
    
    
def add_change_in_balance(data: pd.DataFrame, k: int):
    """
    Detect if outcomming transaction > incomming_transactions (1, 2, 3 months) -> 1
    args:
        data : pandas dataframe
        k : number of negative periods after which change_is_balance marked as 1
    """
    temp = pd.DataFrame()
    for i in range(0, 13):
        temp[f"balance_h{i}"] = data[f"inc_transactions_amt_H{i}"] - data[f"out_transactions_amt_H{i}"]
        
    data[f"change_in_balance_H{k}"] = 0

    negative = [f"balance_h{i}" for i in range(0, k+1)]
    positive = [f"balance_h{i}" for i in range(k+1, 13)]
    mask = (temp[positive] > 0).all(axis=1) & (temp[negative] < 0).all(axis=1)

    
    data.loc[mask, f"change_in_balance_H{k}"] = 1
    
    return data

    
def add_change_in_debt(data: pd.DataFrame, k: int):
    """
    Detect if sum of debt [Os_term_loan_Hx, Os_credit_card_Hx, Os_mortgage_Hx] stopps decresing (1, 2, 3 months) -> 1
    args:
        data : pandas dataframe
        k : number of negative periods after which change_is_balance marked as 1
    """
    temp = pd.DataFrame()
    for i in range(0, 13):
        temp[f"Os_debt_h{i}"] = data[f"Os_term_loan_H{i}"] + data[f"Os_credit_card_H{i}"] + data[f"Os_mortgage_H{i}"]
        
    for i in range(0, 12):
        temp[f"Os_debt_h{i}_change"] = temp[f"Os_debt_h{i}"] - temp[f"Os_debt_h{i+1}"]
        
    data[f"change_in_debt_H{k}"] = 0
    data[f"change_in_debt_last_{k}"] = 0

    positive = [f"Os_debt_h{i}_change" for i in range(0, k+1)]
    negative = [f"Os_debt_h{i}_change" for i in range(k+1, 12)]
    mask = (temp[positive] >= 0).all(axis=1) & (temp[negative] < 0).all(axis=1)

    
    data.loc[mask, f"change_in_debt_H{k}"] = 1 # customer hasnt't reduced his debt for last k-months (previous to that every month he reduced his debt)
    data.loc[(temp[positive] >= 0).all(axis=1), f"change_in_debt_last_{k}"] = 1 # customer hasnt't reduced his debt for last k-months
    return data



# jeżeli komuś przez 1/2/3/4 miesiące spadły savingsy
def add_change_in_savings(data: pd.DataFrame, k: int):
    """
    Detect if savings decreased for last k periods (no matter what happened before).
    args:
        data : pandas dataframe
        k : number of periods for which savings have decreased
    """
    temp = pd.DataFrame()
        
    for i in range(0, 12):
        temp[f"change_in_savingsH{i}"] = data[f"Savings_amount_balance_H{i}"] - data[f"Savings_amount_balance_H{i+1}"]
        
    data[f"change_in_savings_lastH{k}"] = 0
    positive = [f"change_in_savingsH{i}" for i in range(0, k+1)]
    
    mask = (temp[positive] < 0).all(axis=1)    
    data.loc[mask, f"change_in_savings_lastH{k}"] = 1 # customer has reduced his savings for all of last 'k' periods

    return data


def add_change_in_overdue(data: pd.DataFrame, k: int):
    """
    Detect if sum of overdue [Overdue_term_loan_Hx, Overdue_credit_card_Hx, Overdue_mortgage_Hx] stopps increasead in last (1, 2, 3 months) -> 1
    args:
        data : pandas dataframe
        k : number of last periods in which change_is_balance marked as 1
    """
    temp = pd.DataFrame()
    for i in range(0, 13):
        temp[f"Overdue_H{i}"] = data[f"Overdue_term_loan_H{i}"] + data[f"Overdue_credit_card_H{i}"] + data[f"Overdue_mortgage_H{i}"]
        
    for i in range(0, 12):
        temp[f"Overdue_H{i}_change"] = temp[f"Overdue_H{i}"] - temp[f"Overdue_H{i+1}"]
        
    data[f"change_in_overdue_H{k}"] = 0

    positive = [f"Overdue_H{i}_change" for i in range(0, k+1)]
    mask = (temp[positive] > 0).all(axis=1)

    
    data.loc[mask, f"change_in_overdue_H{k}"] = 1 # customer has increased his overdue for last k-months
    return data

 
def add_change_in_limit_use(data: pd.DataFrame, k: int):
    """
    Detect if customer used more than 80% of the credit card limit for last k periods (no matter what happened before).
    args:
        data : pandas dataframe
        k : number of periods
    """
    exceeded = [f"Limit_use_H{i}" for i in range(1, k+1)]
    data[f"Limit_use_exceeded_H{k}"] = 0
    mask = (data[exceeded] > 80).all(axis=1)    
    data.loc[mask, f"Limit_use_exceeded_H{k}"] = 1 # customer has reduced his savings for all of last 'k' periods

    return data

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

def normalize_data(data):
    scaler = MinMaxScaler()
    for column in data.columns:
        data[column] = scaler.fit_transform(data[column])
    return data