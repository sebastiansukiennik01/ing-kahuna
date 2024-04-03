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


def change_to_dates(data: pd.DataFrame, columns: list = ["Ref_month",  "Birth_date", "Oldest_account_date", "Contract_origination_date", "Contract_end_date"]):
    """
    changes to dates
    """
    # data[columns] = pd.to_datetime(data[columns])
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
    # contract origin date + contract end data ===> months_left
    data["months_left"] = (data["Contract_end_date"] - data["Contract_origination_date"]) / pd.Timedelta(days=30)
    
    return data
    
    
