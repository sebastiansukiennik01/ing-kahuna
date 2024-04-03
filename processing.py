import os
import pandas as pd
import logging

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
    data = add_custom_variables(data)
    data = drop_unnecessary_columns(data)
    data = fill_missing_values(data)
    
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
    data["months_left"] = (data["Contract_end_date"] - data["Contract_origination_date"]) / pd.Timedelta(days=30)
    
    return data


def drop_unnecessary_columns(data, columns: list = []):
    """
    Drops provided columns (lack of significance or data) AND deault columns.
    args:
        data : pandas dataframe
        columns : columns names to be dropped
    """ 
    log.info("Dropping unnecesssary columns..")
    deafault = ["External_credit_card_balance", "External_term_loan_balance", "External_mortgage_balance"]
    
    return data.drop(columns=deafault + columns)
    
    
def fill_missing_values(data):
    """
    Fill invalid data with more descriptive/appropriate values.
    """
    log.info("Filling missing values..")
    data["Active_mortgages"] = data["Active_mortgages"].replace({-9999: 0})
    
    return data
    