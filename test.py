from processing import load_data, change_to_dates, add_custom_variables

from plots import plot_stats, draw_boxplots
from hellwig import perform_hellwig
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train = load_data("in_time.csv")
    
    print(list(train.columns))
    
    print(train.shape)
    # print(train['Limit_use_exceeded_H1'])
    # print(train.select_dtypes(include=['object']).head())

    # train = change_to_dates(train)
    # train = add_custom_variables(train)
    numeric = ['No_dependants', 'Time_in_address', 'Credit_cards', 'Active_credit_card_lines', 'Debit_cards', 'Active_accounts', 'Active_loans', 'Active_mortgages', 'Time_in_current_job', 'Num_borrowers', 'Current_installment']
    numeric2 = ['inc_transactions_H', 'out_transactions_H', 'Income_H', 'inc_transactions_amt_H', 'out_transactions_amt_H', 'Current_amount_balance_H', 'Savings_amount_balance_H', 'Os_term_loan_H', 'Os_credit_card_H', 'Os_mortgage_H', 'Payments_term_loan_H', 'Payments_credit_card_H', 'Payments_mortgage_H', 'limit_in revolving_loans_H', 'utilized_limit_in revolving_loans_H', 'DPD_term_loan_H', 'DPD_credit_card_H', 'DPD_mortgage_H', 'Overdue_term_loan_H', 'Overdue_credit_card_H', 'Overdue_mortgage_H']
    
    # train = calculate_customer_age(train)
    # train = calculate_relation_time(train)
    # train = calculate_limit_use(train)

    customer_data = ['No_dependants', 'Age', 'Time_in_address', 'Time_in_current_job', 'Num_borrowers', 'Relation_time']

    products_data = ['Credit_cards', 'Active_credit_card_lines', 'Debit_cards', 'Active_accounts', 'Active_loans', 'Active_mortgages', 'Months_left', 'Current_installment']
    # Current_installment I mONTHS_LEFT - DAMY JESZCZE DO TRANSACTIONAL
    behavioral_transactional_data = ['Months_left', 'Current_installment', 'change_in_balance_H0', 'change_in_balance_H1', 'change_in_balance_H2', 'change_in_balance_H3', 'change_in_balance_H4', 'change_in_balance_H5', 'change_in_debt_H0', 'change_in_debt_last_0', 'change_in_savings_lastH0', 'change_in_savings_lastH1', 'change_in_savings_lastH2', 'change_in_savings_lastH3', 'Limit_use_exceeded_H1', 'Limit_use_exceeded_H2', 'Limit_use_exceeded_H3', 'change_in_overdue_H0', 'change_in_overdue_H1', 'change_in_overdue_H2', 'change_in_overdue_H3']
    # perform_hellwig(train, 'Target', customer_data, 1, 4)
    # perform_hellwig(train, 'Target', products_data, 1, 4)
    print(train.std()/train.mean())
    perform_hellwig(train, 'Target', behavioral_transactional_data, 1, 7)

    perform_hellwig(train, 'Target', train.columns, 1, 7)
    
    # x_train = train[numeric]
    # y_train = train['Target']
    # x_train = x_train.select_dtypes(include=['number']).copy()
    # rfe(x_train, y_train)



    # count_special_values(train)
    # print(train.shape)
    # print(list(train.columns))
