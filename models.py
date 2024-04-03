import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from imblearn.under_sampling import NearMiss
import sklearn.model_selection as skm
from sklearn.metrics import make_scorer

def divide_x_y(df):
    y_train = df['target']
    x_train = df.drop(columns=['target'])
    return x_train, y_train

def oversample_data(df):
    smote = SMOTE(sampling_strategy="minority", random_state=42)
    features = df.drop(columns=['target'])
    labels = df['target']
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    # Combine resampled features and labels into a new DataFrame
    resampled_data = pd.concat([pd.DataFrame(features_resampled, columns=features.columns),
                                pd.DataFrame(labels_resampled, columns=['target'])], axis=1)

    return resampled_data

def oversample_data_ros(df):
    ros = RandomOverSampler(random_state=42)
    x, y = divide_x_y(df)
    x_train_ros, y_train_ros = ros.fit_resample(x, y)
    # Combine resampled features and labels into a new DataFrame
    df = pd.concat([x_train_ros, y_train_ros], axis=1)
    return df

def undersample_data(df):
    rus = RandomUnderSampler(random_state=42)
    x, y = divide_x_y(df)
    x_train_rus, y_train_rus = rus.fit_resample(x, y)
    df = pd.concat([x_train_rus, y_train_rus], axis=1)
    return df

def undersample_data_nearmiss2(df):
    nearMiss2 = NearMiss(version=3)
    x, y = divide_x_y(df)
    x_train_nearMiss2, y_train_nearMiss2 = nearMiss2.fit_resample(x, y)
    df = pd.concat([x_train_nearMiss2, y_train_nearMiss2], axis=1)
    return df


def logistic_regression(x_train, y_train):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    return lr

def logistic_regression_with_cross_validation(x_train, y_train, model):
    kfold = skm.KFold(5,random_state=0,shuffle=True)
    
    scorers = {
                'precision_score': make_scorer(precision_score),
                'recall_score': make_scorer(recall_score),
                'accuracy_score': make_scorer(accuracy_score)
                }
    # , 'max_iter': [300, 1000, 1600]
    param_grid = { 'C':[0.01, 0.05, 0.2, 0.5, 1.5]}
    grid_search_LR = skm.GridSearchCV(model, param_grid=param_grid, cv=kfold, scoring=scorers, 
            refit="recall_score")
    grid_search_LR.fit(x_train, y_train)
    sorted_results = sorted(zip(grid_search_LR.cv_results_['params'],
                                grid_search_LR.cv_results_['mean_test_recall_score'],
                                grid_search_LR.cv_results_['mean_test_precision_score'],
                                grid_search_LR.cv_results_['mean_test_accuracy_score']),
                            key=lambda x: (x[1], x[2], x[3]), reverse=True)

    print("Five best results for given hyperparameters of Logistic Regression model:")
    for i, (params, recall, precision, accuracy) in enumerate(sorted_results[:5], 1):
        print(f"{i}. Parameters: {params}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    return grid_search_LR.best_estimator_

def calculate_performance(x, y, model):
    proba = model.predict_proba(x)
    plt.hist(proba[:, 1], bins=100)
    plt.show()

    y_pred = model.predict(x)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

def calculate_summary_table(x, model):
    feature_names = x.columns.values
    summary_table = pd.DataFrame(columns=['Feature name'], data=feature_names)
    summary_table['Coefficients'] = np.transpose(model.coef_)
    summary_table['Exponential_Coefficients'] = np.transpose(np.exp(model.coef_))
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', model.intercept_[0], model.intercept_[0]]
    summary_table = summary_table.sort_index()
    print(summary_table)

def random_forest(x_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    return rf

def create_confusion_matrix(x, y, model):
    y_pred = model.predict(x)
    conf_matrix_rf = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def random_forest_with_cross_validation(x_train, y_train, model):
    kfold = skm.KFold(5,random_state=0,shuffle=True)

    scorers = {
                'precision_score': make_scorer(precision_score),
                'recall_score': make_scorer(recall_score),
                'accuracy_score': make_scorer(accuracy_score)
                }

    param_grid = { 'n_estimators':[5, 20],
            #   'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth': [1, 3],
              'max_features': ["sqrt"]}
    grid_search_LR = skm.GridSearchCV(model, param_grid=param_grid, cv=kfold, scoring=scorers, 
            refit="recall_score")
    grid_search_LR.fit(x_train, y_train)
    sorted_results = sorted(zip(grid_search_LR.cv_results_['params'],
                                grid_search_LR.cv_results_['mean_test_recall_score'],
                                grid_search_LR.cv_results_['mean_test_precision_score'],
                                grid_search_LR.cv_results_['mean_test_accuracy_score']),
                            key=lambda x: (x[1], x[2], x[3]), reverse=True)

    print("Five best results for given hyperparameters:")
    for i, (params, recall, precision, accuracy) in enumerate(sorted_results, 1):
        print(f"{i}. Parameters: {params}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    return grid_search_LR.best_estimator_
    
def show_feature_importance_rf(x_train, model):
    feature_names = x_train.columns.values
    feature_importances = model.feature_importances_

    summary_table = pd.DataFrame(columns=['Feature name'], data=feature_names)
    summary_table['Feature Importance'] = feature_importances
    summary_table = summary_table.sort_values(by='Feature Importance', ascending=False)

    sorted_indices = feature_importances.argsort()
    sorted_feature_names = feature_names[sorted_indices]
    sorted_feature_importances = feature_importances[sorted_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(sorted_feature_names[-20:], sorted_feature_importances[-20:]) 
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Random Forest Feature Importance')
    plt.show()

def generate_ROC(x, y, model):
    RocCurveDisplay.from_estimator(model, x, y)