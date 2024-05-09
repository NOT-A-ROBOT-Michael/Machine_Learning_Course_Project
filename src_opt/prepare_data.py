import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from skimpy import clean_columns
from sklearn.preprocessing import OneHotEncoder






def prepare_data(filepath):

    df = pd.read_csv(filepath)

    df = clean_columns(df)

    numerical_columns = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term']
    categorical_columns = ['gender', 'married', 'dependents', 'education', 'self_employed', 'credit_history', 'property_area']

    for col in categorical_columns:
         df[col].fillna(df[col].mode()[0], inplace=True)

    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    ohe = OneHotEncoder(sparse=False)
    columns_encode = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
    encoded_columns = ohe.fit_transform(df[columns_encode])
    new_columns = ohe.get_feature_names_out(columns_encode)
    encoded_df = pd.DataFrame(encoded_columns, columns=new_columns)
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df.drop(columns=columns_encode), encoded_df], axis=1)

    return df


# this will be used
def prepare_data_final(df):
    df = clean_columns(df)

    numerical_columns = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term']
    categorical_columns = ['gender', 'married', 'dependents', 'education', 'self_employed', 'credit_history', 'property_area']

    for col in categorical_columns:
         df[col].fillna(df[col].mode()[0], inplace=True)

    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    ohe = OneHotEncoder(sparse=False)
    columns_encode = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
    encoded_columns = ohe.fit_transform(df[columns_encode])
    new_columns = ohe.get_feature_names_out(columns_encode)
    encoded_df = pd.DataFrame(encoded_columns, columns=new_columns)
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df.drop(columns=columns_encode), encoded_df], axis=1)

    return df