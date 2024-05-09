import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from skimpy import clean_columns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# C:\\Users\\iyesme\\repos\\Machine_Learning_Course_Project_GroupM

def dataprep(filepath):
    raw_filepath = filepath + '\\data\\raw_data.csv'
    
    #read raw data from cdv file
    df = pd.read_csv(raw_filepath)
    
    #clean columns
    df = clean_columns(df)
    
    # drop loan id column
    df = df.drop(columns=['loan_id'], inplace=False)
    
    # dropping outliers and filling in missing values
    df = df[
        (df['applicant_income'] <= 20000) &
        (df['coapplicant_income'] <= 10000) &
        (df['loan_amount'] <= 400)
    ]

    categorical_columns = ['gender', 'married', 'dependents', 'self_employed']
    numerical_columns = ['loan_amount', 'loan_amount_term', 'credit_history']

    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    
    # encoding categorical variables
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    columns_encode = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
    encoded_columns = ohe.fit_transform(df[columns_encode])
    new_columns = ohe.get_feature_names_out(columns_encode)
    encoded_df = pd.DataFrame(encoded_columns, columns=new_columns)
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df.drop(columns=columns_encode), encoded_df], axis=1)
    
    # adding a new column for total income by combining applicant income and co-applicant income
    # df['total_income'] = df['applicant_income'] + df['coapplicant_income']
    # df= df.copy()
    
    # skewed features
    skewed_features = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term'] # , 'total_income'
    df[skewed_features] = np.log1p(df[skewed_features])

    clean_encoded_featured_transformed_df = df.copy()
    
    store_filepath = filepath + '\\data\\train_data.csv'
    
    clean_encoded_featured_transformed_df.to_csv(store_filepath, index=True)
    
    
dataprep("C:\\Users\\iyesme\\repos\\Machine_Learning_Course_Project_GroupM")


