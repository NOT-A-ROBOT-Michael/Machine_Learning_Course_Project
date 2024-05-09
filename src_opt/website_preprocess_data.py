import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from skimpy import clean_columns
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(filepath):

    df = pd.read_csv(filepath)

    skewed_features = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term', 'total_income']
    df[skewed_features] = np.log1p(df[skewed_features])

    return df
