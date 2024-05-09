import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.exceptions import ConvergenceWarning


def feature_engineering(filepath):
    
    read_filepath = filepath+'\\data\\train_data.csv'
    
    # reading data from csv file
    df = pd.read_csv(read_filepath)

    # Unamed: 0 column
    df.drop(df.columns[0], axis=1, inplace=True)

    # splitting dataframe columns into x and y
    y= df['loan_status']
    X = df.drop('loan_status', axis = 1)

    # Create the Random Over Sampler
    oversampler = RandomOverSampler(sampling_strategy='minority')

    # Resample the data
    X_oversampled, y_oversampled = oversampler.fit_resample(X, y)

    # Contactinating columns
    df = pd.concat([X_oversampled, y_oversampled], axis=1)
    
    write_filepath = filepath + "\\data\\feature_e_data.csv"

    # writing to csv file
    df.to_csv(write_filepath, index=True)

    return df

# feature_engineering("C:\\Users\\iyesme\\repos\\Machine_Learning_Course_Project_GroupM")

