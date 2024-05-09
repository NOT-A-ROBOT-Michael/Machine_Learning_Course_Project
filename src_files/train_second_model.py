import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_second_model(filepath):
    
    
    read_filepath = filepath+'\\data\\feature_e_data.csv'
    
    # reading data from csv file
    df = pd.read_csv(read_filepath)

    # Unamed: 0 column
    df.drop(df.columns[0], axis=1, inplace=True)

    # splitting dataframe columns into x and y
    y= df['loan_status']
    X = df.drop('loan_status', axis = 1)

    # splitting the data between test and train, although x_test is not needed
    X_train, X_test, y_train, y_test = train_test_split(X,y , 
                                   random_state=104,  
                                   test_size=0.25,  
                                   shuffle=True)

    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)

    dump_filepath = filepath+'\\artifacts\\model_2.pkl'
    
    #dump the forest model into model_2.pkl
    joblib.dump(forest,dump_filepath)
