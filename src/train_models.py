import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv('../data/train_data.csv') 
df2 = pd.DataFrame()

y= df['loan_status']
X = df.drop('loan_status', axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X,y , 
                                   random_state=104,  
                                   test_size=0.25,  
                                   shuffle=True)


lr = LogisticRegression()
lr.fit(X_train, y_train)
ypred_lr =lr.predict(X_test)
print('Logistic Regression Accuracy Score:')
print(accuracy_score(ypred_lr,y_test))

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
ypred_forest = forest.predict(X_test)
print('Random Forest Accuracy Score:')
print(accuracy_score(ypred_forest,y_test))

def modeling(df2):
    y = df2['loan_status']
    X = df2.drop('loan_status', axis = 1)


    X_train, X_test, y_train, y_test = train_test_split(X,y , 
                                    random_state=104,  
                                    test_size=0.25,  
                                    shuffle=True)
    
    return X_train, X_test, y_train, y_test
