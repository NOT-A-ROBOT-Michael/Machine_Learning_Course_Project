import pandas as pd
import numpy as np
#!pip install imblearn
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import accuracy_score

df = pd.read_csv('../data/train_data.csv')

y= df['loan_status']
X = df.drop('loan_status', axis = 1)

# Create the Random Over Sampler
oversampler = RandomOverSampler(sampling_strategy='minority')

# Resample the data
X_oversampled, y_oversampled = oversampler.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_oversampled,y_oversampled , 
                                   random_state=104,  
                                   test_size=0.25,  
                                   shuffle=True)

mapping = {"N": 0, "Y": 1}
y_train = np.vectorize(mapping.get)(y_train)
y_test = np.vectorize(mapping.get)(y_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)


ypred_lr =lr.predict(X_test)
print('Logistic Regression Accuracy Score:')
print(f'Accuracy: ',accuracy_score(ypred_lr,y_test))
print(f'Precision: ',precision_score(ypred_lr,y_test))
print(f'Recall: ',recall_score(ypred_lr,y_test))
print(f'F1-score: ',f1_score(ypred_lr,y_test))

print()

forest = RandomForestClassifier()
forest.fit(X_train, y_train)

ypred_forest = forest.predict(X_test)
print('Random Forest Accuracy Score:')
print(f'Accuracy: ',accuracy_score(ypred_forest,y_test))
print(f'Precision: ',precision_score(ypred_forest,y_test))
print(f'Recall: ',recall_score(ypred_forest,y_test))
print(f'F1-score: ',f1_score(ypred_forest,y_test))

print()

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
ypred_ada = ada.predict(X_test)

print('Ada Accuracy Score:')
print(f'Accuracy: ',accuracy_score(ypred_ada,y_test))
print(f'Precision: ',precision_score(ypred_ada,y_test))
print(f'Recall: ',recall_score(ypred_ada,y_test))
print(f'F1-score: ',f1_score(ypred_ada,y_test))