import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as sl
import matplotlib.pyplot as plt

def clean_sm(x):
    return np.where(x == 1, 1, 0)
data = {
    'col1': [1, 5, 1,0,6,7,8,9,1,2,3,5],
    'col2': [0, 1, 0,4,5,0,6,7,8,9,5,1]
}

s=pd.read_csv('social_media_usage.csv')

ss=pd.DataFrame()
ss['sm_li']=s['web1h'].apply(clean_sm)
ss['income']=s['income'].apply(lambda x: x if x <98 else float('nan'))
ss['is_parent']=s['par'].apply(clean_sm)
ss['is_married']=s['marital'].apply(clean_sm)
ss['educ2']=s['educ2'].apply(lambda x: x if x <98 else float('nan'))
ss['is_female']=s['gender'].apply(lambda x: 1 if x== 2 else 0)
ss['age_years']=s['age'].apply(lambda x: x if x != 98 else float('nan'))
ss=ss.dropna()

y = ss['sm_li'] 
X = ss.drop(columns=['sm_li']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm=confusion_matrix(y_test, y_pred)

classification_report=classification_report(y_test, y_pred, target_names=["Is not a LinkedIn user (0)", "Is a LinkedIn user (1)"])

print("\n\nClassification Report (sklearn) :\n", classification_report)

case_1=pd.DataFrame(model.predict_proba(pd.DataFrame([{
  'income': 8,
  'is_parent': 0,
  'is_married': 1,
  'educ2': 7.0,
  'is_female': 1,
  'age_years': 42.0
}])), columns=['Negative Class Probability','Positive Class Probability'],index=['Case 1'])
case_2=pd.DataFrame(model.predict_proba(pd.DataFrame([{
  'income': 8,
  'is_parent': 0,
  'is_married': 1,
  'educ2': 7.0,
  'is_female': 1,
  'age_years': 82.0
}])), columns=['Negative Class Probability','Positive Class Probability'],index=['Case 2'])
pd.concat([case_1,case_2])

