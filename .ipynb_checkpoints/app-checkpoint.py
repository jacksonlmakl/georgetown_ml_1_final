import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as sl

# Clean the social media usage data
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Load and preprocess dataset
@sl.cache_data
def load_data():
    s = pd.read_csv('social_media_usage.csv')
    ss=pd.DataFrame()
    ss['sm_li']=s['web1h'].apply(lambda x: x if x not in ('8','9') else float('nan'))
    ss['income']=s['income'].apply(lambda x: x if x <98 else float('nan'))
    ss['is_parent']=s['par'].apply(lambda x: x if x <8 else float('nan'))
    ss['is_married']=s['marital'].apply(lambda x: x if x <8 else float('nan'))
    ss['educ2']=s['educ2'].apply(lambda x: x if x <98 else float('nan'))
    ss['is_female']=s['gender'].apply(lambda x: x if x < 98 else float('nan'))
    ss['age_years']=s['age'].apply(lambda x: x if x != 98 else float('nan'))
    ss=ss.dropna()
    
    
    ss['sm_li']=ss['sm_li'].apply(clean_sm)
    ss['is_parent']=ss['is_parent'].apply(clean_sm)
    ss['is_married']=ss['is_married'].apply(clean_sm)
    ss['is_female']=ss['is_female'].apply(lambda x: 1 if x == 2 else 0)
    return ss

# Train the logistic regression model
@sl.cache_data
def train_model():
    ss = load_data()
    y = ss['sm_li']
    X = ss.drop(columns=['sm_li'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit UI
sl.title("LinkedIn Usage Predictor")
sl.write("Enter your information below to predict if you are likely to be a LinkedIn user and see the probability.")

# Get user inputs
income = sl.slider("Income Level (1-10)", min_value=1, max_value=10, value=5)
is_parent = sl.radio("Are you a parent?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
is_married = sl.radio("Are you married?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
educ2 = sl.slider("Education Level (1-8)", min_value=1, max_value=8, value=4)
is_female = sl.radio("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
age_years = sl.number_input("Age (in years)", min_value=1, max_value=120, value=30)

# Train the model and predict
model, X_test, y_test = train_model()
user_input = pd.DataFrame([{
    'income': income,
    'is_parent': is_parent,
    'is_married': is_married,
    'educ2': educ2,
    'is_female': is_female,
    'age_years': age_years
}])
probabilities = model.predict_proba(user_input)[0]
prediction = model.predict(user_input)[0]

# Display prediction results
sl.subheader("Prediction Result")
sl.write("You are classified as a LinkedIn user." if prediction == 1 else "You are classified as not a LinkedIn user.")

sl.subheader("Probability")
sl.write(f"Probability of LinkedIn usage: {probabilities[1]:.2%}")
sl.write(f"Probability of not using LinkedIn: {probabilities[0]:.2%}")

# Show evaluation metrics
sl.subheader("Model Performance")
y_pred = model.predict(X_test)
sl.write("Classification Report:")
sl.table(pd.DataFrame(classification_report(y_test, y_pred, target_names=["Not a LinkedIn user (0)", "LinkedIn user (1)"],output_dict=True)))

