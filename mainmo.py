import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from xgboost import XGBClassifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from pickle import load
st.header("REVIEW PREDICTION APP")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("merged_dataset1.csv")

# load the scaler
scaler = load(open('scaler.pkl', 'rb'))

# load model
#best_xgboost_model = XGBClassifier()
#best_xgboost_model.load_model("best_model.json")
model = XGBClassifier()
model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please enter review you want")
user_input = st.text_area("Review", default_value_goes_here)
             
if st.button('Make Prediction'):
    # transform the test dataset
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)
    print("final pred: ",prediction)

if(prediction == 0):
    st.write(f"Your review was a complaint ")
else:
     st.write(f"Your review was a suggestion ")
st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")


