import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
st.header("REVIEW PREDICTION APP")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("merged_dataset1.csv")

# load model
best_xgboost_model = xgb.XGBClassifier()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please enter review you want")
user_input = st.text_area("Review", default_value_goes_here)
             
if st.button('Make Prediction'):
    prediction = best_xgboost_model.predict(user_input)
    print("final pred: ",prediction)

if(prediction == 0):
    st.write(f"Your review was a complaint ")
else:
     st.write(f"Your review was a suggestion ")
st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")


