import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


data = pd.read_csv("merged_dataset1.csv")
X=data['Review'].values
Y=data['flair_sentiment'].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.35)
Vect=CountVectorizer()
Bow_train=Vect.fit_transform(x_train)
Bow_test=Vect.transform(x_test)

st_x= StandardScaler(with_mean=False)   
x_train= st_x.fit_transform(Bow_train)    
x_test= st_x.transform(Bow_test)


# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)

model.save_model("best_model.json")
