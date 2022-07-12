import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier

data = pd.read_csv("final4k.csv")
X=data['Review'].values
Y=data['flair_sentiment'].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.35)
Vect=CountVectorizer()
Bow_train=Vect.fit_transform(x_train)
Bow_test=Vect.transform(x_test)

st_x= StandardScaler(with_mean=False)   
x_train= st_x.fit_transform(Bow_train)    
x_test= st_x.transform(Bow_test)

#load model
best_model = XGBClassifier()
best_model.load_model("bestmodel.json")
ac = accuracy_score(Y_test,y_pred)
cm= confusion_matrix(Y_test, y_pred)

class_label = ["Negative", "Positive"]
df = pd.DataFrame(cm, index = class_label, columns = class_label)
#%%

inputs =input()
prediction = best_model.predict(inputs)
print("final pred",prediction)
