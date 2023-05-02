import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('weather.data')

x = df[['temp_max','temp_min','wind']].values
y = df['weather']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
sv = model.fit(x_train,y_train)

pickle.dump(sv, open('weather.pkl','wb'))