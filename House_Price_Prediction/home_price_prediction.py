#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#loading csv file using pandas
df=pd.read_csv('/content/township_data.csv')
df

#taking null value as median 
import math
bedrooms_median=math.floor(df.bedrooms.median())
bedrooms_median
df.bedrooms=df.bedrooms.fillna(bedrooms_median)
df

#Loading model and training model
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

#predicting price using the model
reg.predict([[3000,3,40]])