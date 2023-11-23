# Dlithe_Internship
Introduction to our project
project name: price forecasting  for crops.
aim: predict the crop price rate based on the fluctuation of the market and the farmers can able to sell their crop with max profit.
By using the machine learning model we will  able to predict the price of the crop
The alogrithm like linear regression and classification models are been implemented 
the train and test algorithm is also been implement for the predication

PROGRAM CODE:
import pandas as pd
import numpy as np
import seaborn as sns
file=pd.read_csv("test2.csv")
print(file)
data2=df.copy()
data2=data2.dropna()
data2["Date"][1]
str=data2["Date"][1]
str2=str.split('-')
print(str)
print(str2)
print(str2[1])
dict={1:'jan',2:'feb',3:'march',4:'april',5:'may',6:'june',7:'july',8:'august',9:'september',10:'october',11:'november',12:'december'}
month_column=[]
for i in data2['Date']:
    str=i
    str2=str.split('-')
    month_column.append(dict[(int(str2[1]))])
data2["month_column"]=month_column
data2.head()
season_name=[]
day_week=[]
for r in data2["Date"]:
    str=r
    de=pd.Timestamp(r)
    day=de.dayofweek
    day_week.append(day)
data2=data2.drop('Date',axis=1)
data2=data2.head(23000)
import seaborn as sns
sns.boxplot(data2['Modal Price'])
Q1=np.percentile(data2['Modal Price'],25,interpolation ="midpoint")
Q3=np.percentile(data2['Modal Price'],75,interpolation ="midpoint")
IQR=Q3-Q1
upper = np.where(data2['Modal Price'] >= (Q3+1.5*IQR))
lower = np.where(data2['Modal Price']<= (Q1-1.5*IQR))
print(upper[0],lower[0])
data2.drop(upper[0],inplace=True)
data2.drop(lower[0],inplace=True)
print("New Shape:",data2.shape)
sns.boxplot(data2['Modal Price'])
df=data2.copy()
import plotly.express as px
a=sns.relplot(data=df,x="State",y="Modal Price",hue="season_name",kind='line')
a.set_xticklabels(rotation=90)
a=sns.relplot(data=df,x='District',y="Modal Price",hue="season_name",kind='line')
a.set_xticklabels(rotation=90)
fig=px.bar(df,x='District',y="Modal Price",color="season_name",height=400)
fig.show()
a=sns.relplot(data=df,x='Market',y='Modal Price',hue='season_name',kind='line')
a.set_xticklabels(rotation=90)
dist=(data2['Commodity'])
distset=set(dist)
dd=list(distset)
dictOfwords={dd[i] :i for i in range(0,len(dd))}
data2['Commodity']=data2['Commodity'].map(dictOfwords)
dist=(data2['State'])
distset=set(dist)
dd=list(distset)
dictOfwords={dd[i] :i for i in range(0,len(dd))}
data2['State']=data2['State'].map(dictOfwords)
dist=(data2['District'])
distset=set(dist)
dd=list(distset)
dictOfwords={dd[i] :i for i in range(0,len(dd))}
data2['District']=data2['District'].map(dictOfwords)
dist=(data2['Market'])
distset=set(dist)
dd=list(distset)
dictOfwords={dd[i] :i for i in range(0,len(dd))}
data2['Market']=data2['Market'].map(dictOfwords)
dist=(data2['month_column'])
distset=set(dist)
dd=list(distset)
dictOfwords={dd[i] :i for i in range(0,len(dd))}
data2['month_column']=data2['month_column'].map(dictOfwords)
dist=(data2['season_name'])
distset=set(dist)
dd=list(distset)
dictOfwords={dd[i] :i for i in range(0,len(dd))}
data2['season_name']=data2['season_name'].map(dictOfwords)
import matplotlib.pyplot as plt
dataplot=sns.heatmap(data2.corr(),cmap='YlGnBu',annot=True)
plt.show()
features=data2[['State', 'District', 'Market', 'Commodity', 'Min Price', 'Max Price',
        'month_column', 'season_name', 'day']]
labels=data2['Modal Price']
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,labels,test_size=0.2,random_state=2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
Xtest[0:1]
y_pred=regr.predict(Xtest)
from sklearn.metrics import r2_score
r2_score(Ytest,y_pred)
y_pred
Xtest[0:1]
user_input=[[24	,178,	712,	103	,700.0,	850.0,	0,	0,	0]]
regr.predict(user_input)














































