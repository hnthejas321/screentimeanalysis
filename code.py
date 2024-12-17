import pandas as pd
import numpy as np 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
data = pd.read_csv("apps12.csv")
print(data.head(21)) #print Data From DataBase print
print ("\nColumn Names:\n")
print (data.columns) # Print Column names
data.describe() 
data.info() 
data.head()
data.shape 
data.columns 
data.info()
data['App'].unique()
data['Notifications'].unique()
data['Times opened'].unique() 
figure1 = px.scatter(data, x='Notifications', y='Usage', color='Notifications',
title='Usagebased on Notifications')
figure1.show() #Scatter Plot Graph based on Notification
figure2 = px.scatter(data, x='Times opened', y='Usage', color='Times opened',title='Usage based on number of Times opened')
figure2.show() #Scatter Plot Graph based on No of times Opened
sns.barplot(data, x="App", y="Usage") 
plt.title('Usage Vs App')
plt.show() #Bar Graph of Based On Usage and App
sns.barplot(data, x="App", y="Notifications")
plt.title('Notifications Vs App')
plt.show() #Bar Graph of Based On Notification and App
sns.barplot(data, x="App", y="Times opened")
plt.title('Times opened Vs App')
plt.show() #Bar Graph of Based On Times Opend and App
data.isnull().sum() 
data.describe()
data.App.value_counts().plot.bar(); #Bar Graph Of Whatsapp and Instagram
plt.figure(figsize=(29,12))
sns.barplot(x=data['Date'],y=data['Usage'],hue=data['App'])
plt.title('Usage amount of apps', fontsize=20) 
plt.show()
plt.figure(figsize=(29,12))
sns.barplot(x=data['Date'],y=data['Notifications'],hue=data['App'])
plt.title('Number of notifications from the apps', fontsize=20) 
plt.show()
plt.figure(figsize=(10,8))
sns.lineplot(x=data['Date'],y=data['Times opened'],hue=data['App'])
plt.title('Times spent in each app during time') 
plt.xticks(rotation=45) 
plt.show()
plt.figure(figsize=(15,5))
sns.lineplot(x=data['Date'],y=data['Notifications'],hue=data['App'])
plt.title('Number of notifications from apps')
plt.xticks(rotation=45) 
plt.show()
sns.barplot(data, x="App", y="Times opened")
plt.title('Time opened vs App') 
plt.show()
notification=data.groupby('App').agg({'Usage':'mean','Notifications':'mean','Times opened':'mean'})
plt.figure (figsize=(10,10)) 
plt.subplot(311)
plt.title('1 - Mean value of the apllications usage')
plt.pie(x=notification.Usage,labels=notification.index,autopct='%d')
plt.subplot(312)
plt.title('2 - Mean value of the number of notifications taken from apps')
plt.pie(x=notification.Notifications ,labels=notification.index,autopct='%d')
plt.subplot(313)
plt.title('3 - Mean value of times spent on apps')
plt.pie(x=notification['Times opened'],labels=notification.index,autopct='%d')
plt.show()
# app ranking
ranking_data=pd.read_csv("C:/Users/Dell/OneDrive/Documents/Desktop/python mini project/dataset py/apprank.csv")
print(ranking_data.head(10))
ranking_data.head() 
ranking_data.shape
ranking_data.columns
ranking_data.isnull().sum()
ranking_data.info()
ranking_data['Rank 1'].value_counts()
ranking_data['Rank 2'].value_counts()
plt.figure(figsize=(15,4))
sns.barplot(x=ranking_data['Rank1'].value_counts().index,y=ranking_data['Rank1'].value_counts().values) 
plt.title("Number of apps use in Rank 1")
plt.show()
plt.figure(figsize=(15,4))
sns.barplot(x=ranking_data['Rank2'].value_counts().index,y=ranking_data['Rank2'].value_counts().values) 
plt.title("Number of apps use in Rank 2")
plt.show()
ranking_data['Rank 3'].value_counts()
plt.figure(figsize=(15,7))
sns.barplot(x=ranking_data['Rank3'].value_counts().index,y=ranking_data['Rank3'].value_counts().values) 
plt.title("Number of apps use in Rank 3")
plt.show()
data1 = pd.get_dummies(data=ranking_data,columns=['Rank 1', 'Rank 2', 'Rank3'])
data1.head()
data1.sample()
data1.shape
rank1=set(ranking_data['Rank 1'].unique())
rank2=set(ranking_data['Rank 2'].unique())
rank3=set(ranking_data['Rank 3'].unique())
app_list=rank1.union(rank2).union(rank3)
app_list 
col=[] 
for i in app_list: 
    for name in data1.columns:
        if i in name:
            col.append(name)
print(col)
ranking_data[i]=data1[col].sum(axis=1)
col=[]
ranking_data.drop(['Rank 1','Rank 2','Rank3'],axis=1,inplace=True) 
ranking_data.head() 
dic={}
g=ranking_data.drop(['Date '],axis=1) 
for i in g.columns:
    a=g[i].sum()
dic[i]=a
print(dic)
data2=pd.DataFrame.from_dict(dic, orient='index')
data2.index.rename('App',inplace=True)
data2.rename(columns={0:'sum'},inplace=True)
data2.columns
data3=data2.sort_values('sum',ascending=False)
data2.reset_index(inplace=True)
plt.figure(figsize=(20,10))
#px.pie(names=data2.App,values=data2 ['sum'])
plt.pie(x=data2['sum'],labels=data2.App,autopct='%d')
plt.show()
