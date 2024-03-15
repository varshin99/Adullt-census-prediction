# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:44:05 2022

@author: akash
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import pickle

df = pd.read_csv('adult.csv')

df.drop('relationship',axis = 1,inplace = True)
df.drop('education',axis = 1,inplace = True)
df.drop('fnlwgt',axis = 1,inplace = True)

df['salary'] = le.fit_transform(df['salary'])

df["workclass"] = df["workclass"].replace(' ?',df["workclass"].mode()[0])
df["occupation"] = df["occupation"].replace(' ?',df["occupation"].mode()[0])
df["country"] = df["country"].replace(' ?',df["country"].mode()[0])

mean_encoded_race = df.groupby('race')['salary'].mean().to_dict()
df['race'] = df['race'].map(mean_encoded_race)

df['Male'] = pd.get_dummies(df['sex'],drop_first = True)
df.drop('sex',inplace = True,axis = 1)

mean_encoded_marital = df.groupby('marital-status')['salary'].mean().to_dict()
df['marital-status'] = df['marital-status'].map(mean_encoded_marital)

mean_encoded_workclass = df.groupby('workclass')['salary'].mean().to_dict()
df['workclass'] = df['workclass'].map(mean_encoded_workclass)

mean_encoded_occupation = df.groupby('occupation')['salary'].mean().to_dict()
df['occupation'] = df['occupation'].map(mean_encoded_occupation)

mean_encoded_country = df.groupby('country')['salary'].mean().to_dict()
df['country'] = df['country'].map(mean_encoded_country)

#plt.figure(figsize = (15,15))
#sns.heatmap(df.corr(),annot = True,cmap = 'BuGn')

#Feature separation
x = df.drop('salary',axis = 1)
y = df['salary']

#Traing and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)

#There are 2 categories where one occupies 75% and the other occupies 25%. to remove the imbalance we go through
# augmentation of the dataset.
from imblearn.combine import SMOTETomek
smtom = SMOTETomek(random_state = 101)
x_train_smtom,y_train_smtom = smtom.fit_resample(x_train,y_train)

#Augmented Dataset
df_new = pd.concat([x_train_smtom,y_train_smtom],axis = 1)

#Feature separation after augmentation
x = df_new.drop(['salary'],axis = 1)
y = df_new['salary']

#Training and splitting after augmentation
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,stratify = y,random_state = 101)

#MODEL_FINAL after hyper tuning of params.
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 65,max_depth = None,random_state = 101)

#Training
randomforest.fit(x_train,y_train)

#DUMP MODEL
pickle.dump(randomforest,open('model.pkl','wb'),protocol=pickle.HIGHEST_PROTOCOL)

#Prediction
#prediction_hyper = randomforest.predict(x_test)



#return predicted value


#print(confusion_matrix(y_test,prediction_hyper))
#print('\n')
#print(classification_report(y_test,prediction_hyper))
#print('\n')
#print(accuracy_score(y_test,prediction_hyper))

