# IMPORTING LIBRARIES

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from bokeh.plotting import figure,show


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# GETTING DATA

test_df=pd.read_csv('test_lp.csv')
train_df=pd.read_csv('train_lp.csv')

# DATA ANALYSIS

train_df.info()
# the training set has 614 examples and 12 features + 1 target feature (Loan_status) 4 features are floats,
# 1 integer and 8 objects

train_df.describe()

# lets take a more detailed look at what data is missing

total=train_df.isnull().sum().sort_values(ascending=False)
per1=train_df.isnull().sum()/train_df.isnull().count() *100
percent=(round(per1,1)).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['total','%'])
missing_data.head(13)

train_df.columns.values

# DATA PROCESSING

train_df=train_df.drop(['Loan_ID'],axis=1)

# 1. Gender
train_df['Gender'].unique()
# since gender has 13 missing values we will fill them with the most common one

train_df['Gender'].describe()
common_value='Male'
data=[train_df,test_df]
for dataset in data:
    dataset['Gender']=dataset['Gender'].fillna(common_value)

# converting gender
genders={'Male':0,'Female':1}
data=[train_df,test_df]

for dataset in data:
    dataset['Gender']=dataset['Gender'].map(genders)

# 2. Married
train_df['Married'].unique()
# since married has 3 missing values we will fill them with the most common one

train_df['Married'].describe()
common_value='Yes'
data=[train_df,test_df]
for dataset in data:
    dataset['Married']=dataset['Married'].fillna(common_value)

# converting Married

ms={'No':0,'Yes':1}
data=[train_df,test_df]

for dataset in data:
    dataset['Married']=dataset['Married'].map(ms)

# 3. Dependents

train_df['Dependents'].unique()

# dependents has 15 missing values

train_df['Dependents'].describe()

common_value='0'
data=[train_df,test_df]
for dataset in data:
    dataset['Dependents']=dataset['Dependents'].fillna(common_value)

# converting Dependents

dep={'0':0,'1':1,'2':2,'3+':3}
data=[train_df,test_df]

for dataset in data:
    dataset['Dependents']=dataset['Dependents'].map(dep)

# 4. Education

train_df['Education'].unique()

# education has no missing values so we will convert it
# converting education

edu={'Graduate':1,'Not Graduate':0}
data=[train_df,test_df]

for dataset in data:
    dataset['Education']=dataset['Education'].map(edu)

# 5. Self_Employed

train_df['Self_Employed'].unique()

# self employed has 32 missing values

train_df['Self_Employed'].describe()

common_value='No'
data=[train_df,test_df]
for dataset in data:
    dataset['Self_Employed']=dataset['Self_Employed'].fillna(common_value)

# converting self employed

se={'Yes':1,'No':0}
data=[train_df,test_df]

for dataset in data:
    dataset['Self_Employed']=dataset['Self_Employed'].map(se)





