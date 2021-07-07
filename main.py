# IMPORTING LIBRARIES

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


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