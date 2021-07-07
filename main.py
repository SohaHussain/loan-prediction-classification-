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

# 6. ApplicantIncome
# no missing value and is already of integer type

# 7.CoapplicantIncome

#I will create an array that contains random numbers, which are computed based on the mean Coapplicantincome value in regards
#to the standard deviation and is_null.

data = [train_df,test_df]
for dataset in data:
    mean=train_df['CoapplicantIncome'].mean()
    std=test_df['CoapplicantIncome'].std()
    is_null=dataset['CoapplicantIncome'].isnull().sum()

    # compute random numbers between mean , std, is_null
    rand_value=np.random.randint(mean-std, mean+std, size=is_null)

    # filling NaN values in age column with random values generated
    cai_slice= dataset['CoapplicantIncome'].copy()
    cai_slice[np.isnan(cai_slice)]=rand_value
    dataset['CoapplicantIncome']=cai_slice
    dataset['CoapplicantIncome']=train_df['CoapplicantIncome'].astype(int)

train_df['CoapplicantIncome'].isnull().sum()

# 8. LoanAmount
train_df['LoanAmount'].describe()

data = [train_df,test_df]
for dataset in data:
    mean=train_df['LoanAmount'].mean()
    std=test_df['LoanAmount'].std()
    is_null=dataset['LoanAmount'].isnull().sum()

    # compute random numbers between mean , std, is_null
    rand_value=np.random.randint(mean-std, mean+std, size=is_null)

    # filling NaN values in age column with random values generated
    cai_slice= dataset['LoanAmount'].copy()
    cai_slice[np.isnan(cai_slice)]=rand_value
    dataset['LoanAmount']=cai_slice
    dataset['LoanAmount']=train_df['LoanAmount'].astype(int)

train_df['LoanAmount'].isnull().sum()

# 9.Loan_Amount_Term

train_df['Loan_Amount_Term'].unique()
train_df['Loan_Amount_Term'].describe()

data = [train_df,test_df]
for dataset in data:
    mean=train_df['Loan_Amount_Term'].mean()
    std=test_df['Loan_Amount_Term'].std()
    is_null=dataset['Loan_Amount_Term'].isnull().sum()

    # compute random numbers between mean , std, is_null
    rand_value=np.random.randint(mean-std, mean+std, size=is_null)

    # filling NaN values in age column with random values generated
    cai_slice= dataset['Loan_Amount_Term'].copy()
    cai_slice[np.isnan(cai_slice)]=rand_value
    dataset['Loan_Amount_Term']=cai_slice
    dataset['Loan_Amount_Term']=train_df['Loan_Amount_Term'].astype(int)

train_df['Loan_Amount_Term'].isnull().sum()

# 10. Credit_History

train_df['Credit_History'].unique()
train_df['Credit_History'].describe()

common_value=1
data=[train_df,test_df]
for dataset in data:
    dataset['Credit_History']=dataset['Credit_History'].fillna(common_value)
    dataset['Credit_History']=train_df['Credit_History'].astype(int)

# 11. Property_Area

train_df['Property_Area'].unique()
pa={'Rural':0,'Semiurban':1,'Urban':2}
data=[train_df,test_df]

for dataset in data:
    dataset['Property_Area']=dataset['Property_Area'].map(pa)

# 12. Loan_Status
train_df['Loan_Status'].unique()
train_df['Loan_Status']=train_df['Loan_Status'].replace(['Y','N'],[1,0])

train_df.head()
train_df.info()

sns.barplot(x='Gender',y='Loan_Status',data=train_df)



