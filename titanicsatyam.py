#importing important libraries

import pandas as pd
import random as rnd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#importing various algorithms 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#importing train and test file

train_df=pd.read_csv('C:/Users/satyam chauhan/Desktop/titanic/train.csv')        # you need to change address of train file here 
test_df=pd.read_csv('C:/Users/satyam chauhan/Desktop/titanic/test.csv')          # you need to change address of test file here

train_df.info()

train_df.describe(include=['O'])

g=sn.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=20)
sn.plt.show()

grid=sn.FacetGrid(train_df,col='Pclass',hue='Survived',size=2.2,aspect=1.2,legend_out=False)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
sn.plt.show()
grid.add_legend();
sn.plt.show()

grid = sn.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.2, legend_out=False)
grid.map(sn.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
sn.plt.plot()

grid = sn.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.2, legend_out=False)
grid.map(sn.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
sn.plt.plot()
sn.plt.show()

grid = sn.FacetGrid(train_df, hue='Embarked', size=2.2, aspect=1.2, legend_out=False)
grid.map(sn.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
sn.plt.show()

grid=sn.FacetGrid(train_df,col='Embarked', hue='Survived',palette={0:'k',1:'w'},size=2.2,aspect=1.2,legend_out=False)
grid.map(sn.barplot,'Sex','Fare',alpha=.5,ci=None)
grid.add_legend()
sn.plt.show()

train_df=train_df.drop(['Ticket','Cabin'],axis=1)
test_df=test_df.drop(['Ticket','Cabin'],axis=1)
train_df['Title'] = train_df.Name.str.extract('(\w+\.)', expand=False)
sn.barplot(hue='Survived',x='Age',y='Title',data=train_df,ci=False)
sn.plt.show()

test_df['Title']=test_df.Name.str.extract('(\w+\.)',expand=False)
train_df=train_df.drop(['Name','PassengerId'],axis=1)
test_df=test_df.drop(['Name'],axis=1)
test_df.describe(include=['O'])
train_df['Gender']=train_df['Sex'].map({'female':1,'male':0}).astype(int)
train_df.loc[:,['Gender','Sex']].head()
test_df['Gender']=test_df['Sex'].map({'female':1,'male':0}).astype(int)
train_df=train_df.drop(['Sex'],axis=1)
test_df=test_df.drop(['Sex'],axis=1)
guess_ages=np.zeros((2,3))
guess_ages

for i in range(0,2):
	for j in range(0,3):
		guess_df=train_df[(train_df['Gender']==i)&(train_df['Pclass']==j+1)]['Age'].dropna()
		age_guess=guess_df.median()
		guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5
guess_ages

train_df['AgeFill']=train_df['Age']

for i in range(0,2):
	for j in range(0,3):
		train_df.loc[(train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1),'AgeFill']=guess_ages[i,j]

train_df[train_df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)

guess_ages=np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
		guess_df=test_df[(test_df['Gender']==i)&(test_df['Pclass']==j+1)]['Age'].dropna()
		age_guess=guess_df.median()
		guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5

test_df['AgeFill']=test_df['Age']

for i in range(0,2):
	for j in range(0,3):
		test_df.loc[(test_df.Age.isnull())&(test_df.Gender==i)&(test_df.Pclass==j+1),'AgeFill']=guess_ages[i,j]

test_df[test_df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)

train_df = train_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)
train_df.head()

train_df['FamilySize']=train_df['SibSp']+train_df['Parch']
test_df['FamilySize']=test_df['SibSp']+test_df['Parch']
train_df.loc[:,['Parch','SibSp','FamilySize']].head(10)
test_df['Age*Class']=test_df.AgeFill*test_df.Pclass

train_df['Age*Class']=train_df.AgeFill*train_df.Pclass
freq_port=train_df.Embarked.dropna().mode()
freq_port
freq_port=train_df.Embarked.dropna().mode()[0]
freq_port

import statistics as ss

train_df['EmbarkedFill']=train_df['Embarked']

train_df.loc[train_df['Embarked'].isnull(),'EmbarkedFill']=freq_port
train_df[train_df['Embarked'].isnull()][['EmbarkedFill','Embarked']]
test_df['EmbarkedFill']=test_df['Embarked']

train_df = train_df.drop(['Embarked'], axis=1)
test_df = test_df.drop(['Embarked'], axis=1)
train_df.head()

ports=list(enumerate(np.unique(train_df['EmbarkedFill'])))

ports_dict={name:i for i,name in ports}

train_df['Port']=train_df.EmbarkedFill.map(lambda x:ports_dict[x]).astype(int)

ports=list(enumerate(np.unique(test_df['EmbarkedFill'])))

ports_dict={name:i for i,name in ports}
test_df['Port']=test_df.EmbarkedFill.map(lambda x:ports_dict[x]).astype(int)
train_df[['EmbarkedFill','Port']].head(10)
Titles=list(enumerate(np.unique(train_df['Title'])))
title_dict={name:i for i,name in Titles}
train_df['TitleBand']=train_df.Title.map(lambda x:title_dict[x]).astype(int)
Titles=list(enumerate(np.unique(test_df['Title'])))
title_dict={name:i for i,name in Titles}
test_df['TitleBand']=test_df.Title.map(lambda x:title_dict[x]).astype(int)
train_df[['Title','TitleBand']].head(10)
train_df=train_df.drop(['EmbarkedFill','Title'],axis=1)
test_df=test_df.drop(['EmbarkedFill','Title'],axis=1)
train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace=True)
train_df['Fare']=train_df['Fare'].round(2)
test_df['Fare']=test_df['Fare'].round(2)
test_df.head(10)
print ('----------------')
x_train=train_df.drop('Survived',axis=1)
y_train=train_df['Survived']
x_test=test_df.drop('PassengerId',axis=1).copy()
x_train.shape,y_train.shape,x_test.shape

logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
logreg.score(x_train,y_train)
coeff_df=pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns=['Features']
coeff_df['Correlation']=pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation',ascending=False)

random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train,y_train)
y_pred=random_forest.predict(x_test)
random_forest.score(x_train,y_train)

submission=pd.DataFrame({
	'PassengerId':test_df['PassengerId'],
	'Survived':y_pred
	})
submission.to_csv('C:/Users/satyam chauhan/Desktop/titanic/submission.csv',index=False)
