#==========================================#
# KAGGLE CHALLENGE - TITANIC DATA
#==========================================#
# Created by: Lena Nguyen - April 13, 2015
#==========================================#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

#=============================#
# READ, EXPLORE, PROCESS DATA #
#=============================#

# READ IN DATA
train = pd.read_csv('../train.csv')
# data downloaded directly from Kaggle
len(train)  # 891 obs
train['type'] = 'train'  # just in case for the split later

test = pd.read_csv('../test.csv')
len(test)  # 418 obs
test['type'] = 'test'  # just in case for the split later

# Append train and test for full titanic dataset
# This will allow me to clean the test dataset also
titanic = train.append(test, ignore_index=True)
len(titanic)
titanic.type.value_counts()

# QUICK LOOK
titanic.head(20)
titanic.describe()

# CHECK FOR MISSING VALUES
titanic.isnull().sum()
#- Age has 263 missing values total
#- Cabin has 1014 missing values. That's a lot of missing values.
#- Probably will not use Cabin variable in the model

# Characteristics of missing age values (is it random?)
titanic.groupby(['Sex', 'Pclass']).Age.apply(lambda x: x.isnull().sum()) / titanic.groupby(['Sex', 'Pclass']).Age.count()
#- It appears that more age values are missing in the 3rd class passengers

#==================#
# DATA EXPLORATION #
#==================#
# Note: this only looks at training data

# Look at how many people survived
train.Survived.value_counts().plot(kind='barh', color="blue", alpha=.75)
plt.show()
# Not a lot at all

# Look at gender split of survivors (1 - female, 0 - male)
train.groupby('Sex').Survived.value_counts() / train.groupby('Sex').Survived.count()
# By proportion, more women survived than men (74% versus 19%)

# Graph of Percent of men that survived
(train.Survived[train.Sex == 0].value_counts()/train.Sex[train.Sex == 0].count()).plot(kind='bar', color='b',label='Male')
plt.xlabel('Survived')
plt.ylabel('Percentage of all male passengers')
plt.show()

# Graph of Percent of women that survived
(train.Survived[train.Sex == 1].value_counts()/train.Sex[train.Sex == 1].count()).plot(kind='bar', color='r',label='Female')
plt.xlabel('Survived')
plt.ylabel('Percentage of all female passengers')
plt.show()

# Split by class
train.groupby('Pclass').Survived.value_counts() / train.groupby('Pclass').Survived.count()
# Class has an effect on the likelihood of survival. More first class people survived.

train.groupby('SibSp').Survived.value_counts() / train.groupby('SibSp').Survived.count()
# The people that had smaller families (SibSp=1-2) are more likely to survive
# However, having no sibling or spouse means you'll be less likely to survive
# The larger the family, the less likely they'll survive

train.groupby('Parch').Survived.value_counts() / train.groupby('Parch').Survived.count()
# Same as seen in SibSp variable.
# Trend of no family members or large families are less likely to survive.

#======================================#
# CLEAN/PROCESS DATA/ADD NEW FEATURES  #
#======================================#

# Change sex varialbe into binary variable
titanic['Sex'] = np.where(titanic.Sex == 'female', 1, 0)
titanic.head(5)  # Check it worked

# Change Embarked variable into numbers for random forest
titanic['Embarked'] = pd.factorize(titanic['Embarked'])[0]
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode())

# Fill in NaN value for fare with mean of that class's fare price
titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'][titanic['Pclass'] == 3].mean())

# Find passenger's title using regex
titanic['Title'] = titanic['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
titanic['Title'] = titanic['Title'].astype(str)
titanic.head(10)  # Check it worked

# Count how many times each title occurs
titanic['Title'].apply(pd.value_counts).sum()

# Group low occuring counts together or into related groups
titanic['Title'][titanic.Title.isin(['Ms','Mlle'])] = 'Miss'
titanic['Title'][titanic.Title == 'Mme'] = 'Mrs'
titanic['Title'][titanic.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir', 'Jonkheer'])] = 'Sir'
# Jonkheer is a belgian title for male nobility
titanic['Title'][titanic.Title.isin(['Dona', 'the Countess'])] = 'Lady'
titanic['Title'][(titanic.Title == 'Dr') & (titanic.Sex == 0)] = 'Mr'
titanic['Title'][(titanic.Title == 'Dr') & (titanic.Sex == 1)] = 'Mrs'

titanic['Title'].apply(pd.value_counts).sum()  # Check it worked

# Change title into a numeric variable for random forest
titanic['Title_id'] = pd.factorize(titanic['Title'])[0]

# If family size affects survival, then we should try to figure out a passenger's
# family size using these variables: Surname, SibSp, Parch
# Add sibling/spouse, paraents/children, and the person themselves
titanic['Fam_size'] = titanic['SibSp'] + titanic['Parch'] + 1

# Fill in missing values for age by using a random forest regressor
df = titanic[['Age', 'Sex', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Title_id', 'Pclass']]

X = df.loc[(titanic.Age.notnull())].values[:,1:]
X = pd.DataFrame(data=X, columns=[list(df.columns.values[1:])])
y = df.loc[(titanic.Age.notnull())].values[:, 0]
y = pd.DataFrame(data=y,columns=['Age'])

rtr = ensemble.RandomForestRegressor()
rtr.fit(X,y)

# See how the age prediction did
preds = rtr.predict(X)
mean_squared_error(y, preds)  # 55 currently

resid = preds - y['Age']
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Age")
plt.ylabel("Residuals")
plt.show()

# Predict missing ages
predictedAges = rtr.predict(df.loc[(df.Age.isnull())].values[:, 1:])
predictedAges.mean()
predictedAges.max()
predictedAges.min()

# Replace predicted values in titanic df
titanic.loc[(titanic.Age.isnull()), 'Age'] = predictedAges

titanic.isnull().sum()

# Simple clean of missing values for Fare since there is only 1
titanic[titanic.Fare.isnull()]
# Replace with median Fare for their class
titanic.loc[titanic.Fare.isnull(),'Fare'] = titanic['Fare'][titanic['Pclass'] == 3].median()
titanic.isnull().sum()

# Make bins for variables. Might be more useful than just the continuous variable
titanic['Age_bin'] = pd.qcut(titanic['Age'], 3)
titanic['Age_bin_id'] = pd.factorize(titanic['Age_bin'])[0]  # Make bins into a categorial variable

#==============#
# CREATE MODEL #
#==============#

# Splice titanic data frame into test and train df
train = titanic.iloc[:891,:]
train.head(10)
train.type.value_counts()  # Check that the slice was correct
del train['type']

# Replace test df with new one
test = titanic.iloc[891:, :]
test.head(10)
test.type.value_counts()  # Check that the slice was correct
del test['type']

train.columns.values
cols = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Title_id', 'Fam_size']

X_train = train[cols].values
Y_train = train['Survived'].values
X_test = test[cols].values

rfc = ensemble.RandomForestClassifier()
# Parameters tuning
param_grid = [{'criterion':['gini','entropy'],'min_samples_leaf':range(1,10),'n_estimators':range(1,20),'max_depth':range(1,20)}]
grid = GridSearchCV(rfc, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, Y_train)  # Takes a while, good time for a bathroom break

grid.best_score_
grid.best_params_

rfc = ensemble.RandomForestClassifier(criterion='entropy', max_depth=19, min_samples_leaf=6, n_estimators=17)
np.mean(cross_val_score(rfc, X_train, Y_train, cv=10, scoring='roc_auc'))
# 0.87388 accuracy

rfc.fit(X_train, Y_train)
rfc.feature_importances_
Y_pred = rfc.predict(X_train)
confusion_matrix(Y_train, Y_pred)

#==================#
# PREDICT SURVIVAL #
#==================#

Predicted = rfc.predict(X_test)
test['Survived'] = Predicted

test.to_csv('../predict.csv',columns=['PassengerId', 'Survived'], header=True, index=False)
# 0.79426 score on kaggle with this model

#===================================#
# EXPLORE FALSE POSITIVES/NEGATIVES #
#===================================#

# Look at characteristics of false positives/negatives
Y_pred = pd.DataFrame(data=Y_pred, columns=['Prediction'])
preds = train.merge(Y_pred, left_index=True, right_index=True)
preds['FN'] = np.where(preds['Prediction'] > preds['Survived'], 1, 0)  # false negatives
preds['FP'] = np.where(preds['Prediction'] < preds['Survived'], 1, 0)  # false positives

# See what is throwing off the model by looking at the features
preds.groupby('Sex').FP.value_counts() / preds.groupby('Sex').FP.count()
# More false positives for men
preds.groupby('Pclass').FP.value_counts() / preds.groupby('Pclass').FP.count()
# More false poitives for 1st/3rd class
preds.groupby('SibSp').FP.value_counts() / preds.groupby('SibSp').FP.count()
# Sibsp of 4
preds.groupby('Fam_size').FP.value_counts() / preds.groupby('Fam_size').FP.count()
# Family size of 7, Family size of 1
preds.groupby('Age_bin').FP.value_counts() / preds.groupby('Age_bin').FP.count()
# More false negatives for adults
preds.groupby('Title').FP.value_counts() / preds.groupby('Title').FP.count()

# TO DO:
# Make the age imputations more accurate or simply (Median age by title?)
# Split up sibling/spouse variable?
# Read this: https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md
