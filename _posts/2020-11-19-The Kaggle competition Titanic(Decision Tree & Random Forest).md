---
layout:     post
title:      Using Decision Tree & Random Forest to Predict The Survival of Titanic
subtitle:   Kaggle Competition
date:       2020-11-20
author:     JieKun Liu
header-img: img/the-first.png
catalog:   true
tags:
    - Machine Learning
---

By analyzing the information, the toughest task is to clean and find out the useful data.
Then let's Begin!!!
I ran this code in Jupyter Notebook, Colab is also a good choice!

import pandas as pd

# 1.
# Obtain Data
path = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"

titanic = pd.read_csv(path)

titanic.head()

# Filter Characteristic values and target values

x = titanic[["pclass","age","sex"]]

y = titanic["survived"]

x.head()

y.head()

# 2、Data Processing

## 1）Dealing with the Null 

x["age"].fillna(x["age"].mean(),inplace=True)

## 2)Convert to Dictionary

x = x.to_dict(orient="records")

from sklearn.model_selection import train_test_split

# 3、Divide data set

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=22)

# 4、Dictionary feature extraction

from sklearn.feature_extraction import DictVectorizer

from sklearn.tree import DecisionTreeClassifier,export_graphviz

transfer = DictVectorizer()

x_train = transfer.fit_transform(x_train)

x_test = transfer.transform(x_test)

## 3）Decision tree predictor

estimator = DecisionTreeClassifier(criterion="entropy",max_depth=8)

estimator.fit(x_train,y_train)

## 4）Model evaluation

# Method 1：Directly compare the true value and the predicted value

y_predict = estimator.predict(x_test)

print("y_predict\n", y_predict)

print("Directly compare the true value and the predicted value\n", y_test == y_predict)

# Method 2：Calculation accuracy

score = estimator.score(x_test, y_test)

print("Accuracy is :", score)

# Visualize decision tree

export_graphviz(estimator,out_file="titanic_tree.dot",feature_names=transfer.get_feature_names())

Random forest predicts the survival of the Titanic

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

estimator = RandomForestClassifier()

# Join grid search and cross validation

param_dict = {"n_estimators":[100,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}

estimator = GridSearchCV(estimator,param_grid=param_dict,cv=3)

estimator.fit(x_train,y_train)

# 5) Model Evaluation

#Method 1：Directly compare the true value and the predicted value

y_predict = estimator.predict(x_test)

print("y_predict\n",y_predict)

print("Directly compare the true value and the predicted value\n",y_test==y_predict)

# Method 2：Calculation accuracy

score = estimator.score(x_test,y_test)

print("Accuracy is :",score)

print("Best Parameter：\n",estimator.best_params_)

print("Best result：\n",estimator.best_score_)

print("Best Estimator：\n",estimator.best_estimator_)

print("Cross validation result：\n",estimator.cv_results_)


## Till now you will have a better understaning of the procedure of training a machine learning model

## Copyrights Reserved by JieKun Liu




