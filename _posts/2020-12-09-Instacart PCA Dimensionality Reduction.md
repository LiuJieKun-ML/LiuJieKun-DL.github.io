---
layout:     post
title:      Instacart PCA Dimensionality Reduction
subtitle:   Life is Short, I Use Python
date:       2020-12-09
author:     JieKun Liu
header-img: img/the-first.png
catalog:   true
tags:
    - Machine Learning
---




import pandas as pd

# 1. Read data
order_products = pd.read_csv("/Users/liujiekun/Desktop/instacart/order_products__prior.csv")
products = pd.read_csv("/Users/liujiekun/Desktop/instacart/products.csv")
orders = pd.read_csv("/Users/liujiekun/Desktop/instacart/orders.csv")
aisles = pd.read_csv("/Users/liujiekun/Desktop/instacart/aisles.csv")


# 2. Consolidate table
tab1 = pd.merge(aisles,products,on=["aisle_id","aisle_id"])

tab2 = pd.merge(tab1,order_products,on=["product_id","product_id"])

tab3 = pd.merge(tab2,orders,on=["order_id","order_id"])

tab3.head()

# 3. Find the relationship between user_id and aisle
table = pd.crosstab(tab3["user_id"],tab3["aisle"])

table

data = table

# 4. PCA dimensionality reduction
from sklearn.decomposition import PCA

# 1. Instantiation
transfer = PCA(n_components = 0.95)
# 2. Call the fit_transform method
data_new = transfer.fit_transform(data)

data_new.shape



from sklearn.cluster import KMeans

estimator = KMeans(n_clusters=3)
estimator.fit(data_new)

y_predict = estimator.predict(data_new)

y_predict[:300]

# Model evaluation——silhouette_score
from sklearn.metrics import silhouette_score

result = silhouette_score(data_new,y_predict)




print(result)

