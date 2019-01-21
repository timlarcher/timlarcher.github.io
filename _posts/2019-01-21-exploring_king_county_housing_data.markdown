---
layout: post
title:      "Exploring King County Housing Data"
date:       2019-01-21 17:22:46 -0500
permalink:  exploring_king_county_housing_data
---

Today's entry will walk you through an evaluation of housing data for King county, WA. This is the Seattle area and includes data from 2014 and 2015. I will propose additional analysis beyond what I did as an exercise.

The best way to experience this is to follow along and either cut and pase my code or write your own.

# Load the dataset and explore it.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df_orig = pd.read_csv('kc_house_data.csv')
print(df_orig.info())
print(df_orig.head())
print(df_orig.describe())
```

Info() will tell you about the columns; head() will print the first five rows; and describe() will provide statistics about the NUMERIC columns.

If you have followed along so far, this should make some sense. You can compare with your own observations.

Observations:
* waterfront, view, and yr_renovated contain NaNs. Waterfront appears to be boolean. yr_renovated probably is same as a zero in that column, meaning it hasn't been renovated or there is no record of it. View appears to be useless as almost all values are zero. For our purposes, id is of no value. lat and long and the 15 values could help with more sophisticated methods such as modelling the neighborhoods and surrounding values.
* Some datatypes need to be converted to clean it up.
* Grade and condition are sort of categorical, but since they are numeric and continuous it will be easy to graph them against sale price, for example.
* It is likely that many of the features are colinear.

For now I am going to drop all the columns that I don't see a need for now. I can always grab them back later but this makes it easier to see those items that might have a direct affect on prices right now.

Next we will clean up the datatypes, drop columns, fix NaNs, and get another feel for the data with histograms.

```
df = df_orig.drop(['date','id','view','lat','long','sqft_living15','sqft_lot15'], axis=1)
df.head()
df.waterfront.fillna(0,inplace=True)
df.yr_renovated.fillna(0,inplace=True)
df.waterfront = df.waterfront.astype('int64')

#sqft_basement: This was tricky because the 0.0 string wouldn't convert directly to an int. 
#The to_numeric produced NaNs for some values so those had to be eliminated then all types converted to int.

df.sqft_basement = pd.to_numeric(df.sqft_basement,errors='coerce')
df.sqft_basement.fillna(0,inplace=True)
df.sqft_basement = df.sqft_basement.astype('int64')
df.yr_renovated = df.yr_renovated.astype('int64')
print(df.isna().sum())
df.head()
```

Next the histograms:

```
df.hist(figsize=(20,18),bins=10)
```


