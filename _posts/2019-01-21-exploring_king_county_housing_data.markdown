---
layout: post
title:      "Exploring King County Housing Data"
date:       2019-01-21 17:22:46 -0500
permalink:  exploring_king_county_housing_data
---

Today's entry will walk you through an evaluation of housing data for King county, WA. This is the Seattle area and includes data from 2014 and 2015. I will propose additional analysis beyond what I did as an exercise.

The best way to experience this is to follow along and either cut and pase my code or write your own.

## Load the dataset and explore it.

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

Bedrooms and bathrooms seem pretty normally distributed with a slight skew, and some definate outliers. 33 bedrooms?

About 95% of houses don't have a renovation date so that is going to be a pretty useless column.

Note that grade and condition seem to be pretty highly correlated, as you would expect.

Next we'll check for colinearity in the features using Seaborn.

```
import seaborn as sns

sns.set(style='white') # Visualization style
corr = df.corr() # Create covariance matrix from the dataframe.
# Generate a mask the size of our covariance matrix:
mask = np.zeros_like(corr,dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette( 220, 10, as_cmap=True )
sns.heatmap( corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
			 square=True, linewidths=0.5, cbar_kws={'shrink': 0.5} )
```

This creates a heatmap. More information is available [https://seaborn.pydata.org/examples/many_pairwise_correlations.html] here.

From this it appears that our assumption above was true - that many of these features are colinear. The real question will be which ones are the best predictor of house value. Bedrooms, bathrooms, sqft_living, grade, sqft_living, sqft_basement, and floors all seem highly colinear. Interestingly condition and grade are not that colinear which is counter intuitive I think.

These are the most likely fields to base our train test model. It might be interesting to split out the waterfront from non-waterfront and get an idea of what value waterfront adds to otherwise similar houses. Ditto for breaking the data down by zip code.

We'll use joint plots to look for linear relationships between these and the sales price. For fun I am going to do one for each feature even though they might seem to have no relationship just to see what they show. I am also treating what might be considered categorical data (such as condition and grade) as numerical since they are numeric already and they can be plotted against price.

We'll do scaling later when we perform the multiple regression modelling.

The 33 bedrooms is causing a problem so we'll eliminate that row.


```
df = df[df.bedrooms < 33]
for c in df.drop(['price'],axis=1):
    sns.jointplot( x=c, y='price', data=df, kind='reg', joint_kws={'line_kws':{'color':'green'}})
plt.legend()
plt.show()
```

Several of these features provide no relationship with the price. These include year built and year renovated and waterfront.

That means we will go ahead with these fields: 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade'. The others don't seem to add anything to determining a value.

First we'll follow up the above with a complete regression with the full set of features so that we can compare and verify our observations.

```
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import scipy.stats as stats

def complete_regression( df ):
    """
        Performs a complete OLS and plots it.
            df = pandas dataframe
        returns stats for each feature:
            slope
            y-int
            p-value
            r-squared
            normality
        """
    statistics = [['feature', 'slope', 'y-int', 'p-value', 'r-squared', 'normality']]
    for i, c in enumerate(df.drop(columns=['price']).columns):
        f = 'price~' + c
        m = smf.ols(formula=f, data=df).fit()
        X = pd.DataFrame({c: [df[c].min(), df[c].max()] })
        y = m.predict(X)
        statistics.append( [c, m.params[1], m.params[0], m.pvalues[1], 
                            m.rsquared, sms.jarque_bera(m.resid)[0]] )
        df.plot(kind='scatter', x=c, y='price')
        plt.plot(X,y,c='red',linewidth=2)
    
        fig = plt.figure(figsize=(15,9))
        fig = sm.graphics.plot_regress_exog(m,c,fig=fig)
    
        fig = sm.graphics.qqplot(m.resid, dist=stats.norm, line='45', fit=True)
        plt.show()
    return statistics
# end 

s = complete_regression(df)
pd.DataFrame(s)
```

This produces three sets of graphs for each independent variable:
1. A scatter-line graph that shows the data points and the regression line.
2. Four separate graphs from plot_regress_exog.
3. A Q-Q plot.

The first is fairly obvious. The second one is beyond the scope of this post. The third plots sample vs. theoretical values on the x-axis. A linear plot indicates a normal distribution. A curved plot typifies a non-normal distribution of data.

Based on the above results, sqft_living, bathrooms, grade, and sqft_above are the most likely based on r-squared to be good inidcators of price. Note that the normality is very low for all of these, so we will try to normalize the log of the data and rerun for those four features. The others are really not worth persuing at this point.

```
df_new = df_orig[['price','sqft_living','sqft_living15','bathrooms','grade','sqft_above']]
print(df_new.head())

# Now do logs
df_new.sqft_living = np.log(df_new.sqft_living)
df_new.sqft_living15 = np.log(df_new.sqft_living15)
df_new.sqft_above = np.log(df_new.sqft_above)

# And Norm
df_new.sqft_living = (df_new.sqft_living - df_new.sqft_living.mean()) / (df_new.sqft_living.max() - df_new.sqft_living.min())
df_new.sqft_living15 = (df_new.sqft_living15 - df_new.sqft_living15.mean()) / (df_new.sqft_living15.max() - df_new.sqft_living15.min())
df_new.sqft_above = (df_new.sqft_above - df_new.sqft_above.mean()) / (df_new.sqft_above.max() - df_new.sqft_above.min())

# And replot the above
s = complete_regression(df_new)
pd.DataFrame(s)
```

Sqft_living15 seems to be an even better predictor than the sqft_living of the house itself. That kind of reinforces the old adage that "location, location, location" is the real estate mantra.

Grade still seems to be the best predictor and that makes sense because it is based on a human judgement with presumably some talent and using all the attributes including soft ones such as style, design, condition, floor plans, colors, etc. and bringing them together into a single feature.

Next step is to perform an initial multi-feature ordinary least squares and analyze the results.

```
# Step 10: Perform multi-featured OLS

from statsmodels.formula.api import ols

features = df_new.drop(columns='price')
formula = "price" + "~" + "+".join(features.columns)
model = ols( formula=formula, data=df_new ).fit()
model.summary()
```

Observations:

* Skew and Kurtosis are both high, which we had already observed.
* Based on the p-value these all seem to have some predictive value.
* Based on the r-squared value, these features explain about half of the price.
* We already proposed that GRADE was probably the best predictor, so we will test that with two different methods:

    1. stepwise selection and
    2. feature ranking with recursive feature elimination.

We can then compare the results and check them against our predictions.

```
# Step 12: Stepwise Selection

import statsmodels.api as sm

def stepwise_selection( X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True ):
    """
        performs stepwise selection with forward and backward steps.
        X = pandas dataframe of feature values
        y = target values
        initial_list = list of features to include from beginning
        threshold_in = lowest pval to include
        threshold_out = highest pval before removing
        verbose = print actions
    """
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for c in excluded:
            mod = sm.OLS( y, sm.add_constant(pd.DataFrame(X[included+[c]]))).fit()
            new_pval[c] = mod.pvalues[c]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        mod = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = mod.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    # end while
    return included
# end

selected = stepwise_selection( X=features, y=df_new.price )
print(selected)



Add  sqft_living15                  with p-value 0.0
Add  sqft_above                     with p-value 0.0
Add  sqft_living                    with p-value 0.0
Drop sqft_above                     with p-value 0.919488
Add  grade                          with p-value 0.0
Add  sqft_above                     with p-value 7.61918e-55
Add  bathrooms                      with p-value 8.23871e-08
['sqft_living15', 'sqft_living', 'grade', 'sqft_above', 'bathrooms']
```

Observations on Stepwise Selection:

Again, these all gave very good p-values. After looking at these I am going to add bedrooms back in and see where it comes up because bathrooms gave a good result.
I am going to drop _above and _living15 because they are not as good as _living.

```
df_new = df_orig[['price','sqft_living','grade','bedrooms','bathrooms']]
print(df_new.head())

features = df_new.drop(columns='price')
selected = stepwise_selection( X=features, y=df_new.price )
print(selected)
      price  sqft_living  grade  bedrooms  bathrooms
0  221900.0         1180      7         3       1.00
1  538000.0         2570      7         3       2.25
2  180000.0          770      6         2       1.00
3  604000.0         1960      7         4       3.00
4  510000.0         1680      8         3       2.00
Add  bathrooms                      with p-value 0.0
Add  sqft_living                    with p-value 0.0
Drop bathrooms                      with p-value 0.149588
Add  grade                          with p-value 0.0
Add  bedrooms                       with p-value 1.68881e-79
Add  bathrooms                      with p-value 6.76888e-15
['sqft_living', 'grade', 'bedrooms', 'bathrooms']
C:\Users\timla\Anaconda3\lib\site-packages\ipykernel_launcher.py:38: FutureWarning: 'argmax' is deprecated, use 'idxmax' instead. The behavior of 'argmax'
will be corrected to return the positional maximum in the future.
Use 'series.values.argmax' to get the position of the maximum now.
Let's compare with Scikit-Learn's feature ranking.

# Step 14: Feature ranking with recursive feature elminination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
selector = RFE( linreg, n_features_to_select=2 )
selector = selector.fit( features, df_new.price )

print(features.columns)
selector.support_
Index(['sqft_living', 'grade', 'bedrooms', 'bathrooms'], dtype='object')
array([False,  True, False,  True])
```

With n_features_to_select set to one, it chose GRADE. With two it added BATHROOMS.

Now run the larger dataset now with just the really obvious things removed such as lat, long, zip, etc. to see if we overlooked any other relevant features.

```
linreg = LinearRegression()
selector = RFE( linreg, n_features_to_select=2 )

# Using DF from before creating DF_NEW
features = df.drop(columns='price')

selector = selector.fit( features, df.price )

print(features.columns)
print(selector.support_)
print(selector.ranking_)
Index(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'condition', 'grade', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated', 'zipcode'],
      dtype='object')
[False False False False False  True False  True False False False False
 False]
[ 6  3  7 12  4  1  2  1  9  8  5 11 10]
```

Interesting. It chose WATERFRONT and GRADE this time. That's not completely counter-intuitive. This shows CONDITION as the second ranking, and BATHROOMS again ranked near the top.

Another observation is that this data has not been normalized and the result seems mostly consistent with the normalized data.

Next we will drop the zip, waterfront, floors, and yr columns and normalize the remaining columns and rerun the above tests starting at "Step 10: Perform multi-featured OLS." From that we can decide what dataset to continue on to the remaining tests.

Perform a train-test split and see how well we can predict results.

Finally, we'll do some other rankings to see trends and whatever other interesting facts from our data.

```
# Step 20: Drop and normalize and rerun from Step 10

# df is the original minus the lat, long, etc. fields and our starting point.

df_20 = df.drop(columns=['floors', 'waterfront', 'yr_built', 'yr_renovated', 'zipcode'])

# Normalize everything except price - no log scaling.
def normalize(x):
    """
        return mean-normalized x
    """
    return (x - x.mean()) / (x.max() - x.min())
# end

df_20.sqft_living = normalize(df_20.sqft_living)
df_20.sqft_lot = normalize(df_20.sqft_lot)
df_20.sqft_above = normalize(df_20.sqft_above)
df_20.sqft_basement = normalize(df_20.sqft_basement)
df_20.bedrooms = normalize(df_20.bedrooms)
df_20.bathrooms = normalize(df_20.bathrooms)
df_20.grade = normalize(df_20.grade)
df_20.condition = normalize(df_20.condition)
print(df_20.sqft_living.mean())
print(df_20.head())
print(df_20.describe())

# Step 20A: Rerun starting with Step 10

features = df_20.drop(columns='price')
formula = "price" + "~" + "+".join(features.columns)
model = ols( formula=formula, data=df_20 ).fit()
model.summary()

1.6842786555048013e-17

      price  bedrooms  bathrooms  sqft_living  sqft_lot  condition     grade  \
0  221900.0 -0.037183  -0.148779    -0.068363 -0.005724  -0.102438 -0.065795   
1  538000.0 -0.037183   0.017888     0.037180 -0.004760  -0.102438 -0.065795   
2  180000.0 -0.137183  -0.148779    -0.099495 -0.003089  -0.102438 -0.165795   
3  604000.0  0.062817   0.117888    -0.009138 -0.006118   0.397562 -0.065795   
4  510000.0 -0.037183  -0.015446    -0.030398 -0.004252  -0.102438  0.034205   

   sqft_above  sqft_basement  
0   -0.067326      -0.059274  
1    0.042187       0.023713  
2   -0.112680      -0.059274  
3   -0.081707       0.129522  
4   -0.012017      -0.059274  
              price      bedrooms     bathrooms   sqft_living      sqft_lot  \
count  2.159600e+04  2.159600e+04  2.159600e+04  2.159600e+04  2.159600e+04   
mean   5.402920e+05  8.520998e-18 -1.415835e-16  1.684279e-17 -1.795771e-18   
std    3.673760e+05  9.041137e-02  1.025331e-01  6.971314e-02  2.508636e-02   
min    7.800000e+04 -2.371828e-01 -2.154458e-01 -1.298666e-01 -8.831770e-03   
25%    3.220000e+05 -3.718281e-02 -4.877909e-02 -4.938065e-02 -6.093768e-03   
50%    4.500000e+05 -3.718281e-02  1.788757e-02 -1.293418e-02 -4.531532e-03   
75%    6.450000e+05  6.281719e-02  5.122091e-02  3.566111e-02 -2.673992e-03   
max    7.700000e+06  7.628172e-01  7.845542e-01  8.701334e-01  9.911682e-01   

          condition         grade    sqft_above  sqft_basement  
count  2.159600e+04  2.159600e+04  2.159600e+04   2.159600e+04  
mean  -6.312170e-16  1.095714e-15  1.537892e-17   1.026247e-18  
std    1.626177e-01  1.173218e-01  9.156673e-02   9.125009e-02  
min   -6.024380e-01 -4.657946e-01 -1.569283e-01  -5.927447e-02  
25%   -1.024380e-01 -6.579459e-02 -6.622030e-02  -5.927447e-02  
50%   -1.024380e-01 -6.579459e-02 -2.529110e-02  -5.927447e-02  
75%    1.475620e-01  3.420541e-02  4.661156e-02   5.483341e-02  
max    3.975620e-01  5.342054e-01  8.430717e-01   9.407255e-01  
OLS Regression Results
Dep. Variable:	price	R-squared:	0.562
Model:	OLS	Adj. R-squared:	0.562
Method:	Least Squares	F-statistic:	3466.
Date:	Mon, 21 Jan 2019	Prob (F-statistic):	0.00
Time:	13:54:24	Log-Likelihood:	-2.9846e+05
No. Observations:	21596	AIC:	5.969e+05
Df Residuals:	21587	BIC:	5.970e+05
Df Model:	8		
Covariance Type:	nonrobust		
coef	std err	t	P>|t|	[0.025	0.975]
Intercept	5.403e+05	1654.246	326.609	0.000	5.37e+05	5.44e+05
bedrooms	-4.956e+05	2.36e+04	-20.976	0.000	-5.42e+05	-4.49e+05
bathrooms	-1.328e+05	2.59e+04	-5.137	0.000	-1.84e+05	-8.22e+04
sqft_living	2.702e+06	2.86e+05	9.461	0.000	2.14e+06	3.26e+06
sqft_lot	-4.77e+05	6.76e+04	-7.061	0.000	-6.09e+05	-3.45e+05
condition	2.348e+05	1.06e+04	22.204	0.000	2.14e+05	2.56e+05
grade	1.101e+06	2.37e+04	46.453	0.000	1.05e+06	1.15e+06
sqft_above	8296.6822	1.95e+05	0.042	0.966	-3.75e+05	3.91e+05
sqft_basement	2.957e+05	1.04e+05	2.837	0.005	9.14e+04	5e+05
Omnibus:	16646.452	Durbin-Watson:	1.986
Prob(Omnibus):	0.000	Jarque-Bera (JB):	969580.347
Skew:	3.211	Prob(JB):	0.00
Kurtosis:	35.191	Cond. No.	217.


Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
This is getting better. R-squared is higher by about 10%. Skew and Kurtosis are much lower. sqft_above exceeds the 0.05 p-value but all others are close to or zero.
```


Next we'll do feature selection two ways.

```
# Step 22 - Feature selection.

print("Feature selection (from 12):\n")

selected = stepwise_selection( X=features, y=df_20.price )
print(selected)


print("\n\nFeature ranking with recursive feature elminination (from 14):\n")

linreg = LinearRegression()
selector = RFE( linreg, n_features_to_select=2 )
selector = selector.fit( features, df_20.price )

print(features.columns)
print(selector.support_)
print(selector.ranking_)
Feature selection (from 12):

Add  sqft_above                     with p-value 0.0
Add  sqft_living                    with p-value 0.0
Add  grade                          with p-value 0.0
Add  condition                      with p-value 2.57373e-101
Add  bedrooms                       with p-value 1.22374e-104
Add  sqft_lot                       with p-value 8.49996e-12
Add  bathrooms                      with p-value 2.53963e-07
Add  sqft_basement                  with p-value 0.00455771
Drop sqft_above                     with p-value 0.966144
['sqft_living', 'grade', 'condition', 'bedrooms', 'sqft_lot', 'bathrooms', 'sqft_basement']


Feature ranking with recursive feature elminination (from 14):

Index(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'condition',
       'grade', 'sqft_above', 'sqft_basement'],
      dtype='object')
[False False  True False False  True False False]
[3 6 1 2 5 1 7 4]
```

Well this is entirely consistent with our results from previous tests with the other dataset. (Same data, just different treatment.) We have sqft_living and grade again leading the results. Note on the FEATURE SELECTION test it rejected sqft_above just as we would have predicted from our OLS results above.

And the worst ranking on the RFE was also sqft_above. This time bedrooms was ranked higher than bathrooms however. And the lot square foot had the second highest ranking, whereas it was ranked near last in the previous tests.

Now we'll continue with the train test splits and continue using this set of data.

```
# Step 30: Test-train splits using df_20 dataframe.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( features, df_20.price, test_size=0.2 )
print( len(X_train), len(X_test), len(y_train), len(y_test) )
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test =np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)

from sklearn.metrics import mean_squared_error

print( "Train MSE: ", mean_squared_error(y_train, y_hat_train) )
print( "Test MSE:  ", mean_squared_error(y_test, y_hat_test) )

pct_diff = abs( (mse_train - mse_test) / mse_train )
print( "Pct Diff:  ", pct_diff )
17276 4320 17276 4320
Train Mean Squarred Error: 59922978655.69318
Test Mean Squarred Error: 55717202766.23915
Train MSE:  59922978655.69318
Test MSE:   55717202766.23915
Pct Diff:   0.07018636229049408
```

This seems like a very odd number -- the size of them -- however they are very close together - only a 4.7% difference between the two errors. So based on this the model is doing a good job of predicting home sales prices.

Next we'll use Scikit Learn's cross_val_score to obtain the errors.

```
# Step 32: Cross Validation 

from sklearn.model_selection import cross_val_score

linreg = LinearRegression()
cv_5 = cross_val_score( linreg, features, df_20.price, cv=5 )
print(cv_5)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_5.mean(), cv_5.std() * 2))
[0.54544402 0.5577057  0.54565884 0.57080104 0.58187994]
Accuracy: 0.56 (+/- 0.03)
```

Well this agrees nicely with our results from above. Our model predicts about 56% of the price of the house.

Now we'll run the above with one feature at a time to see what kind of differences we get. That way we can be sure that some are not dragging the model down.

```
# Step 34: Make above a function and call for each feature:

def cvs( feature, target, cv ):
    """
        Perform cross value score
            feature = list of features to use
            target = target values
            cv = number of buckets
        returns the cv scores
    """
    linreg = LinearRegression()
    cv_5 = cross_val_score( linreg, feature, target, cv=cv )
    return cv_5
    # print(feature)
    # print(cv_5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (cv_5.mean(), cv_5.std() * 2))

i = 1
for col in features:
    cv = cvs( np.array(features[col]).reshape(-1,1), df_20.price, 5 )
    print( i, col, ":  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    i += 1

print(selector.support_)
print(selector.ranking_)
1 bedrooms :  0.10 (+/- 0.01)
2 bathrooms :  0.27 (+/- 0.01)
3 sqft_living :  0.49 (+/- 0.01)
4 sqft_lot :  0.01 (+/- 0.01)
5 condition :  -0.00 (+/- 0.01)
6 grade :  0.45 (+/- 0.04)
7 sqft_above :  0.36 (+/- 0.01)
8 sqft_basement :  0.10 (+/- 0.01)
[False False  True False False  True False False]
[3 6 1 2 5 1 7 4]
```

This seems consistent with some things and not with others. We have reprinted the feature selection from Step 22 here for comparison.

Here the two most significant are grade and living, which was indicated before and was pretty consistent throughout.
But lot is very low score here, but was graded number 2 in feature selection (Step 22.)
Let's run grade and living together.

```
cv = cvs( features[['grade','sqft_living']], df_20.price, 5 )
print( "Grade,Living:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )

cv = cvs( features[['grade','sqft_living','bathrooms']], df_20.price, 5 )
print( "Grade,Living,Bathrooms:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )

cv = cvs( features[['grade','sqft_living','sqft_lot']], df_20.price, 5 )
print( "Grade,Living,Lot:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )
Grade,Living:  0.53 (+/- 0.02)
Grade,Living,Bathrooms:  0.54 (+/- 0.02)
Grade,Living,Lot:  0.53 (+/- 0.02)
```

So from this, it looks like each of the features improves the model, however almost all of the value is in grade and living. So, the only problem with this is that grade is a human-generated value. Using purely objective measures would mean that we should see how well we can predict without Grade and Condition. We'll return the above without those two and compare our results.

```
# Step 40: Repeat without grade and condition. NOTE DF_20 REPLACED BY DF_40.

df_40 = df_20.drop(columns=['grade','condition'])
print(df_40.head())

# Repeat from step 20A:
features = df_40.drop(columns='price')
formula = "price" + "~" + "+".join(features.columns)
model = ols( formula=formula, data=df_40 ).fit()
print(model.summary())

# FROM Step 22:
print("Feature selection (from 12):\n")

selected = stepwise_selection( X=features, y=df_40.price )
print(selected)


print("\n\nFeature ranking with recursive feature elminination (from 14):\n")

linreg = LinearRegression()
selector = RFE( linreg, n_features_to_select=2 )
selector = selector.fit( features, df_40.price )

print(features.columns)
print(selector.support_)
print(selector.ranking_)

# FROM Step 30: Test-train splits using df_40 dataframe.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( features, df_40.price, test_size=0.2 )
print( len(X_train), len(X_test), len(y_train), len(y_test) )
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test =np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)

from sklearn.metrics import mean_squared_error

print( "Train MSE: ", mean_squared_error(y_train, y_hat_train) )
print( "Test MSE:  ", mean_squared_error(y_test, y_hat_test) )

pct_diff = abs( (mse_train - mse_test) / mse_train )
print( "Pct Diff:  ", pct_diff )

# FROM Step 32: Cross Validation 

from sklearn.model_selection import cross_val_score

linreg = LinearRegression()
cv_5 = cross_val_score( linreg, features, df_40.price, cv=5 )
print(cv_5)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_5.mean(), cv_5.std() * 2))


# FROM Step 34: Use above function and call for each feature:

i = 1
for col in features:
    cv = cvs( np.array(features[col]).reshape(-1,1), df_40.price, 5 )
    print( i, col, ":  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    i += 1

print(selector.support_)
print(selector.ranking_)

cv = cvs( features[['sqft_living']], df_40.price, 5 )
print( "Living:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )

cv = cvs( features[['sqft_living','bathrooms']], df_40.price, 5 )
print( "Living,Bathrooms:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )

cv = cvs( features[['sqft_living','sqft_lot']], df_40.price, 5 )
print( "Living,Bathrooms,Lot:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )
      price  bedrooms  bathrooms  sqft_living  sqft_lot  sqft_above  \
0  221900.0 -0.037183  -0.148779    -0.068363 -0.005724   -0.067326   
1  538000.0 -0.037183   0.017888     0.037180 -0.004760    0.042187   
2  180000.0 -0.137183  -0.148779    -0.099495 -0.003089   -0.112680   
3  604000.0  0.062817   0.117888    -0.009138 -0.006118   -0.081707   
4  510000.0 -0.037183  -0.015446    -0.030398 -0.004252   -0.012017   

   sqft_basement  
0      -0.059274  
1       0.023713  
2      -0.059274  
3       0.129522  
4      -0.059274  
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.511
Model:                            OLS   Adj. R-squared:                  0.511
Method:                 Least Squares   F-statistic:                     3762.
Date:                Mon, 21 Jan 2019   Prob (F-statistic):               0.00
Time:                        13:54:26   Log-Likelihood:            -2.9965e+05
No. Observations:               21596   AIC:                         5.993e+05
Df Residuals:                   21589   BIC:                         5.994e+05
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept      5.403e+05   1748.165    309.062      0.000    5.37e+05    5.44e+05
bedrooms      -6.666e+05   2.44e+04    -27.268      0.000   -7.15e+05   -6.19e+05
bathrooms      6.811e+04   2.65e+04      2.571      0.010    1.62e+04     1.2e+05
sqft_living    3.747e+06   3.01e+05     12.452      0.000    3.16e+06    4.34e+06
sqft_lot      -6.123e+05   7.13e+04     -8.588      0.000   -7.52e+05   -4.73e+05
sqft_above     2.445e+05   2.06e+05      1.187      0.235   -1.59e+05    6.48e+05
sqft_basement  2.598e+05    1.1e+05      2.359      0.018     4.4e+04    4.76e+05
==============================================================================
Omnibus:                    14149.844   Durbin-Watson:                   1.984
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           464598.144
Skew:                           2.671   Prob(JB):                         0.00
Kurtosis:                      25.086   Cond. No.                         217.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Feature selection (from 12):

Add  bathrooms                      with p-value 0.0
Add  sqft_above                     with p-value 0.0
Add  sqft_living                    with p-value 0.0
Drop bathrooms                      with p-value 0.262705
Add  bedrooms                       with p-value 1.38729e-152
Add  sqft_lot                       with p-value 2.10405e-18

['sqft_above', 'sqft_living', 'bedrooms', 'sqft_lot']


Feature ranking with recursive feature elminination (from 14):

Index(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above',
       'sqft_basement'],
      dtype='object')
[ True False  True False False False]
[1 5 1 2 4 3]
17276 4320 17276 4320
Train Mean Squarred Error: 65319739754.27574
Test Mean Squarred Error: 68724083373.44386
Train MSE:  65319739754.27574
Test MSE:   68724083373.44386
Pct Diff:   0.0521181442543221
[0.50978584 0.51226304 0.4946307  0.51014239 0.51548337]
Accuracy: 0.51 (+/- 0.01)
1 bedrooms :  0.10 (+/- 0.01)
2 bathrooms :  0.27 (+/- 0.01)
3 sqft_living :  0.49 (+/- 0.01)
4 sqft_lot :  0.01 (+/- 0.01)
5 sqft_above :  0.36 (+/- 0.01)
6 sqft_basement :  0.10 (+/- 0.01)
[ True False  True False False False]
[1 5 1 2 4 3]
Living:  0.49 (+/- 0.01)
Living,Bathrooms:  0.49 (+/- 0.01)
Living,Bathrooms,Lot:  0.49 (+/- 0.01)
```

Conclusion:

We still get a fairly good predictive value from sqft_living of about half of the value. None of the other features provide much help. sqft_above is mostly colinear with sqft_living so doesn't really mean much. Bedrooms and bathrooms are not much help and bathrooms seems to move around a lot.

The tran v. test pct difference is about two percentage points higher on these tests than prior.

Bottom line: Humans seem to make help when it comes to assigning value to homes.

Afterthought: Let's remove the outliers in bedrooms and say only include those with six or fewer bedrooms and see if that improves our results. After all, if we are trying to predict home values we are not going to expect many examples with more than six bedrooms.

```
# Step 50: Repeat without bedrooms > 6. NOTE df_40 REPLACED BY DF_50 after dropping rows.

# NOTE: df_50 and later are normalized so we have to go back to df 
# and redrop the columns and normalize after removing rows of 
# six or more bedrooms.

# df is the original minus the lat, long, etc. fields and our starting point.
df_50 = df.drop(columns=['grade','condition','floors', 'waterfront', 'yr_built', 'yr_renovated', 'zipcode'])
df_50 = df_50[df_50.bedrooms < 6]

df_50.sqft_living = normalize(df_50.sqft_living)
df_50.sqft_lot = normalize(df_50.sqft_lot)
df_50.sqft_above = normalize(df_50.sqft_above)
df_50.sqft_basement = normalize(df_50.sqft_basement)
df_50.bedrooms = normalize(df_50.bedrooms)
df_50.bathrooms = normalize(df_50.bathrooms)
print(df_50.sqft_living.mean())
print(df_50.head())
print(df_50.describe())


print(df_50.describe())

# Repeat from step 20A:
features = df_50.drop(columns='price')
formula = "price" + "~" + "+".join(features.columns)
model = ols( formula=formula, data=df_50 ).fit()
print(model.summary())

# FROM Step 22:
print("Feature selection (from 12):\n")

selected = stepwise_selection( X=features, y=df_50.price )
print(selected)


print("\n\nFeature ranking with recursive feature elminination (from 14):\n")

linreg = LinearRegression()
selector = RFE( linreg, n_features_to_select=2 )
selector = selector.fit( features, df_50.price )

print(features.columns)
print(selector.support_)
print(selector.ranking_)

# FROM Step 30: Test-train splits using df_50 dataframe.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( features, df_50.price, test_size=0.2 )
print( len(X_train), len(X_test), len(y_train), len(y_test) )
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test =np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)

from sklearn.metrics import mean_squared_error

print( "Train MSE: ", mean_squared_error(y_train, y_hat_train) )
print( "Test MSE:  ", mean_squared_error(y_test, y_hat_test) )

pct_diff = abs( (mse_train - mse_test) / mse_train )
print( "Pct Diff:  ", pct_diff )

# FROM Step 32: Cross Validation 

from sklearn.model_selection import cross_val_score

linreg = LinearRegression()
cv_5 = cross_val_score( linreg, features, df_50.price, cv=5 )
print(cv_5)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_5.mean(), cv_5.std() * 2))


# FROM Step 34: Use above function and call for each feature:

i = 1
for col in features:
    cv = cvs( np.array(features[col]).reshape(-1,1), df_50.price, 5 )
    print( i, col, ":  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    i += 1

print(selector.support_)
print(selector.ranking_)

cv = cvs( features[['sqft_living']], df_50.price, 5 )
print( "Living:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )

cv = cvs( features[['sqft_living','bathrooms']], df_50.price, 5 )
print( "Living,Bathrooms:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )

cv = cvs( features[['sqft_living','sqft_lot']], df_50.price, 5 )
print( "Living,Bathrooms,Lot:  %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2) )
-5.534667761382053e-18
      price  bedrooms  bathrooms  sqft_living  sqft_lot  sqft_above  \
0  221900.0 -0.081503  -0.175866    -0.090978 -0.005696   -0.077931   
1  538000.0 -0.081503   0.024134     0.052766 -0.004731    0.051481   
2  180000.0 -0.331503  -0.175866    -0.133377 -0.003061   -0.131526   
3  604000.0  0.168497   0.144134    -0.010316 -0.006089   -0.094924   
4  510000.0 -0.081503  -0.015866    -0.039271 -0.004224   -0.012572   

   sqft_basement  
0      -0.057652  
1       0.025335  
2      -0.057652  
3       0.131145  
4      -0.057652  
              price      bedrooms     bathrooms   sqft_living      sqft_lot  \
count  2.126300e+04  2.126300e+04  2.126300e+04  2.126300e+04  2.126300e+04   
mean   5.354206e+05 -9.110481e-16 -8.026313e-17 -5.534668e-18 -2.195348e-18   
std    3.547799e+05  2.069517e-01  1.199676e-01  9.236963e-02  2.511525e-02   
min    7.800000e+04 -5.815031e-01 -2.558661e-01 -1.747419e-01 -8.803270e-03   
25%    3.200000e+05 -8.150308e-02 -9.586606e-02 -6.615869e-02 -6.065268e-03   
50%    4.500000e+05 -8.150308e-02  2.413394e-02 -1.652063e-02 -4.514542e-03   
75%    6.399665e+05  1.684969e-01  6.413394e-02  4.759519e-02 -2.683352e-03   
max    7.060000e+06  4.184969e-01  7.441339e-01  8.252581e-01  9.911967e-01   

         sqft_above  sqft_basement  
count  2.126300e+04   2.126300e+04  
mean   9.770451e-17  -2.774716e-16  
std    1.065104e-01   8.915708e-02  
min   -1.838134e-01  -5.765214e-02  
25%   -7.662384e-02  -5.765214e-02  
50%   -2.956502e-02  -5.765214e-02  
75%    5.409512e-02   5.230637e-02  
max    8.161866e-01   9.423479e-01  
              price      bedrooms     bathrooms   sqft_living      sqft_lot  \
count  2.126300e+04  2.126300e+04  2.126300e+04  2.126300e+04  2.126300e+04   
mean   5.354206e+05 -9.110481e-16 -8.026313e-17 -5.534668e-18 -2.195348e-18   
std    3.547799e+05  2.069517e-01  1.199676e-01  9.236963e-02  2.511525e-02   
min    7.800000e+04 -5.815031e-01 -2.558661e-01 -1.747419e-01 -8.803270e-03   
25%    3.200000e+05 -8.150308e-02 -9.586606e-02 -6.615869e-02 -6.065268e-03   
50%    4.500000e+05 -8.150308e-02  2.413394e-02 -1.652063e-02 -4.514542e-03   
75%    6.399665e+05  1.684969e-01  6.413394e-02  4.759519e-02 -2.683352e-03   
max    7.060000e+06  4.184969e-01  7.441339e-01  8.252581e-01  9.911967e-01   

         sqft_above  sqft_basement  
count  2.126300e+04   2.126300e+04  
mean   9.770451e-17  -2.774716e-16  
std    1.065104e-01   8.915708e-02  
min   -1.838134e-01  -5.765214e-02  
25%   -7.662384e-02  -5.765214e-02  
50%   -2.956502e-02  -5.765214e-02  
75%    5.409512e-02   5.230637e-02  
max    8.161866e-01   9.423479e-01  
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.505
Model:                            OLS   Adj. R-squared:                  0.505
Method:                 Least Squares   F-statistic:                     3621.
Date:                Mon, 21 Jan 2019   Prob (F-statistic):               0.00
Time:                        14:07:56   Log-Likelihood:            -2.9441e+05
No. Observations:               21263   AIC:                         5.888e+05
Df Residuals:                   21256   BIC:                         5.889e+05
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept      5.354e+05   1711.218    312.889      0.000    5.32e+05    5.39e+05
bedrooms      -2.784e+05   1.04e+04    -26.776      0.000   -2.99e+05   -2.58e+05
bathrooms      6.183e+04   2.19e+04      2.826      0.005    1.89e+04    1.05e+05
sqft_living    2.825e+06   2.25e+05     12.580      0.000    2.39e+06    3.27e+06
sqft_lot      -5.349e+05   6.97e+04     -7.679      0.000   -6.71e+05   -3.98e+05
sqft_above     1.067e+05   1.77e+05      0.602      0.547   -2.41e+05    4.54e+05
sqft_basement  2.055e+05   1.12e+05      1.835      0.067    -1.4e+04    4.25e+05
==============================================================================
Omnibus:                    12743.901   Durbin-Watson:                   1.980
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           294744.237
Skew:                           2.457   Prob(JB):                         0.00
Kurtosis:                      20.565   Cond. No.                         179.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Feature selection (from 12):

Add  bathrooms                      with p-value 0.0
Add  sqft_above                     with p-value 0.0
Add  sqft_living                    with p-value 0.0
Drop bathrooms                      with p-value 0.638803
Add  bedrooms                       with p-value 2.96501e-147
C:\Users\timla\Anaconda3\lib\site-packages\ipykernel_launcher.py:38: FutureWarning: 'argmax' is deprecated, use 'idxmax' instead. The behavior of 'argmax'
will be corrected to return the positional maximum in the future.
Use 'series.values.argmax' to get the position of the maximum now.
Add  sqft_lot                       with p-value 4.07092e-15
Add  bathrooms                      with p-value 0.00489856
['sqft_above', 'sqft_living', 'bedrooms', 'sqft_lot', 'bathrooms']


Feature ranking with recursive feature elminination (from 14):

Index(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above',
       'sqft_basement'],
      dtype='object')
[False False  True  True False False]
[2 5 1 1 4 3]
17010 4253 17010 4253
Train Mean Squarred Error: 62124964203.81789
Test Mean Squarred Error: 62770426237.22359
Train MSE:  62124964203.81789
Test MSE:   62770426237.22359
Pct Diff:   0.010389736906537064
[0.51025027 0.49159679 0.4865039  0.5068647  0.51462981]
Accuracy: 0.50 (+/- 0.02)
1 bedrooms :  0.09 (+/- 0.01)
2 bathrooms :  0.27 (+/- 0.02)
3 sqft_living :  0.48 (+/- 0.02)
4 sqft_lot :  0.01 (+/- 0.01)
5 sqft_above :  0.36 (+/- 0.02)
6 sqft_basement :  0.09 (+/- 0.01)
[False False  True  True False False]
[2 5 1 1 4 3]
Living:  0.48 (+/- 0.02)
Living,Bathrooms:  0.48 (+/- 0.02)
Living,Bathrooms,Lot:  0.49 (+/- 0.02)
```

This didn't really make much difference so the extra bedrooms really weren't causing trouble, probably because there were so few relative to the sample size.


