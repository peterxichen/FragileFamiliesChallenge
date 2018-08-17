import pandas as pd
import numpy as np
import scipy.stats as st
import math
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import resample

# read in data
df = pd.read_csv('pFiltered.csv', low_memory=False)
df = df.values
train = pd.read_csv('train.csv')

# return X and y of by matching challengeID
# arguments: 'gpa' 'grit' 'materialHardship' 'eviction' 'layoff' 'jobTraining'
def getData(outcome):
    cid = train['challengeID'].values
    val = train[outcome].values
    X=[]
    y=[]
    for i in range(0, len(val)-1):
        if not math.isnan(val[i]):
            X.append(list(df[cid[i]-1]))
            y.append(val[i])
            # make sure the indices match
            if (cid[i] != df[cid[i]-1][0]):
                prints("Mismatch")
                break
    X=np.array(X)
    X=np.delete(X,0,1) #remove first column
    y=np.array(y)
    return X, y

# runs ordinary least squares regression to predict continuous variables
# returns average r2 for K=10 folds
def ols(X, y):
    lm = linear_model.LinearRegression()
    kf = KFold(n_splits=10, shuffle=True)
    score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = score + metrics.r2_score(y_test, y_pred)
    return score/10


# runs logistic regression to classify binary variables
# returns average accuracy for K=10 folds
def logit(X, y):
    logit = linear_model.LogisticRegression()
    kf = KFold(n_splits=10, shuffle=True)
    score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = logit.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = score + metrics.accuracy_score(y_test, y_pred)
    return score/10

# runs ridge regression
# returns average r^2 for K=10 folds
def ridge(X, y, a):
    lm = linear_model.Ridge(alpha=a)
    kf = KFold(n_splits=10, shuffle=True)
    score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = score + metrics.r2_score(y_test, y_pred)
    return score/10

# runs lasso regression
# returns average r^2 for K=10 folds
def lasso(X, y, a):
    lm = linear_model.Lasso(alpha=a)
    kf = KFold(n_splits=10, shuffle=True)
    score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = score + metrics.r2_score(y_test, y_pred)
    return score/10

# finds optimal alpha for Ridge regression given range
def findRidgeAlpha(X,y,alphas):
    model = linear_model.Ridge()
    search = GridSearchCV(estimator=model, param_grid = dict(alpha=alphas))
    search.fit(X,y)
    return search.best_estimator_.alpha

# finds optimal alpha for Lasso regression given range
def findLassoAlpha(X,y,alphas):
    model = linear_model.Lasso()
    search = GridSearchCV(estimator=model, param_grid = dict(alpha=alphas))
    search.fit(X,y)
    return search.best_estimator_.alpha

# ordinary bootstrap method and returns r2's (k = 1000, n = 10000)
def bootstrapRegression(X,y,lm):
    r2 = []
    for i in range(1000):
        print(i)
        X_train, y_train = resample(X, y, replace=True, n_samples = 10000)
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X)
        r2.append(metrics.r2_score(y, y_pred))
    return np.array(r2)

# ordinary bootstrap method and returns r2's (k = 1000, n = 10000)
def bootstrapLogistic(X,y):
    acc = []
    logit = linear_model.LogisticRegression()
    for i in range(1000):
        X_train, y_train = resample(X, y, replace=True, n_samples = 10000)
        model = logit.fit(X_train, y_train)
        y_pred = model.predict(X)
        acc.append(metrics.accuracy_score(y, y_pred))
    return np.array(acc)
    
#prediction using OLS, ridge, lasso
alphas = [.001,.01,.1,1,10,100,1000,10000,100000]
X,y = getData('gpa')
print('GPA r2 values:')
print('OLS:',ols(X,y))
ar = findRidgeAlpha(X,y,alphas)
print('Ridge (w/ alpha = ',ar,'):',ridge(X,y,ar))
al = findLassoAlpha(X,y,alphas)
print('Lasso (w/ alpha = ',al,'):',lasso(X,y,al))
lm = linear_model.LinearRegression()
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap OLS mean:',r2.mean(),'stdev:',r2.std())
lm = linear_model.Ridge(ar)
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap Ridge (w/ alpha = ',ar,') mean:',r2.mean(),'stdev:',r2.std())
lm = linear_model.Lasso(ar)
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap Lasso (w/ alpha = ',al,') mean:',r2.mean(),'stdev:',r2.std())
print()

X,y = getData('grit')
print('Grit r2 values:')
print('OLS:',ols(X,y))
a = findRidgeAlpha(X,y,alphas)
print('Ridge (w/ alpha = ',a,'):',ridge(X,y,a))
a = findLassoAlpha(X,y,alphas)
print('Lasso (w/ alpha = ',a,'):',lasso(X,y,a))
lm = linear_model.LinearRegression()
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap OLS mean:',r2.mean(),'stdev:',r2.std())
lm = linear_model.Ridge(ar)
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap Ridge (w/ alpha = ',ar,') mean:',r2.mean(),'stdev:',r2.std())
lm = linear_model.Lasso(ar)
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap Lasso (w/ alpha = ',al,') mean:',r2.mean(),'stdev:',r2.std())
print()

X,y = getData('materialHardship')
print('Material Hardship r2 values:')
print('OLS:',ols(X,y))
a = findRidgeAlpha(X,y,alphas)
print('Ridge (w/ alpha = ',a,'):',ridge(X,y,a))
a = findLassoAlpha(X,y,alphas)
print('Lasso (w/ alpha = ',a,'):',lasso(X,y,a))
lm = linear_model.LinearRegression()
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap OLS mean:',r2.mean(),'stdev:',r2.std())
lm = linear_model.Ridge(ar)
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap Ridge (w/ alpha = ',ar,') mean:',r2.mean(),'stdev:',r2.std())
lm = linear_model.Lasso(ar)
r2 = bootstrapRegression(X,y,lm)
print('Bootstrap Lasso (w/ alpha = ',al,') mean:',r2.mean(),'stdev:',r2.std())
print()

#binary logistic classification
X,y = getData('eviction')
print('Eviction accuracy values:')
print('Logistic:',logit(X,y))
acc = bootstrapLogistic(X,y)
print('Bootstrap Logistic mean:',acc.mean(),'stdev:',acc.std())
print()

X,y = getData('layoff')
print('Layoff accuracy values:')
print('Logistic:',logit(X,y))
acc = bootstrapLogistic(X,y)
print('Bootstrap Logistic mean:',acc.mean(),'stdev:',acc.std())
print()

X,y = getData('jobTraining')
print('Job Training accuracy values:')
print('Logistic:',logit(X,y))
acc = bootstrapLogistic(X,y)
print('Bootstrap Logistic mean:',acc.mean(),'stdev:',acc.std())
print()
