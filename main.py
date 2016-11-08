#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from readers.mat_reader import MatReader
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import pandas as pd
import pyflux as pf


from sklearn import svm

def main(mat_folder_path):
    mat_reader = MatReader(mat_folder_path)
    ts, xs, ys = mat_reader.read()
    cmp_regression_algs(xs, ys)


def predict_residuals(xs, ys,ts):
    # Assume hardcoded model for now
    clf = svm.SVR()
    clf.fit(xs[:3000], ys[:3000])
    mat_reader = MatReader('ColdComplaintData/Testing')
    ttest, xtest, ytest = mat_reader.read()
    plt.figure()
    plt.title('Actual vs. Prediction')
    plt.plot(ytest)
    pred = []
    pred_res = []
    for x in xrange(len(xtest)):
        pred.append(clf.predict(xtest[x])[0])
        pred_res.append(pred[x] - ytest[x])
    plt.plot(pred)
    plt.plot(pred_res)
    plt.show()

    # Predicting future residuals
    df = pd.DataFrame(pred_res, index=ttest, columns=['residuals'])
    df.plot(figsize=(16,12))
    model = pf.ARIMA(data=df,ar=4,ma=4,integ=0,target='residuals')
    mod = model.fit("MLE")
    mod.summary()
    model.plot_predict(h=3,past_values=5,figsize=(15,5))


def cmp_regression_algs(xs, ys):
    num_folds = 10
    # prepare models
    models = [('SVM', svm.SVR()), ('KNN', KNeighborsRegressor()), ('LR', LinearRegression()),
              ('DT', DecisionTreeRegressor), ('BNN', BaggingRegressor()), ('RANSAC', RANSACRegressor())]
    # evaluate each model in tusvm.rn
    results = []
    names = []
    scoring = 'mean_squared_error'
    for name, model in models:
        cv_results = cross_val_score(model, xs, ys, cv=num_folds, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

# Simple example on how to read the data.
if __name__ == "__main__":
    main('ColdComplaintData/Training')

