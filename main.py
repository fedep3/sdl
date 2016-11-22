#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import pyflux as pf
from sklearn.metrics import auc

from readers.mat_reader import MatReader

from sklearn.model_selection import PredefinedSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pyelm.elm import ELMRegressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def main():
    mat_reader = MatReader('ColdComplaintData/Training')
    ts, xs, ys = mat_reader.read()
    
    if __debug__:
        cmp_regression_algs(xs, ys, 5)

    # Build the ideal linear regression model
    clf = LinearRegression()
    clf.fit(xs, ys)

    # Calculate residuals and predict alarms using validation data
    # Using validation or training data for prediction?
    mat_reader = MatReader('ColdComplaintData/Testing')
    ttest, xtest, ytest = mat_reader.read()

    print "Actuals: ", ytest

    pred_future = predict_residuals(clf, ttest, xtest, ytest)
    print "Future residual predictions: ", pred_future

    # TODO change threshold levels
    threshold = calc_threshold(pred_future, ytest)
    cor_count, fa_count, md_count = calc_error_rates(threshold, pred_future, ytest)
    print "Correct Alarm Count: ", cor_count
    print "False Alarm Count: ", fa_count
    print "Missed Detection Count: ", md_count

def predict_residuals(clf, ttest, xtest, ytest):
    # Calculate residuals
    pred = []
    pred_res = []
    for x in xrange(len(xtest)):
        pred.append(clf.predict(xtest[x])[0])
        pred_res.append(pred[x] - ytest[x])

    if __debug__:
        # Plot
        plt.figure()
        plt.title('Actual vs. Prediction')
        plt.plot(ytest)
        plt.plot(pred)
        plt.plot(pred_res)
        plt.show()

    # Predicting future residuals
    pred_future = []
    for x in xrange(len(pred_res)-16):
        df = pd.DataFrame(pred_res[x:x+16], index=ttest[x:x+16], columns=['residuals'])
        #df.plot(figsize=(16,12))
        model = pf.ARIMA(data=df,ar=4,ma=4,integ=0,target='residuals')
        mod = model.fit("MLE")
        #mod.summary()
        #model.plot_predict(h=12,past_values=16,figsize=(15,5))
        pred_future.append(model.predict(h=12)['residuals'][11])
        #print pred_future

    return pred_future


def calc_threshold(pred_future, actuals):
    prod_future_max = max(pred_future)
    prod_future_min = min(pred_future)
    prod_future_delta = prod_future_max - prod_future_min

    # Predict alarms with the given level
    alarms_predictions = []
    for p in pred_future:
        alarms_predictions.append((p - prod_future_min)/prod_future_delta)

    # Calcuate whether or not there was actually an alarm anywhere
    # in the next time period
    alarm_actuals = []
    for i in xrange(len(actuals) - 16):
        alarm_actual = False
        for x in xrange(16):
            if actuals[i + x] < 68.1:
                alarm_actual = True
        alarm_actuals.append(alarm_actual)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(alarm_actuals, alarms_predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    if __debug__:
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    return brentq(lambda x: 1. - x - interp1d(false_positive_rate, true_positive_rate)(x), 0., 1.)


def calc_error_rates(threshold, pred_future, actuals):
    La = threshold
    
    # Predict alarms with the given level
    alarms = []
    for p in pred_future:
        if abs(p) > La:
            alarms.append(True)
        else:
            alarms.append(False)

    # Calcuate whether or not there was actually an alarm anywhere
    # in the next time period
    alarm_actuals = []
    for i in xrange(len(actuals)-16):
        alarm_actual = False
        for x in xrange(16):
            if actuals[i+x] < 68.1:
                alarm_actual = True
        alarm_actuals.append(alarm_actual)

    # Calculate the false alarm and missed detection rate
    fa_count = 0
    md_count = 0
    cor_count = 0
    for r in xrange(len(alarms)):
        if alarms[r] == True and alarm_actuals[r] == False:
            fa_count += 1
        elif alarms[r] == False and alarm_actuals[r] == True:
            md_count += 1
        else:
            cor_count += 1

    return cor_count, fa_count, md_count


def cmp_regression_algs(xs, ys, ps):
    num_folds = 5
    # prepare models
    models = [('SVM', svm.SVR()), ('KNN', KNeighborsRegressor()), ('LR', LinearRegression()),
              ('BNN', BaggingRegressor()), ('RANSAC', RANSACRegressor()), ('ELM', ELMRegressor())]
    # evaluate each model in tusvm.rn
    results = []
    names = []
    scoring = 'neg_mean_squared_error'
    for name, model in models:
        cv_results = cross_val_score(model, xs, ys, cv=ps, scoring=scoring)
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

# only perform cross validation between training and validation sets
def compare_training_validation():
    mat_reader_train = MatReader('ColdComplaintData/Training')
    ts_train, X_train, Y_train = mat_reader_train.read()
    mat_reader_validation = MatReader('ColdComplaintData/Validation')
    ts_validation, X_validation, Y_validation = mat_reader_validation.read()
    xs = X_train + X_validation
    ys = Y_train + Y_validation
    test_fold = len(Y_train)*[-1] + len(Y_validation)*[0]
    ps = PredefinedSplit(test_fold)
    cmp_regression_algs(xs, ys, ps)  

# Simple example on how to read the data.
if __name__ == "__main__":
    main()
















