#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from readers.mat_reader import MatReader
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import datasets


def main(mat_folder_path):
  mat_reader = MatReader(mat_folder_path)
  xs, ys = mat_reader.read()
  cmp_regression_algs(xs, ys)

def cmp_regression_algs(xs, ys):
  num_folds = 10
# prepare models
  models = []
  models.append(('LR', LogisticRegression()))
  models.append(('LDA', LinearDiscriminantAnalysis()))
  models.append(('KNN', KNeighborsClassifier()))
  models.append(('CART', DecisionTreeClassifier()))
  models.append(('NB', GaussianNB()))
  models.append(('SVM', SVC()))
# evaluate each model in turn
  results = []
  names = []
  scoring = 'accuracy'
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

