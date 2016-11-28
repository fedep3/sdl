import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn import svm

from pyelm.elm import ELMRegressor


class RegressionToolbox:

    @staticmethod
    def compare_regression_algorithms(xs, ys, ps):
        # prepare models
        models = [('SVM', svm.SVR()), ('KNN', KNeighborsRegressor()), ('LR', LinearRegression()),
                  ('BNN', BaggingRegressor()), ('RANSAC', RANSACRegressor()), ('ELM', ELMRegressor())]
        # evaluate each model
        results = []
        names = []
        scoring = 'neg_mean_squared_error'
        for name, model in models:
            cv_results = cross_val_score(model, xs, ys, cv=ps, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            print '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Regression algorithms Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        fig.savefig('comparision.png')