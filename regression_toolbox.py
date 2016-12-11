import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn import svm, preprocessing

from pyelm.elm import ELMRegressor
from readers.mat_reader import MatReader


def nothing(x):
    return x


class RegressionToolbox:

    def __init__(self):
        pass

    MODELS = {'SVM': {'constructor': svm.SVR, 'preprocessing': preprocessing.minmax_scale},
              'ELM': {'constructor': ELMRegressor, 'preprocessing': preprocessing.minmax_scale},
              'LR': {'constructor': LinearRegression, 'preprocessing': preprocessing.minmax_scale},
              'KNN': {'constructor': KNeighborsRegressor, 'preprocessing': preprocessing.scale},
              'BNN': {'constructor': BaggingRegressor, 'preprocessing': preprocessing.scale},
              'RANSAC': {'constructor': RANSACRegressor, 'preprocessing': nothing}}

    @staticmethod
    def get_instance(algorithm, training_data_folder, testing_data_folder):
        mat_reader = MatReader()
        training_ts, training_xs, training_ys = mat_reader.read(training_data_folder)
        algorithm_functions = RegressionToolbox.MODELS[algorithm] if algorithm in RegressionToolbox.MODELS \
            else RegressionToolbox.MODELS['SVM']
        regression_model = algorithm_functions['constructor']()
        regression_model.fit(training_xs, training_ys)
        testing_ts, testing_xs, testing_ys = mat_reader.read(testing_data_folder)
        return regression_model, testing_ts, algorithm_functions['preprocessing'](testing_xs), testing_ys

    @staticmethod
    def compare_regression_algorithms(ps=10):
        mat_reader = MatReader()
        dummy, xs, ys = mat_reader.read('ColdComplaintData/Training')
        results = []
        scoring = 'neg_mean_squared_error'
        for name, algorithm_functions in RegressionToolbox.MODELS.iteritems():
            model = algorithm_functions['constructor']()
            cv_results = cross_val_score(model, algorithm_functions['preprocessing'](xs), ys, cv=ps, scoring=scoring)
            results.append(cv_results)
            print '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle('Regression algorithms Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(RegressionToolbox.MODELS.keys())
        ax.set_xlabel('Regression Algorithm')
        ax.set_ylabel('Negative mean squared error')
        fig.savefig('comparision.png')
