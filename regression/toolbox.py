import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn import svm, preprocessing

from pyelm.elm import ELMRegressor
from readers.mat_reader import MatReader


class NothingScaler:
    """
    Scaler that does not transform the data.
    """

    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return x


class RegressionToolbox:

    def __init__(self):
        pass

    # Different regression models with respective constructor and scaler.
    MODELS = {'SVM': {'constructor': svm.SVR, 'scaler': preprocessing.MinMaxScaler()},
              'ELM': {'constructor': ELMRegressor, 'scaler': preprocessing.MinMaxScaler()},
              'LR': {'constructor': LinearRegression, 'scaler': preprocessing.MinMaxScaler()},
              'KNN': {'constructor': KNeighborsRegressor, 'scaler': preprocessing.StandardScaler()},
              'BNN': {'constructor': BaggingRegressor, 'scaler': preprocessing.StandardScaler()},
              'RANSAC': {'constructor': RANSACRegressor, 'scaler': NothingScaler()}}

    @staticmethod
    def get_instance(algorithm, training_data_folder, testing_data_folder):
        """
        Returns an instance of the given trained model of the given algorithm with the training and
        testing/validation data correctly scaled.
        """

        if algorithm not in RegressionToolbox.MODELS:
            raise Exception('Invalid algorithm')
        mat_reader = MatReader()
        training_ts, training_xs, training_ys = mat_reader.read(training_data_folder)
        algorithm_functions = RegressionToolbox.MODELS[algorithm]
        regression_model = algorithm_functions['constructor']()
        algorithm_functions['scaler'].fit(training_xs)
        regression_model.fit(algorithm_functions['scaler'].transform(training_xs), training_ys)
        testing_ts, testing_xs, testing_ys = mat_reader.read(testing_data_folder)
        return regression_model, testing_ts, algorithm_functions['scaler'].transform(testing_xs), testing_ys

    @staticmethod
    def compare_regression_algorithms(ps=10):
        """
        Performs cross-validation between the different regression algorithms.
        """

        mat_reader = MatReader()
        dummy, xs, ys = mat_reader.read('ColdComplaintData/Training')
        results = []
        scoring = 'neg_mean_squared_error'
        for name, algorithm_functions in RegressionToolbox.MODELS.iteritems():
            model = algorithm_functions['constructor']()
            algorithm_functions['scaler'].fit(xs)
            cv_results = cross_val_score(model, algorithm_functions['scaler'].transform(xs), ys, cv=ps, scoring=scoring)
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
