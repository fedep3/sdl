#!/usr/bin/python
# -*- coding: utf-8 -*-
from detection_toolbox import DetectionToolbox
from prediction_model import ARIMAFuturePredictionModel
from readers.mat_reader import MatReader

from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression

import warnings

from regression_toolbox import RegressionToolbox

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning) 

# Do not change this
FUTURE_PREDICTION_HORIZON = 12


def main():
    mat_reader = MatReader()
    training_ts, training_xs, training_ys = mat_reader.read('ColdComplaintData/Training')

    if __debug__:
        RegressionToolbox.compare_regression_algorithms(training_xs, training_ys, 5)

    # Build the ideal linear regression model
    regression_model = LinearRegression()
    regression_model.fit(training_xs, training_ys)

    for past_prediction_horizon in [32, 64]:
        print 'Past prediction horizon: ', past_prediction_horizon
        future_prediction_model = ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 4, 4)
        detection_toolbox = DetectionToolbox(regression_model, past_prediction_horizon, future_prediction_model)

        # Calculate residuals and predict alarms using validation data
        # Using validation or training data for prediction?
        testing_ts, testing_xs, testing_ys = mat_reader.read('ColdComplaintData/Testing')
        if __debug__:
            print 'Actuals: ', testing_ys
        future_residuals_prediction = detection_toolbox.predict_residuals(testing_ts, testing_xs, testing_ys)
        if __debug__:
            print 'Future residual predictions: ', future_residuals_prediction

        roc_auc, threshold, fa_rate, md_rate = detection_toolbox.calculate_roc_curve(future_residuals_prediction, testing_ys)
        print 'ROC AUC: ', roc_auc
        print 'Idea threshold found: ', threshold
        print 'FA rate: %0.2f, MD rate: %0.2f' % (fa_rate, md_rate)


# only perform cross validation between training and validation sets
def compare_training_validation():
    mat_reader = MatReader()
    training_ts, training_xs, training_ys = mat_reader.read('ColdComplaintData/Training')
    validation_ts, validation_xs, validation_ys = mat_reader.read('ColdComplaintData/Validation')
    xs = training_xs + validation_xs
    ys = training_ys + validation_ys
    test_fold = len(training_ys)*[-1] + len(validation_ys)*[0]
    ps = PredefinedSplit(test_fold)
    RegressionToolbox.compare_regression_algorithms(xs, ys, ps)

if __name__ == '__main__':
    main()
















