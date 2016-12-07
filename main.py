#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
import argparse

from detection_toolbox import DetectionToolbox
from prediction_model import ARIMAFuturePredictionModel, GARCHFuturePredictionModel, GGSMFuturePredictionModel, \
    AggregatingFuturePredictionModel
from readers.mat_reader import MatReader

from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression

from regression_toolbox import RegressionToolbox

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning) 

# Do not change this
FUTURE_PREDICTION_HORIZON = 12


def best_run():
    mat_reader = MatReader()
    training_ts, training_xs, training_ys = mat_reader.read('ColdComplaintData/Training')

    regression_model = LinearRegression()
    regression_model.fit(training_xs, training_ys)

    testing_ts, testing_xs, testing_ys = mat_reader.read('ColdComplaintData/Testing')
    future_prediction_model_results = []

    for future_prediction_model, past_prediction_horizon, threshold in [(GGSMFuturePredictionModel(FUTURE_PREDICTION_HORIZON), 64, 3.625),
                                                                        (ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 1, 0), 48, 3.775),
                                                                        (ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 4, 0), 64, 3.925)]:
        fa_rate, md_rate = detect_with_threshold(regression_model, future_prediction_model, past_prediction_horizon, testing_ts, testing_xs, testing_ys, threshold)
        future_prediction_model_results.append(
            (future_prediction_model, past_prediction_horizon, threshold, fa_rate, md_rate))

    future_prediction_model_results.sort(key=lambda s: s[3] + s[4])
    for future_prediction_model_result in future_prediction_model_results:
        print 'Model=%s, Past Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' % future_prediction_model_result


def compare_detection_algorithms():
    mat_reader = MatReader()
    training_ts, training_xs, training_ys = mat_reader.read('ColdComplaintData/Training')

    regression_model = LinearRegression()
    regression_model.fit(training_xs, training_ys)

    testing_ts, testing_xs, testing_ys = mat_reader.read('ColdComplaintData/Validation')
    future_prediction_model_results = []
    best_model_position = -1
    max_score = -1.0
    count = 0

    for future_prediction_model in [ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 1, 1),
                                    ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 1, 0),
                                    ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 2, 0),
                                    ARIMAFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 4, 0),
                                    GARCHFuturePredictionModel(FUTURE_PREDICTION_HORIZON, 1, 1),
                                    GGSMFuturePredictionModel(FUTURE_PREDICTION_HORIZON),
                                    AggregatingFuturePredictionModel(FUTURE_PREDICTION_HORIZON)]:
        for past_prediction_horizon in [32, 48, 64, 80, 96]:
            if isinstance(future_prediction_model, AggregatingFuturePredictionModel) and past_prediction_horizon < 48:
                continue
            print 'Past prediction horizon: ', past_prediction_horizon
            roc_auc, fa_rate, md_rate, threshold = detect(regression_model, future_prediction_model, past_prediction_horizon,
                                                 testing_ts, testing_xs, testing_ys)
            future_prediction_model_results.append(
                (future_prediction_model, roc_auc, past_prediction_horizon, threshold, fa_rate, md_rate))
            print 'Model=%s, ROC AUC=%s, Past Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' \
                  % future_prediction_model_results[-1]
            print '======================'
            if best_model_position == -1 or roc_auc > max_score:
                best_model_position = count
                max_score = roc_auc
            print 'Best: Model=%s, ROC AUC=%s, Past Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' \
                  % future_prediction_model_results[best_model_position]
            count += 1
            print '======================'

    print 'Sorted results'
    future_prediction_model_results.sort(key=lambda s: s[1], reverse=True)
    for future_prediction_model_result in future_prediction_model_results:
        print 'Model=%s, ROC AUC=%s, Past Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' % future_prediction_model_result


def detect_with_threshold(regression_model, future_prediction_model, past_prediction_horizon, ts, xs, ys, threshold):
    detection_toolbox = DetectionToolbox(regression_model, past_prediction_horizon, future_prediction_model)
    future_residuals_prediction = detection_toolbox.predict_residuals(ts, xs, ys)
    total_count, fa_count, ta_count, md_count = detection_toolbox.calculate_counts(future_residuals_prediction, ys, threshold)
    fa_rate = float(fa_count) / total_count
    md_rate = float(md_count) / total_count
    return fa_rate, md_rate


def detect(regression_model, future_prediction_model, past_prediction_horizon, ts, xs, ys):
    detection_toolbox = DetectionToolbox(regression_model, past_prediction_horizon, future_prediction_model)
    # Calculate residuals and predict alarms using validation data
    # Using validation or training data for prediction?
    if __debug__:
        print 'Actuals: ', ys
    future_residuals_prediction = detection_toolbox.predict_residuals(ts, xs, ys)
    if __debug__:
        print 'Future residual predictions: ', future_residuals_prediction
    roc_auc, threshold, fa_rate, md_rate = detection_toolbox.calculate_roc_curve(future_residuals_prediction, ys)
    print 'ROC AUC: ', roc_auc
    print 'Ideal threshold found: ', threshold
    print 'FA rate: %0.3f, MD rate: %0.3f' % (fa_rate, md_rate)
    return roc_auc, fa_rate, md_rate, threshold


def compare_regression_algorithms():
    mat_reader = MatReader()
    training_ts, training_xs, training_ys = mat_reader.read('ColdComplaintData/Training')
    RegressionToolbox.compare_regression_algorithms(training_xs, training_ys, 10)


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
    parser = argparse.ArgumentParser(description='Predicting Adverse Thermal Events in a Smart Building')
    parser.add_argument("-t", "--type", type=int, choices=[1, 2, 3, 4], default=1,
                        help="type of run: "
                             "(1)Best arguments run, "
                             "(2)Compare regression algorithms, "
                             "(3)Compare regression algorithms with training and validation data, "
                             "(4)Compare detection algorithms")
    args = parser.parse_args()
    if args.type == 1:
        best_run()
    elif args.type == 2:
        compare_regression_algorithms()
    elif args.type == 3:
        compare_training_validation()
    elif args.type == 4:
        compare_detection_algorithms()
