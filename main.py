#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
import argparse

from detection_toolbox import DetectionToolbox
from prediction_model import ARIMAFuturePredictionModel, GARCHFuturePredictionModel, GGSMFuturePredictionModel, \
    AggregatingFuturePredictionModel

from regression_toolbox import RegressionToolbox

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning) 


def best_run():
    regression_model, ts, standardized_xs, ys = RegressionToolbox.get_instance('LR', 'ColdComplaintData/Training', 'ColdComplaintData/Testing')
    future_prediction_model_results = []

    for future_prediction_model, past_prediction_horizon, threshold in [(GGSMFuturePredictionModel(18), 96, 3.275),
                                                                        (ARIMAFuturePredictionModel(18, 4, 0), 48, 3.925),
                                                                        (ARIMAFuturePredictionModel(18, 4, 0), 48, 3.775),
                                                                        (GGSMFuturePredictionModel(12), 96, 3.625),
                                                                        (ARIMAFuturePredictionModel(12, 1, 1), 32, 3.575),
                                                                        (ARIMAFuturePredictionModel(12, 1, 0), 48, 3.775),
                                                                        (GGSMFuturePredictionModel(6), 96, 3.875),
                                                                        (ARIMAFuturePredictionModel(6, 1, 1), 32, 3.825),
                                                                        (ARIMAFuturePredictionModel(6, 1, 0), 48, 3.925)]:
        fa_rate, md_rate = detect_with_threshold(regression_model, future_prediction_model, past_prediction_horizon, ts, standardized_xs, ys, threshold)
        future_prediction_model_results.append(
            (future_prediction_model, past_prediction_horizon, future_prediction_model.future_prediction_horizon, threshold, fa_rate, md_rate))

    future_prediction_model_results.sort(key=lambda s: s[4] + s[5])
    print 'Using LR'
    print '======================'
    for future_prediction_model_result in future_prediction_model_results:
        print 'Model=%s, Past Prediction Horizon=%s, Future Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' % future_prediction_model_result

    regression_model, ts, standardized_xs, ys = RegressionToolbox.get_instance('BNN', 'ColdComplaintData/Training', 'ColdComplaintData/Testing')
    future_prediction_model_results = []

    for future_prediction_model, past_prediction_horizon, threshold in [(ARIMAFuturePredictionModel(18, 1, 0), 64, 3.325),
                                                                        (GGSMFuturePredictionModel(18), 96, 2.875),
                                                                        (ARIMAFuturePredictionModel(18, 4, 0), 48, 3.175),
                                                                        (GGSMFuturePredictionModel(12), 96, 3.275),
                                                                        (ARIMAFuturePredictionModel(12, 4, 0), 48, 3.325),
                                                                        (ARIMAFuturePredictionModel(12, 2, 0), 48, 3.325),
                                                                        (GGSMFuturePredictionModel(6), 96, 3.575),
                                                                        (ARIMAFuturePredictionModel(6, 2, 0), 32, 3.325),
                                                                        (ARIMAFuturePredictionModel(6, 4, 0), 48, 3.375)]:
        fa_rate, md_rate = detect_with_threshold(regression_model, future_prediction_model, past_prediction_horizon, ts, standardized_xs, ys, threshold)
        future_prediction_model_results.append(
            (future_prediction_model, past_prediction_horizon, future_prediction_model.future_prediction_horizon, threshold, fa_rate, md_rate))

    print '\nUsing BNN'
    print '======================'
    future_prediction_model_results.sort(key=lambda s: s[4] + s[5])
    for future_prediction_model_result in future_prediction_model_results:
        print 'Model=%s, Past Prediction Horizon=%s, Future Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' % future_prediction_model_result


def compare_detection_algorithms(algorithm):
    regression_model, ts, standardized_xs, ys = RegressionToolbox.get_instance(algorithm, 'ColdComplaintData/Training', 'ColdComplaintData/Validation')

    future_prediction_model_results = []
    best_model_position = -1
    max_score = -1.0
    count = 0

    for future_prediction_horizon in [6, 12, 18]:
        for future_prediction_model in [ARIMAFuturePredictionModel(future_prediction_horizon, 1, 1),
                                        ARIMAFuturePredictionModel(future_prediction_horizon, 1, 0),
                                        ARIMAFuturePredictionModel(future_prediction_horizon, 2, 0),
                                        ARIMAFuturePredictionModel(future_prediction_horizon, 4, 0),
                                        GARCHFuturePredictionModel(future_prediction_horizon, 1, 1),
                                        GGSMFuturePredictionModel(future_prediction_horizon)]:
            for past_prediction_horizon in [32, 48, 64, 80, 96]:
                    if isinstance(future_prediction_model, AggregatingFuturePredictionModel) and past_prediction_horizon < 48:
                        continue
                    print 'Past prediction horizon: ', past_prediction_horizon
                    roc_auc, fa_rate, md_rate, threshold = detect(regression_model, future_prediction_model, past_prediction_horizon,
                                                         ts, standardized_xs, ys)
                    future_prediction_model_results.append(
                        (future_prediction_model, roc_auc, past_prediction_horizon, future_prediction_horizon, threshold, fa_rate, md_rate))
                    print 'Model=%s, ROC AUC=%s, Past Prediction Horizon=%s, Future Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' \
                          % future_prediction_model_results[-1]
                    print '======================'
                    if best_model_position == -1 or roc_auc > max_score:
                        best_model_position = count
                        max_score = roc_auc
                    print 'Best: Model=%s, ROC AUC=%s, Past Prediction Horizon=%s, Future Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' \
                          % future_prediction_model_results[best_model_position]
                    count += 1
                    print '======================'

    print 'Sorted results'
    future_prediction_model_results.sort(key=lambda s: s[1], reverse=True)
    for future_prediction_model_result in future_prediction_model_results:
        print 'Model=%s, ROC AUC=%s, Past Prediction Horizon=%s, Future Prediction Horizon=%s, Threshold=%s, FA=%s, MD=%s' % future_prediction_model_result


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
    RegressionToolbox.compare_regression_algorithms()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting Adverse Thermal Events in a Smart Building')
    parser.add_argument("-t", "--type", type=int, choices=[1, 2, 3], default=1,
                        help="type of run: "
                             "(1)Best arguments run, "
                             "(2)Compare regression algorithms, "
                             "(3)Compare detection algorithms")
    args = parser.parse_args()
    if args.type == 1:
        best_run()
    elif args.type == 2:
        compare_regression_algorithms()
    elif args.type == 3:
        compare_detection_algorithms('LR') #Change the name to the one you want to use
