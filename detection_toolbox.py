import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc


TEMPERATURE_THRESHOLD = 68.1


class DetectionToolbox:

    def __init__(self, regression_model, past_prediction_horizon, future_prediction_model):
        self.regression_model = regression_model
        self.past_prediction_horizon = past_prediction_horizon
        self.future_prediction_model = future_prediction_model

    def predict_residuals(self, ts, xs, ys):
        # Calculate residuals
        predictions = []
        prediction_residuals = []
        for x in xrange(len(xs)):
            predictions.append(self.regression_model.predict(xs[x])[0])
            prediction_residuals.append(predictions[x] - ys[x])

        if __debug__:
            fig = plt.figure()
            plt.title('Actual vs. Prediction')
            plt.plot(ys)
            plt.plot(predictions)
            plt.plot(prediction_residuals)
            fig.savefig('residuals.png')

        # Predicting future residuals
        prediction_future = []
        for x in xrange(len(prediction_residuals)-self.past_prediction_horizon):
            df = pd.DataFrame(prediction_residuals[x:x+self.past_prediction_horizon], index=ts[x:x + self.past_prediction_horizon], columns=['residuals'])
            self.future_prediction_model.train(df)
            self.future_prediction_model.fit()
            prediction_future.append(self.future_prediction_model.future())

        return prediction_future

    def calculate_roc_curve(self, future_residuals_prediction, ys):
        fp_rate_data = []
        fn_rate_data = []
        tp_rate_data = []

        ideal_threshold = -1.0
        ideal_fp_rate = -1.0
        ideal_fn_rate = -1.0

        threshold_found = False

        for t in xrange(1, 81):
            threshold = 1.0 + float(t) * 0.05
            if __debug__:
                print 'Checking threshold: ', threshold
            fp_rate, fn_rate, tp_rate = \
                self.calculate_counts(future_residuals_prediction, ys, threshold)
            
            if __debug__:
                print '(True Positive) True Alarm Rate: ', tp_rate
                print '(False Positive) False Alarm Rate: ', fp_rate
                print '(False Negative) Missed Detection Rate: ', fn_rate
                print '------------------------------------'

            if not threshold_found and fp_rate < fn_rate:
                ideal_fp_rate, ideal_fn_rate, ideal_tp_rate = self.calculate_counts(future_residuals_prediction, ys, threshold - 0.025)
                ideal_threshold = threshold - 0.025
                threshold_found = True
            
            fp_rate_data.append(fp_rate)
            fn_rate_data.append(fn_rate)
            tp_rate_data.append(tp_rate)

        roc_auc = auc(np.array(fp_rate_data), np.array(tp_rate_data), reorder=True)
        
        if __debug__:
            print 'Plotting Detection Error Tradeoff'
            fig = plt.figure()
            plt.title('Detection Error Tradeoff')
            plt.plot(fp_rate_data, fn_rate_data, 'b')
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.ylabel('Missed Detection Rate')
            plt.xlabel('False Positive Rate')
            fig.savefig('det.png')
            print 'Plotting ROC Curve'
            fig = plt.figure()
            plt.title('ROC Curve')
            plt.plot(fp_rate_data, tp_rate_data, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            fig.savefig('roc.png')

        return roc_auc, ideal_threshold, ideal_fp_rate, ideal_fn_rate

    def calculate_counts(self, future_residuals_prediction, ys, threshold):
        return self._calculate_error_rates(threshold,
                                           future_residuals_prediction[
                                           :-self.future_prediction_model.future_prediction_horizon],
                                           ys[self.past_prediction_horizon:])

    def _calculate_error_rates(self, threshold, future_residuals_prediction, actuals):
        La = threshold

        # Predict alarms with the given level
        alarms = []
        for p in future_residuals_prediction:
            if abs(p) > La:
                alarms.append(True)
            else:
                alarms.append(False)

        # Calculate whether or not there was actually an alarm anywhere
        # in the next time period
        alarm_actuals = []
        for i in xrange(len(actuals) - self.future_prediction_model.future_prediction_horizon):
            alarm_actual = False
            for x in xrange(self.future_prediction_model.future_prediction_horizon):
                if actuals[i + x] < TEMPERATURE_THRESHOLD:
                    alarm_actual = True
            alarm_actuals.append(alarm_actual)

        # Calculate the false alarm and missed detection rate
        fp_count = 0
        fn_count = 0
        tp_count = 0
        tn_count = 0
        for r in xrange(len(alarms)):
            if alarms[r] == True and alarm_actuals[r] == False:
                fp_count += 1
            elif alarms[r] == False and alarm_actuals[r] == True:
                fn_count += 1
            elif alarms[r] == True and alarm_actuals[r] == True:
                tp_count += 1
            elif alarms[r] == False and alarm_actuals[r] == False:
                tn_count += 1

        fp_rate = float(fp_count) / float(fp_count + tn_count)
        fn_rate = float(fn_count) / float(tp_count + fn_count)
        tp_rate = float(tp_count) / float(tp_count + fn_count)

        # return total events, false alarms, true alarms, missed detections
        return fp_rate, fn_rate, tp_rate


