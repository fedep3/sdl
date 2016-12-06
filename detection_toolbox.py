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
        fa_rate_data = []
        md_rate_data = []

        ideal_threshold = -1.0
        ideal_fa_rate = -1.0
        ideal_md_rate = -1.0

        threshold_found = False
        for t in xrange(1, 41):
            threshold = 1.0 + float(t) * 0.1
            if __debug__:
                print 'Checking threshold: ', threshold
            correct_count, fa_count, md_count = \
                self.calculate_counts(future_residuals_prediction, ys, threshold)
            total_count = correct_count + fa_count + md_count
            if __debug__:
                print 'Correct Alarm Count: ', correct_count
                print 'False Alarm Count: ', fa_count
                print 'Missed Detection Count: ', md_count
                print '------------------------------------'
            fa_rate = float(fa_count) / total_count
            md_rate = float(md_count) / total_count

            if not threshold_found and fa_count < md_count:
                ideal_fa_rate = fa_rate
                ideal_md_rate = md_rate
                ideal_threshold = threshold
                threshold_found = True
            
            fa_rate_data.append(fa_rate)
            md_rate_data.append(md_rate)

        if __debug__:
            print 'Plotting'
            roc_auc = auc(np.array(fa_rate_data), np.array(md_rate_data), reorder=True)
            fig = plt.figure()
            plt.title('Detection Error Tradeoff')
            plt.plot(fa_rate_data, md_rate_data, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 0.4])
            plt.ylim([0.0, 0.4])
            plt.ylabel('Missed Detection Rate')
            plt.xlabel('False Positive Rate')
            fig.savefig('roc.png')
        else:
            roc_auc = 0

        return roc_auc, ideal_threshold, ideal_fa_rate, ideal_md_rate

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
        correct_count = 0
        fa_count = 0
        md_count = 0
        for r in xrange(len(alarms)):
            if alarms[r] == True and alarm_actuals[r] == False:
                fa_count += 1
            elif alarms[r] == False and alarm_actuals[r] == True:
                md_count += 1
            else:
                correct_count += 1

        return correct_count, fa_count, md_count
