import pyflux as pf


class FuturePredictionModel:

    def __init__(self, future_prediction_horizon):
        self.future_prediction_horizon = future_prediction_horizon
        self.model = None

    def train(self, df):
        pass

    def fit(self):
        pass

    def future(self):
        return self.model.predict(h=self.future_prediction_horizon)['residuals'][self.future_prediction_horizon-1]


class ARIMAFuturePredictionModel(FuturePredictionModel):

    def __init__(self, future_prediction_horizon, ar, ma):
        FuturePredictionModel.__init__(self, future_prediction_horizon)
        self.ar = ar
        self.ma = ma

    def train(self, df):
        self.model = pf.ARIMA(data=df, ar=self.ar, ma=self.ar, integ=0, target='residuals')

    def fit(self):
        self.model.fit("MLE")