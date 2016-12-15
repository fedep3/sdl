import pyflux as pf


class FuturePredictionModel:
    """
    Base class for different prediction models.
    """

    def __init__(self, future_prediction_horizon):
        self.future_prediction_horizon = future_prediction_horizon
        self.model = None

    def train(self, df):
        """
        Trains the model
        :param df: Data used to train.
        :return: nothing.
        """
        pass

    def fit(self):
        """
        Fits the data.
        :return: nothing.
        """
        pass

    def future(self):
        """
        Returns the predicted value for the model future prediction horizon.
        :return: prediction.
        """
        return self.model.predict(h=self.future_prediction_horizon)['residuals'][self.future_prediction_horizon - 1]


class ARIMAFuturePredictionModel(FuturePredictionModel):

    def __init__(self, future_prediction_horizon, ar, ma):
        FuturePredictionModel.__init__(self, future_prediction_horizon)
        self.ar = ar
        self.ma = ma

    def train(self, df):
        self.model = pf.ARIMA(data=df, ar=self.ar, ma=self.ma, integ=0, target='residuals')

    def fit(self):
        self.model.fit('MLE')

    def __str__(self):
        return 'ARIMA(p=%s, q=%s)' % (self.ar, self.ma)


class GARCHFuturePredictionModel(FuturePredictionModel):

    def __init__(self, future_prediction_horizon, p, q):
        FuturePredictionModel.__init__(self, future_prediction_horizon)
        self.p = p
        self.q = q

    def train(self, df):
        self.model = pf.GARCH(data=df, p=self.p, q=self.q, target='residuals')

    def fit(self):
        self.model.fit()

    def __str__(self):
        return 'GARCH(p=%s, q=%s)' % (self.p, self.q)


class GGSMFuturePredictionModel(FuturePredictionModel):

    def __init__(self, future_prediction_horizon):
        FuturePredictionModel.__init__(self, future_prediction_horizon)

    def train(self, df):
        self.model = pf.LLEV(data=df, target='residuals')

    def fit(self):
        self.model.fit()

    def __str__(self):
        return 'GGSM'


class AggregatingFuturePredictionModel(FuturePredictionModel):

    def __init__(self, future_prediction_horizon):
        FuturePredictionModel.__init__(self, future_prediction_horizon)

    def train(self, df):
        mix = pf.Aggregate(learning_rate=1.0, loss_type='squared')
        model_one = pf.ARIMA(data=df, ar=1, ma=0)
        model_two = pf.ARIMA(data=df, ar=2, ma=0)
        model_three = pf.LLEV(data=df)

        mix.add_model(model_one)
        mix.add_model(model_two)
        mix.add_model(model_three)
        mix.tune_learning_rate(16)

        if __debug__:
            print mix.learning_rate
            mix.plot_weights(h=16, figsize=(15, 5))

        self.model = mix

    def fit(self):
        pass

    def future(self):
        return self.model.predict(h=self.future_prediction_horizon).values[self.future_prediction_horizon - 1]

    def __str__(self):
        return 'Aggregating'
