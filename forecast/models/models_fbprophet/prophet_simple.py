from forecast.models.models_fbprophet import BaseModelProphet
from prophet import Prophet

class ProphetSimple(BaseModelProphet):

    def __init__(self, model_hyperparameters: dict):
        BaseModelProphet.__init__(self)
        self.model = Prophet()
        self.model.weekly_seasonality = model_hyperparameters['weekly_seasonality']
        self.model.yearly_seasonality = model_hyperparameters['yearly_seasonality']
        self.model.daily_seasonality = model_hyperparameters['daily_seasonality']

    def train(self, dataset):
        """
        Trains.
        """
        super()._train(self.model, dataset)


    def predict(self, dataset):
        """
        Predicts.
        """
        return super()._predict(self.model, dataset)