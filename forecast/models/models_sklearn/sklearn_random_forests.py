
from sklearn.ensemble import RandomForestRegressor

from forecast.models.models_sklearn import BaseModelSKLearn
from forecast.core import FeatureCreation


class SKLRandomForest(BaseModelSKLearn):
    """
    Random Forest regressor.
    """
    def __init__(self, model_hyperparameters: dict):
        """
        Constructor.
        """
        super().__init__()
        self.estimators = model_hyperparameters['estimators']
        self.criterion_trees = model_hyperparameters['criterion_trees']
        self.max_depth = model_hyperparameters['max_depth']
        self.jobs = model_hyperparameters['jobs']

        self.model = RandomForestRegressor(
            n_estimators=self.estimators, criterion=self.criterion_trees, max_depth=self.max_depth, min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=self.jobs, random_state=None,
            verbose=2, warm_start=False, ccp_alpha=0.0, max_samples=None
        )

    def train(self, dataset_dict):
        """
        Trains.
        """
        super()._train(self.model, dataset_dict)

    def predict(self, dataset_dict):
        """
        Predicts.
        """
        return super()._predict(self.model, dataset_dict)
