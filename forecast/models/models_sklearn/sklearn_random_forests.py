
from sklearn.ensemble import RandomForestRegressor

from forecast.models.models_sklearn import BaseModelSKLearn
from forecast.core import FeatureCreation


class SKLRandomForest(BaseModelSKLearn):
    """
    Random Forest regressor.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        """
        super().__init__(features)
        self.model = RandomForestRegressor(
            n_estimators=self.estimators, criterion=self.criterion_trees, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=self.jobs, random_state=None,
            verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None
        )

    def train(self):
        """
        Trains.
        """
        super()._train(self)

    def predict(self):
        """
        Predicts.
        """
        return super()._predict(self)
