import numpy as np
from sklearn.utils import resample

# TODO: Need to change this so that it works for surrogate models that do not return variance (aleatory uncertainty)
class DeepEnsembleSurrogate:
    """Creates a Deep Ensemble of a given surrogate model"""

    def __init__(self, surrogate, data_pipeline, ensemble_size=5, max_data=None, surrogate_config={}): # TODO: Change max_data back to 200 by default
        self.members = [surrogate(**surrogate_config) for _ in range(ensemble_size)]
        self.data_pipeline = data_pipeline
        self.ensemble_size = ensemble_size
        self.max_data = max_data
        self.fitted_ = False
    
    def fit(self, X, y):
        y = np.asarray(y)

        if self.max_data is not None and len(X) > self.max_data:
            X = X[-self.max_data:]
            y = y[-self.max_data:]
        # Transform data
        X = self.data_pipeline.fit_transform(X)

        # Fit each member of ensemble
        for member in self.members:
            if self.max_data:
                n_samples = min(len(X), self.max_data)
            else:
                n_samples = len(X)

            # Bootstrap with replacement
            X_boot, y_boot = resample(X, y, n_samples = n_samples, replace=True)
            member.fit(X_boot, y_boot)

        self.fitted_ = True

        return self
    
    def predict(self, X, return_predictive_error=False):
        X = self.data_pipeline.transform(X)
        means, variances = self._collect_member_preds(X)
        mean_pred = means.mean(axis=0)
        if return_predictive_error:
            aleatory   = variances.mean(axis=0)
            epistemic  = means.var(axis=0)
            predictive = aleatory + epistemic
            return mean_pred, epistemic, aleatory
        return mean_pred
    
    def _collect_member_preds(self, X):
        means, variances = [], []
        for m in self.members:
            mu, var = self._member_predict(m, X)
            means.append(mu)
            variances.append(var)
        return np.vstack(means), np.vstack(variances)

    def _member_predict(self, member, X):
        # Get mean and std
        y_mean, y_std = member.predict(X, return_std=True)
        return y_mean, (y_std**2) # square for variance


