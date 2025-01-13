import pickle
import shap
import numpy as np
import pandas as pd

def bin_points(scores, bin_edges):
    assert(bin_edges is not None), "Bins have not been defined"
    scores = scores.squeeze()
    assert(np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)

class calib_function(object):
    def __init__(self, n_bins=20):
        ### Hyperparameters
        self.n_bins = n_bins

        ### Internal variables
        self.fitted = False

    def predict_proba(self, y_score):
        assert(self.fitted is True), "Call calib_function.fit() first"
        y_score = y_score.squeeze()

        # assign test data to bins
        y_bins = bin_points(y_score, self.f_bin_edges)

        # get calibrated predicted probabilities
        y_pred_prob = self.f_obs_risk[y_bins-1] + (y_score-self.f_bin_edges[y_bins-1])/(self.f_bin_edges[y_bins]- self.f_bin_edges[y_bins-1]) * (self.f_obs_risk[y_bins]-self.f_obs_risk[y_bins-1])
        return y_pred_prob
    

class manual_ap2_model_convert(object):
  def __init__(self, model, calibration, feature_names):
    self.models = []
    for p in model.models:
      self.models.append([stage.model for stage in p.stages if not stage.name()=='nop'])
    self.weights = model.weights
    self.feature_names = feature_names
    self.calibration = calibration

  def predict_proba(self, X):
    preds_ = []
    for k, model in enumerate(self.models):
      local_X = X.copy()
      for stage in model[:-1]:
          local_X = stage.transform(local_X)
      preds_.append(model[-1].predict_proba(local_X) * self.weights[k])
    pred_ens = np.sum(np.array(preds_), axis=0)
    pred = self.calibration.predict_proba(pred_ens[:,1])

    return pred