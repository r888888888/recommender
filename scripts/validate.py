import sys
import os
import pandas as pd
import pandas_gbq
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
import pickle
from implicit.als import AlternatingLeastSquares
import signal
import calendar
from pathlib import Path
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logging
load_dotenv(find_dotenv())

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOGGER = logging.getLogger("recommender.validate")
GBQ_PROJECT_ID = "danbooru-1343"
GBQ_TABLE = "danbooru_production.post_votes"
GBQ_KEY_PATH = os.environ.get("GOOGLE_JSON_KEY")
MATRIX_PATH = os.environ.get("MATRIX_PATH")

class UtilityMatrixTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, confidence=40):
    self.confidence = confidence

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    return coo_matrix((np.ones(X.shape[0]),
                       (X['post_id'].cat.codes.copy(),
                        X["user_id"].cat.codes.copy()))) * self.confidence

class AlsEstimator(BaseEstimator, TransformerMixin):
  def __init__(self, factors=50,
                     regularization=0.01,
                     iterations=10,
                     filter_seen=True):
    self.factors = factors
    self.regularization = regularization
    self.iterations = iterations
    self.filter_seen = filter_seen

  def fit(self, X, y=None):
    self.model = AlternatingLeastSquares(
      factors=self.factors,
      regularization=self.regularization,
      iterations=self.iterations,
      dtype=np.float32,
      use_native=True,
      use_gpu=True
    )
    self.model.fit(X)
    if self.filter_seen:
      self.fit_X = X
    return self

  def predict(self, X, y=None):
    predictions = np.dot(self.model.item_factors, self.model.user_factors.T)
    if self.filter_seen:
      predictions[self.fit_X.nonzero()] = -99
    return predictions

def dcg_score(y_true, y_score, k=10, gains="exponential"):
  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k], mode="clip")
  if gains == "exponential":
    gains = 2 ** y_true - 1
  elif gains == "linear":
    gains = y_true
  else:
    raise ValueError("Invalid gains option")
  discounts = np.log2(np.arange(len(y_true)) + 2)
  return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
  best = dcg_score(y_true, y_true, k, gains)
  if best == 0:
    return 0
  actual = dcg_score(y_true, y_score, k, gains)
  return actual / best

def get_col(Y, col):
  return np.squeeze(np.asarray(Y[:, col]))

def ndcg_score_matrix(Y_true, Y_score, k=10, gains="exponential"):
  score = 0.0
  n_users = Y_true.shape[1]
  for u in range(n_users):
    s = ndcg_score(get_col(Y_true, u), get_col(Y_score, u))
    score += s
  return score / n_users

from sklearn.model_selection import PredefinedSplit

class LeavePOutByGroup():
  def __init__(self, X, p=5, n_splits=2):
    self.X = X
    self.p = p
    self.n_splits = n_splits
    test_fold = self.X.groupby("user_id").cumcount().apply(lambda x: int(x / p) if x < (n_splits * p) else -1)
    self.s = PredefinedSplit(test_fold)

  def get_n_splits(self, X=None, y=None, groups=None):
    return self.n_splits

  def split(self, X=None, y=None, groups=None):
    return self.s.split()

def ndcg_scorer(estimator, X_test):
  LOGGER.info("starting ndcg scorer for %d", len(X_test))
  truth = UtilityMatrixTransformer(confidence=1).fit_transform(X_test).todense()
  predictions = estimator.predict(X_test)
  results = ndcg_score_matrix(truth, predictions, k=10)
  LOGGER.info("finished ndcg scorer for %d", len(X_test))
  return results

from sklearn.model_selection import cross_val_score, GridSearchCV

rec_pipeline = Pipeline([
  ("matrix", UtilityMatrixTransformer()),
  ("als", AlsEstimator())
])

param_grid = [
  {
    "matrix__confidence": [1, 20, 40],
    "als__factors": [32, 64, 96],
    "als__regularization": [1e-2, 1e-3, 1e-4],
    "als__iterations": [5, 10, 15]
  }
]

data = pd.concat([pd.read_pickle(f) for f in Path(MATRIX_PATH).glob("votes/**/*.pickle")])
data = data.groupby("user_id").filter(lambda x: x["user_id"].count() >= 10)
data["user_id"] = data["user_id"].astype("category")
data["post_id"] = data["post_id"].astype("category")
shuffled_train_set = data.sample(frac=1).sort_values("user_id")
grid_search = GridSearchCV(rec_pipeline, param_grid, cv=LeavePOutByGroup(shuffled_train_set, p=5, n_splits=3), scoring=ndcg_scorer, verbose=1, n_jobs=-1)
grid_search.fit(shuffled_train_set)
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(mean_score, params)

