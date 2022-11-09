from math import sqrt
import pandas as pd
from fitter import Fitter, get_common_distributions
import scipy

class PNN():
  def __init__(self) -> None:
    pass

  def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
    self.x_train = x_train
    self.y_train = y_train
    self.mem = {}

    for class_name in self.y_train.unique():
      filter = (self.y_train == class_name)
      features_dist = {}

      for feature in self.x_train:
        data = self.x_train.loc[filter, feature]
        f = Fitter(data, distributions=get_common_distributions(), bins=round(sqrt(y_train.size)))
        f.fit()

        best_dist = f.get_best()
        dist_fun = eval("scipy.stats." + list(best_dist.keys())[0])
        dist_params = list(best_dist.values())[0]

        features_dist[feature] = {
          "fun": dist_fun,
          "parmas": dist_params
        }

      self.mem[class_name] = features_dist

  def predict(self, x_test: pd.DataFrame):
    y_pred = []

    for i, x in x_test.iterrows():
      class_score = {}

      for class_name in self.y_train.unique():
        score = 0

        for feature in self.x_train:
          dist_fun = self.mem[class_name][feature]["fun"]
          dist_params = self.mem[class_name][feature]["parmas"]
          score += dist_fun.pdf(x[feature].item(), **dist_params)

        class_score[class_name] = score

      best = max(class_score, key=class_score.get)
      y_pred.append(best)

    return y_pred
