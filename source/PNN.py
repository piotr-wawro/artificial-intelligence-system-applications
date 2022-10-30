import pandas as pd
from fitter import Fitter, get_common_distributions

def PNN(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame):
  mem = {}

  for class_name in pd.unique(y_train.unique):
    filter = y_train[y_train == class_name]

    for feature in x_train:
      data = feature.iloc[filter]
      dist = Fitter(data, distributions=get_common_distributions())

      