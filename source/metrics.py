from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_confusion_matrix(y_true, y_pred):
  matrix = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))

  group_counts = [f"{value:.0f}" for value in
                  matrix.flatten()]

  group_percentages = [f"{value:.2%}" for value in
                      matrix.flatten()/np.sum(matrix)]

  labels = [f"{v1}\n\n{v2}" for v1, v2 in
            zip(group_counts,group_percentages)]

  labels = np.asarray(labels).reshape(np.unique(y_pred).size, -1)

  sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues',
              xticklabels=np.unique(y_pred), yticklabels=np.unique(y_pred))
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

def print_summary(y_true, y_pred):
  summary = pd.concat([
    class_summary(y_true, y_pred),
    empty_row(),
    model_summary(y_true, y_pred)
  ])
  summary = summary.fillna('')

  print(summary)

def class_summary(y_true, y_pred):
  class_metrics = pd.DataFrame(index = np.unique(y_pred))

  class_metrics["accuracy"] = accuracy_score(y_true, y_pred)
  class_metrics["precision"] = precision_score(y_true, y_pred)
  class_metrics["sensitivity"] = sensitivity_score(y_true, y_pred)
  class_metrics["specificity"] = specificity_score(y_true, y_pred)
  class_metrics["f1-score"] = FScore(y_true, y_pred)
  class_metrics["support"] = np.unique(y_true, return_counts=True)[1]

  class_metrics = class_metrics.round(3)

  return class_metrics

def model_summary(y_true, y_pred):
  model_metrics = pd.DataFrame(index = ['accuracy'])

  model_metrics["accuracy"] = [accuracy_score(y_true, y_pred, True)]
  if np.unique(y_pred).size == 2:
    model_metrics["precision"] = [precision_score(y_true, y_pred, True)]
    model_metrics["sensitivity"] = [sensitivity_score(y_true, y_pred, True)]
    model_metrics["specificity"] = [specificity_score(y_true, y_pred, True)]
    model_metrics["fscore"] = [FScore(y_true, y_pred, True)]

  model_metrics = model_metrics.round(3)
  model_metrics = model_metrics.transpose()

  return model_metrics

def empty_row():
  empty = pd.DataFrame(index = [''])
  empty["accuracy"] = [pd.NA]
  return empty

def accuracy_score(y_true, y_pred, all=False):
  TP, TN, FP, FN = predictive_values(y_true, y_pred)

  if all:
    ACC = TP.sum()/y_true.size
  else:
    ACC = (TP+TN)/(TP+FP+FN+TN)

  return ACC

def precision_score(y_true, y_pred, binary=False):
  TP, TN, FP, FN = predictive_values(y_true, y_pred, binary)
  PPV = TP/(TP+FP) # positive predictive value
  return PPV

def sensitivity_score(y_true, y_pred, binary=False):
  TP, TN, FP, FN = predictive_values(y_true, y_pred, binary)
  TPR = TP/(TP+FN) # true positive rate / recall
  return TPR

def specificity_score(y_true, y_pred, binary=False):
  TP, TN, FP, FN = predictive_values(y_true, y_pred, binary)
  TNR = TN/(TN+FP) # true negative rate
  return TNR

def FScore(y_true, y_pred, binary=False):
  PPV = precision_score(y_true, y_pred, binary)
  TPR = sensitivity_score(y_true, y_pred, binary)
  F1 = (2*PPV*TPR)/((PPV+TPR))
  return F1

def predictive_values(y_true, y_pred, binary=False):
  matrix = confusion_matrix(y_true, y_pred)

  FP = matrix.sum(axis=0) - np.diag(matrix)
  FN = matrix.sum(axis=1) - np.diag(matrix)
  TP = np.diag(matrix)
  TN = matrix.sum() - (FP + FN + TP)

  if binary:
    TP = TP[0]
    TN = TN[0]
    FP = FP[0]
    FN = FN[0]

  return TP, TN, FP, FN
