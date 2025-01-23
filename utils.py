from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import statsmodels.api as sm
import numpy as np

def pvalue_selection(X, y):
  initial_features = X.columns.tolist()
  selected_features = []
  while initial_features:
      remaining_features = list(set(initial_features) - set(selected_features))
      new_pval = pd.Series(index=remaining_features, dtype=float)

      for feature in remaining_features:
          model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
          new_pval[feature] = model.pvalues[feature]
      
      min_pval = new_pval.min()
      if min_pval < 0.001:  # Set significance level
          selected_features.append(new_pval.idxmin())
      else:
          break

  return selected_features

def auc_selection(X, y, threshold):
  features = X.columns.tolist()
  # features.remove("CA15")

  X_const = sm.add_constant(X)
  selected_features = []

  best_auc = 0
  max_iteration = len(features)

  for _ in range(0,max_iteration):
    aucs = []
    for feature_2 in features:
      features_to_use = [*selected_features, feature_2]
      log_reg_sm = sm.Logit(y, X_const[features_to_use]).fit(disp=0)
      y_pred = log_reg_sm.predict(X_const[features_to_use])
      auc = roc_auc_score(y, y_pred)
      aucs.append(auc)
      
    if max(aucs) - best_auc > threshold:
      best_feature = aucs.index(max(aucs))
      selected_features.append(features[best_feature])
      best_auc = max(aucs)
      features.remove(features[best_feature])
    
    aucs = []
  return selected_features

def accuracy_selection(X, y, threshold):
  initial_features = X.columns.tolist()
  selected_features = []
  best_accuracy = 0
  max_iteration = len(initial_features)

  for _ in range(max_iteration):
    accuracies = []
    for feature in initial_features:
      features_to_use = selected_features + [feature]
      X_train_const = sm.add_constant(X[features_to_use])
      log_reg_sm = sm.Logit(y, X_train_const).fit(disp=0)
      y_pred = log_reg_sm.predict(X_train_const)
      accuracy = accuracy_score(y, np.round(y_pred))
      accuracies.append(accuracy)
      #print(accuracies)

    if max(accuracies) - best_accuracy > threshold:
      best_feature = initial_features[accuracies.index(max(accuracies))]
      selected_features.append(best_feature)
      best_accuracy = max(accuracies)
      initial_features.remove(best_feature)
    else:
      break

  return selected_features

def logistic_regression(
  X_train,
  X_test,
  y_train,
  y_test,
  features,
  verbose=True, # prints the summary
):
  X_train_const = sm.add_constant(X_train[features])
  X_test_const = sm.add_constant(X_test[features])

  # Fit the logistic regression model
  log_reg = sm.Logit(y_train, X_train_const).fit(disp=0)

  # Extract p-values and coefficients with more decimals
  summary_frame = log_reg.summary2().tables[1]  # Extract the coefficient table
  summary_frame["P>|z|"] = summary_frame["P>|z|"].apply(lambda x: f"{x:.8f}")  # Format p-values
  summary_frame["Coef."] = summary_frame["Coef."].apply(lambda x: f"{x:.8f}")  # Format coefficients

  # Predictions
  y_pred = log_reg.predict(X_test_const)
  y_pred = np.round(y_pred)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = conf_matrix.ravel()
  precision = tp / (tp + fp)
  specificity = tn / (tn + fp)
  sensitivity = tp / (tp + fn)
  f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
  auc = roc_auc_score(y_test, log_reg.predict(X_test_const))

  index_info = {
    "metrics": {
      "accuracy": accuracy,
      "precision": precision,
      "specificity": specificity,
      "sensitivity": sensitivity,
      "f1": f1,
      # "conf_matrix": conf_matrix,
      "auc": auc
    },
    "summary": summary_frame,
  }

  if verbose:
    print(summary_frame)
  return log_reg, index_info

def get_vif(data, features):
  features_const = sm.add_constant(data[features])
  vif_data = pd.DataFrame()
  vif_data["Feature"] = features_const.columns
  vif_data["VIF"] = [variance_inflation_factor(features_const.values, i) for i in range(features_const.shape[1])]
  return vif_data

