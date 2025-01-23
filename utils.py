from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from matplotlib import pyplot as plt
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
  y_pred_prob = log_reg.predict(X_test_const)
  y_pred = np.round(y_pred_prob)

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
      "auc": auc
    },
    "summary": summary_frame,
  }

  if verbose:
    print(summary_frame)
  return log_reg, index_info, y_pred_prob

def get_vif(data, features):
  features_const = sm.add_constant(data[features])
  vif_data = pd.DataFrame()
  vif_data["Feature"] = features_const.columns
  vif_data["VIF"] = [variance_inflation_factor(features_const.values, i) for i in range(features_const.shape[1])]
  return vif_data

def threshold_estimation(X, y, selected_features):
  # Initialize StratifiedKFold
  kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

  # Initialize lists to store results
  thresholds = np.arange(0.0, 1.1, 0.1)

  sensitivity_array = []
  specificity_array = []
  accuracy_array = []
  aic_array = []
  bic_array = []
  # Perform cross-validation
  for train_index, test_index in tqdm(kf.split(X[selected_features], y)):
      X_train, X_test = X[selected_features].iloc[train_index], X[selected_features].iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      X_train = sm.add_constant(X_train)
      X_test = sm.add_constant(X_test)

      # Fit the logistic regression model
      log_reg = sm.Logit(y_train, X_train).fit(disp=0)

      sensitivity_list = []
      specificity_list = []
      accuracy_list = []

      # Predict probabilities
      y_prob = log_reg.predict(X_test)
      
      # Calculate sensitivity and specificity for each threshold
      for threshold in thresholds:
          y_pred = (y_prob >= threshold).astype(int)
          tp = np.sum((y_test == 1) & (y_pred == 1))
          tn = np.sum((y_test == 0) & (y_pred == 0))
          fp = np.sum((y_test == 0) & (y_pred == 1))
          fn = np.sum((y_test == 1) & (y_pred == 0))

          accuracy = accuracy_score(y_test, np.round(y_pred))
          
          sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
          specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
          
          sensitivity_list.append(sensitivity)
          specificity_list.append(specificity)
          accuracy_list.append(accuracy)

      aic_value = log_reg.aic
      bic_value = log_reg.bic

      # Convert lists to numpy arrays for easier manipulation
      sensitivity_array.append(sensitivity_list)
      specificity_array.append(specificity_list)
      accuracy_array.append(accuracy_list)
      aic_array.append(aic_value)
      bic_array.append(bic_value)


  sensitivity_array = np.array(sensitivity_array)
  specificity_array = np.array(specificity_array)
  accuracy_array = np.array(accuracy_array)
  aic_array = np.array(aic_array)
  bic_array = np.array(bic_array)

  sensitivity_mean = np.array(sensitivity_array).mean(axis=0)
  specificity_mean = np.array(specificity_array).mean(axis=0)
  accuracy_mean = np.array(accuracy_array).mean(axis=0)

  sensitivity_std = np.array(sensitivity_array).std(axis=0)
  specificity_std = np.array(specificity_array).std(axis=0)
  accuracy_std = np.array(accuracy_array).std(axis=0)

  sensitivity = {
    "mean": sensitivity_mean,
    "std": sensitivity_std
  }
  specificity = {
    "mean": specificity_mean,
    "std": specificity_std
  }
  accuracy = {
    "mean": accuracy_mean,
    "std": accuracy_std
  }

  return thresholds, sensitivity, specificity, accuracy, aic_array, bic_array

def thresholds_plotter(thresholds, sensitivity, specificity, accuracy):
  sensitivity_mean = sensitivity["mean"]
  specificity_mean = specificity["mean"]
  accuracy_mean = accuracy["mean"]
  sensitivity_std = sensitivity["std"]
  specificity_std = specificity["std"]
  accuracy_std = accuracy["std"]
  # Create subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

  # Plot sensitivity, specificity, and accuracy
  ax1.plot(thresholds, sensitivity_mean, label='Sensitivity')
  ax1.plot(thresholds, specificity_mean, label='Specificity')
  ax1.plot(thresholds, accuracy_mean, label='Accuracy')
  ax1.fill_between(thresholds, sensitivity_mean - sensitivity_std, sensitivity_mean + sensitivity_std, color='blue', alpha=0.2, label='Standard Deviation')
  ax1.fill_between(thresholds, specificity_mean - specificity_std, specificity_mean + specificity_std, color='orange', alpha=0.2, label='Standard Deviation')
  ax1.fill_between(thresholds, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, color='green', alpha=0.2, label='Standard Deviation')

  ax1.set_xlabel('Threshold')
  ax1.set_ylabel('Score')
  ax1.set_title('Sensitivity, Specificity, and Accuracy vs. Threshold')
  ax1.legend()
  ax1.grid(True)

  # Plot ROC curve based on sensitivity and specificity
  ax2.plot(1 - specificity_mean, sensitivity_mean, color='darkorange', lw=2, label='ROC curve')
  ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  ax2.fill_between(1 - specificity_mean, sensitivity_mean - sensitivity_std, sensitivity_mean + sensitivity_std, color='darkorange', alpha=0.3, label='Standard Deviation')
  ax2.set_xlim([0.0, 1.0])
  ax2.set_ylim([0.0, 1.05])
  ax2.set_xlabel('False Positive Rate (1 - Specificity)')
  ax2.set_ylabel('True Positive Rate (Sensitivity)')
  ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
  ax2.legend(loc="lower right")
  ax2.grid(True)

  plt.tight_layout()
  plt.show()
