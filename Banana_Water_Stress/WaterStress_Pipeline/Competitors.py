import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#from supervised_cd_utils import Evaluate_competitors
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
import random
import sys

sys.path.append(r'/Discretization-MDLPC-master')


def predict_competitors(X, y, jump):
    random.seed(28)
    ind = jump * 2
    break_points = np.arange(start=0, stop=y.shape[0] + jump, step=jump)
    neighbors = 5

    y_pred_adwin = np.zeros(y.shape[0])
    y_adwin_proba = np.zeros(y.shape[0])
    y_pred_awe = np.zeros(y.shape[0])
    y_awe_proba = np.zeros(y.shape[0])
    y_pred_dwm = np.zeros(y.shape[0])
    y_dwm_proba = np.zeros(y.shape[0])
    y_pred_lnse = np.zeros(y.shape[0])
    y_lnse_proba = np.zeros(y.shape[0])
    y_pred_srpc = np.zeros(y.shape[0])
    y_srpc_proba = np.zeros(y.shape[0])

    knn_adwin = KNNADWINClassifier(n_neighbors=neighbors)
    awe = AccuracyWeightedEnsembleClassifier(window_size=jump)
    dwm = DynamicWeightedMajorityClassifier(period=jump)  # , n_estimators=jump)
    lnse = LearnPPNSEClassifier(window_size=jump)
    srpc = StreamingRandomPatchesClassifier(n_estimators=10, subspace_size=90)

    Xt, yt = X[:neighbors], y[:neighbors]
    for j in range(neighbors):
        knn_adwin.partial_fit(Xt[j].reshape(1, -1), yt[j].reshape(1), classes=[0, 1])

    # X_first = X[break_points[0]:break_points[1]]
    # Y_first = y[break_points[0]:break_points[1]]
    # awe.fit(X_first, Y_first, classes=[0, 1])
    # y_pred_awe[0:jump] = awe.predict(X[0:jump])
    # print("Y_first pred is: ", y_pred_awe[0:jump])

    # Predict randomly the first day
    for i in tqdm(range(2 * jump)):
        y_pred_adwin[i] = knn_adwin.predict(X[i].reshape(1, -1))
        y_adwin_proba[i] = np.double(knn_adwin.predict_proba(X[i].reshape(1, -1))[0][1])
        y_pred_awe[i] = awe.predict(X[i].reshape(1, -1))
        # y_awe_proba[i] = np.double(awe.predict_proba(X[i].reshape(1, -1))[0][1])
        y_pred_dwm[i] = dwm.predict(X[i].reshape(1, -1))
        # y_dwm_proba[i] = np.double(dwm.predict_proba(X[i].reshape(1, -1))[0][1])
        y_pred_lnse[i] = lnse.predict(X[i].reshape(1, -1))
        y_lnse_proba[i] = np.double(lnse.predict_proba(X[i].reshape(1, -1))[0])
        y_pred_srpc[i] = srpc.predict(X[i].reshape(1, -1))
        y_srpc_proba[i] = np.double(srpc.predict_proba(X[i].reshape(1, -1))[0])

    # Fit the day before and predict the current day
    for b in tqdm(range(2, len(break_points) - 1)):
        X_day_i_minus = X[break_points[b - 2]:break_points[b]]
        y_day_i_minus = y[break_points[b - 2]:break_points[b]]
        X_day_i = X[break_points[b]:break_points[b + 1]]

        # knn_adwin.fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
        # awe.fit(X_day_i_minus, y_day_i_minus)
        # dwm.fit(X_day_i_minus, y_day_i_minus)
        # lnse.fit(X_day_i_minus, np.array(y_day_i_minus),classes=[0, 1])
        # srpc.fit(X_day_i_minus, np.array(y_day_i_minus))

        for j in range(X_day_i_minus.shape[0]):
            knn_adwin.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
            awe.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
            dwm.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1), classes=[0, 1])
            lnse.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1), classes=[0, 1])
            srpc.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1))

        for k in range(X_day_i.shape[0]):
            y_pred_adwin[ind] = knn_adwin.predict(X_day_i[k].reshape(1, -1))
            y_adwin_proba[ind] = np.double(knn_adwin.predict_proba(X_day_i[k].reshape(1, -1))[0, 1])
            y_pred_awe[ind] = awe.predict(X_day_i[k].reshape(1, -1))
            # y_awe_proba[ind] = np.double(awe.predict_proba(X_day_i[k].reshape(1, -1))[0, 1])
            y_pred_dwm[ind] = dwm.predict(X_day_i[k].reshape(1, -1))
            # y_dwm_proba[ind] = np.double(dwm.predict_proba(X_day_i[k].reshape(1, -1))[0, 1])
            y_pred_lnse[ind] = lnse.predict(X_day_i[k].reshape(1, -1))
            y_lnse_proba[ind] = np.double(lnse.predict_proba(X_day_i[k].reshape(1, -1))[0, 1])
            y_pred_srpc[ind] = srpc.predict(X_day_i[k].reshape(1, -1))
            y_srpc_proba[ind] = np.double(srpc.predict_proba(X_day_i[k].reshape(1, -1))[0, 1])
            ind += 1

    return y_pred_adwin, y_pred_awe, y_pred_dwm, y_pred_lnse, y_pred_srpc, \
           y_adwin_proba, y_awe_proba, y_dwm_proba, y_lnse_proba, y_srpc_proba


def compute_scores(pred, probs, true, jump):
    random.seed(28)
    # day = np.arange(1, 42, 1)
    # day = day.reshape(-1, 1)
    break_points = np.arange(start=0, stop=true.shape[0] + jump, step=jump)
    all_results = np.zeros((len(break_points) - 1, 17))
    day = 1
    cols = ["Day", "TP", "FP", "FN", "TN", "FA", "TP_Precision", "TP_Recall", "TN_Precision", "TN_Recall", "F1_pos",
            "F1_neg", "F1_Avg", "Geometric_mean", "Balanced_Accuracy", "Y_score", "AUC"]

    for j in range(len(break_points) - 1):
        y_pred, y_true = pred[break_points[j]:break_points[j + 1]], true[break_points[j]:break_points[j + 1]]
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        auc = 0.5
        for i in range(y_true.shape[0]):
            if y_pred[i] == 1 and y_true[i] == 1:
                tp += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                fn += 1
            if y_pred[i] == 0 and y_true[i] == 0:
                tn += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                fp += 1
        if tp == 0:
            tp_precision = 0
            tp_recall = 0
        else:
            tp_precision = tp / (tp + fp)
            tp_recall = tp / (tp + fn)
        if tn == 0:
            tn_precision = 0
            tn_recall = 0
        else:
            tn_precision = tn / (tn + fn)
            tn_recall = tn / (tn + fp)
        if (tp_precision + tp_recall) == 0:
            f1_pos = 0
        else:
            f1_pos = 2 * tp_precision * tp_recall / (tp_precision + tp_recall)
        if (tn_precision + tn_recall) == 0:
            f1_neg = 0
        else:
            f1_neg = 2 * tn_precision * tn_recall / (tn_precision + tn_recall)
        if fp + tn == 0:
            fa_rate = 0
        else:
            fa_rate = fp / (fp + tn)

        f1_avg = (f1_pos + f1_neg) / 2
        gm = np.sqrt(tp_recall * tn_recall)
        b_accuracy = (tp_recall + tn_recall) / 2
        y_score = (tp_recall + tn_recall + tp_precision + tn_precision) / 4
        if j > 7:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, probs[break_points[j]:break_points[j + 1]], pos_label=1)
            auc = metrics.auc(fpr, tpr)
        all_results[j] = [day, tp, fp, fn, tn, fa_rate, tn_precision, tp_recall, tn_precision, tn_recall, f1_pos,
                          f1_neg, f1_avg, gm, b_accuracy, y_score, auc]
        day = day + 1
    all_results = pd.DataFrame(all_results)
    all_results.columns = cols
    return all_results


# -------------------------------------------- Evaluate Competitors -------------------------------------------------- #

# Set parameters
random.seed(28)
jump = 201
time_steps = 28
steps = range(1, time_steps + 1)

# Load data
data_path = os.path.abspath(
    r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Fertilizer_Stress\Data')
save_results_path = os.path.abspath(
    r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Fertilizer_Stress\Results\Competitors')

data_disc = pd.read_csv(data_path + '/‏‏fs_disc_entropy_stressOnly.csv')
data_all_smooth = pd.read_csv(os.path.join(data_path, 'fs_features_smooth.csv'))

# Save labels and get only the selected features
y = data_disc['class']
y = np.array(y)
y = y - 1
del data_disc['class']
columns = data_disc.columns
X_selected = data_all_smooth[columns]
Plants = data_all_smooth[['Date', 'Treatment', 'Num']]
columns = ['Date', 'Treatment', 'Plant_Num', 'Prediction', 'Probability']

# Scale the selected data
eval_competitors = Evaluate_competitors(is_batchs=True, jump=jump, n_neighbors=5, metric='F1')
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected.copy())

# Get predictions for all competing algorithms
res_adwin, res_awe, res_dwm, res_lnse, res_srp, \
adwin_probs, awe_probs, dwm_probs, lnse_probs, srpc_probs = predict_competitors(X_selected, y, jump)
# res_adwin, res_awe, res_dwm, res_lnse, res_srp = eval_competitors.fit_predict_evaluate(X_selected, y)

# Compute all algorithms' scores
Results_adwin = compute_scores(res_adwin, adwin_probs, y, jump)
Results_awe = compute_scores(res_awe, awe_probs, y, jump)
Results_dwm = compute_scores(res_dwm, dwm_probs, y, jump)
Results_lnse = compute_scores(res_lnse, lnse_probs, y, jump)
Results_srpc = compute_scores(res_srp, srpc_probs, y, jump)

# Save all algorithms' scores
Results_adwin.to_csv(os.path.join(save_results_path, 'adwin_stressOnly_results.csv'), index=False)
Results_awe.to_csv(os.path.join(save_results_path, 'awe_stressOnly_results.csv'), index=False)
Results_dwm.to_csv(os.path.join(save_results_path, 'dwm_stressOnly_results.csv'), index=False)
Results_lnse.to_csv(os.path.join(save_results_path, 'lnse_stressOnly_results.csv'), index=False)
Results_srpc.to_csv(os.path.join(save_results_path, 'srpc_stressOnly_results.csv'), index=False)

# Create evaluation data frame for plant by plant predictions for all algorithms
Results_by_plant_adwin = pd.DataFrame(
    np.concatenate([Plants, res_adwin.reshape(-1, 1), adwin_probs.reshape(-1, 1)], axis=1), columns=columns)
# Results_by_plant_adwin = pd.DataFrame(np.concatenate([Plants, res_adwin.reshape(-1, 1)], axis=1), columns=columns)
Results_by_plant_awe = pd.DataFrame(
    np.concatenate([Plants, res_awe.reshape(-1, 1), awe_probs.reshape(-1, 1)], axis=1), columns=columns)
Results_by_plant_dwm = pd.DataFrame(
    np.concatenate([Plants, res_dwm.reshape(-1, 1), dwm_probs.reshape(-1, 1)], axis=1), columns=columns)
Results_by_plant_lnse = pd.DataFrame(
    np.concatenate([Plants, res_lnse.reshape(-1, 1), lnse_probs.reshape(-1, 1)], axis=1), columns=columns)
Results_by_plant_srpc = pd.DataFrame(
    np.concatenate([Plants, res_srp.reshape(-1, 1), srpc_probs.reshape(-1, 1)], axis=1), columns=columns)

Results_by_plant_adwin['True_Value'] = 0
Results_by_plant_awe['True_Value'] = 0
Results_by_plant_dwm['True_Value'] = 0
Results_by_plant_lnse['True_Value'] = 0
Results_by_plant_srpc['True_Value'] = 0

Results_by_plant_adwin.loc[Results_by_plant_adwin.Treatment == 'A', 'True_Value'] = 1
Results_by_plant_awe.loc[Results_by_plant_awe.Treatment == 'A', 'True_Value'] = 1
Results_by_plant_dwm.loc[Results_by_plant_dwm.Treatment == 'A', 'True_Value'] = 1
Results_by_plant_lnse.loc[Results_by_plant_lnse.Treatment == 'A', 'True_Value'] = 1
Results_by_plant_srpc.loc[Results_by_plant_srpc.Treatment == 'A', 'True_Value'] = 1

# Save all plant by plant predictions for all algorithms
Results_by_plant_adwin.to_csv(os.path.join(save_results_path, 'adwin_results_by_plant.csv'), index=False)
Results_by_plant_awe.to_csv(os.path.join(save_results_path, 'awe_results_by_plant.csv'), index=False)
Results_by_plant_dwm.to_csv(os.path.join(save_results_path, 'dwm_results_by_plant.csv'), index=False)
Results_by_plant_lnse.to_csv(os.path.join(save_results_path, 'lnse_results_by_plant.csv'), index=False)
Results_by_plant_srpc.to_csv(os.path.join(save_results_path, 'srpc_results_by_plant.csv'), index=False)


