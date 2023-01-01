import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ------------------------------------------- Set parameters and load results ---------------------------------------- #
time_steps = 41
steps = range(1, time_steps + 1)

path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results'
cddrl_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\only_stress_labels\train_test_All\CV5\different plants'
# cddrl_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Fertilizer_Stress\Results\CDDRL\stable_5\Entropy_disc_tabu'
lstm_path = os.path.abspath(path + '/Deep_Learning')
competitors_path = os.path.abspath(path + '/Competitors/stress_only_labels/cv5')
expert_path = os.path.abspath(path + '/Expert')
graphs_path = os.path.abspath(path + '/graphs')

cddrl_path_ad = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\only_stress_labels\train_test_AD\Train_test_AD_tabu10_params_stable3_4bins_cv5_window3'
cddrl_path_all = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\only_stress_labels\train_test_All\CV5\different plants'
results_static_CDDRL = pd.read_csv(os.path.join(cddrl_path, 'All_Results.csv'))
results_dynamic_CDDRL = pd.read_csv(os.path.join(cddrl_path, 'All_Results.csv'))
results_dynamic_CDDRL_ad = pd.read_csv(os.path.join(cddrl_path_ad, 'All_Results.csv'))
results_dynamic_CDDRL_all = pd.read_csv(os.path.join(cddrl_path_all, 'All_Results.csv'))
# CDDRL Ensembles
cddrl_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\only_stress_labels\Ensembles'
results_cddrl_bns_all = pd.read_csv(os.path.join(cddrl_path, 'Ensemble BNs/All/All_Results_Ensemble.csv'))
results_cddrl_bns_ad = pd.read_csv(os.path.join(cddrl_path, 'Ensemble BNs/AD/All_Results_Ensemble.csv'))
results_cddrl_methods_all = pd.read_csv(os.path.join(cddrl_path, 'Ensemble methods/All/All_Results.csv'))
results_cddrl_methods_ad = pd.read_csv(os.path.join(cddrl_path, 'Ensemble methods/AD/All_Results.csv'))
results_aucs_methods_all = pd.read_csv(os.path.join(cddrl_path, 'Ensemble methods/All/AUCs_methods.csv'))
results_weights_methods_all = pd.read_csv(os.path.join(cddrl_path, 'Ensemble methods/All/weights.csv'))
results_aucs_methods_ad = pd.read_csv(os.path.join(cddrl_path, 'Ensemble methods/AD/AUCs_methods.csv'))
results_weights_methods_ad = pd.read_csv(os.path.join(cddrl_path, 'Ensemble methods/AD/weights.csv'))

# Competitors
lstm_path = os.path.abspath(path + '/Deep_Learning')
results_lstm = pd.read_csv(os.path.join(lstm_path, 'LSTM_CV_Results_fromstart_stressOnly_auc.csv'))
competitors_path = os.path.abspath(path + '/Competitors/stress_only_labels/cv5')
results_awe = pd.read_csv(os.path.join(competitors_path, 'awe_stressOnly_results_diffPlants.csv'))
results_dwm = pd.read_csv(os.path.join(competitors_path, 'dwm_stressOnly_results_diffPlants.csv'))
results_adwin = pd.read_csv(os.path.join(competitors_path, 'adwin_stressOnly_results_diffPlants.csv'))
results_lnse = pd.read_csv(os.path.join(competitors_path, 'lnse_stressOnly_results_diffPlants.csv'))
results_srpc = pd.read_csv(os.path.join(competitors_path, 'srpc_stressOnly_results_diffPlants.csv'))
results_expert = pd.read_csv(os.path.join(expert_path, 'Aviv_Results_MadeByMe.csv'))

# ------------------------------------------ Comparing CDDRL two versions F1 ----------------------------------------- #
plt.rcParams["font.family"] = "serif"
steps = range(1, time_steps + 1)
plt.plot(steps, results_dynamic_CDDRL['F1'])
plt.plot(steps, results_static_CDDRL['F1'])
plt.xlabel('Day', fontsize=14)
plt.ylabel('F1', fontsize=14)
plt.legend(['CDDRL-dynamic', 'CDDRL-static'], loc='lower right', fontsize=14)
plt.vlines(x=7, ymin=0.0, ymax=1, color='r', linestyles=':')
plt.title('F1 CDDRL Dynamic vs Static 0.3', fontsize=14)
plt.xticks(np.arange(start=1, stop=29, step=2), fontsize=14)
plt.grid()
plt.text(7 - 3, 0.8, "Change point", rotation=360, verticalalignment='center', fontsize=14)

# ------------------------------------------- Real CD Datasets Plotting ---------------------------------------------- #
time_steps = 70
steps = range(1, time_steps + 1)
# CDDRL Ensembles
cddrl_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\NYC_Subway_Traffic\Exits'
results_cddrl = pd.read_csv(os.path.join(cddrl_path, 'CDDRL/Tabu10+Params_18 Vars/All_Results.csv'))
# results_cddrl_bns = pd.read_csv(os.path.join(cddrl_path, 'Ensemble_BNs/All_Results.csv'))
# results_cddrl_methods = pd.read_csv(os.path.join(cddrl_path, 'Ensemble_Methods/All_Results.csv'))

plt.grid()
plt.plot(steps, results_cddrl['W_AUC'], ls='solid')#, color='b')
# plt.plot(steps, results_cddrl_bns['W_AUC'], ls='solid')#, color='r')
# plt.plot(steps, results_cddrl_methods['W_AUC'], ls='solid')#, color='g')
plt.grid()
plt.xlabel('Day', fontsize=14)
plt.ylabel('AUCs', fontsize=14)
plt.legend(['CDDRL', 'CDDRL-BNs', 'CDDRL-Methods'], loc='lower right', fontsize=12)
plt.title('AUCs CDDRL NYC Traffic Exits', fontsize=18)
plt.xticks(np.arange(start=1, stop=time_steps + 1, step=2), fontsize=10)
plt.grid()

# --------------------------------- Old Water Stress Experiment Multiclass Accuracy ---------------------------------- #
time_steps = 17
steps = range(1, time_steps + 1)
# CDDRL Ensembles
cddrl_path = r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Results\Multiclass\CDDRL'
results_cddrl_normal = pd.read_csv(os.path.join(cddrl_path, 'Normal/All_Results.csv'))
results_cddrl_tabu10_params = pd.read_csv(os.path.join(cddrl_path, 'Tabu10+Params/All_Results.csv'))
results_cddrl_robust = pd.read_csv(os.path.join(cddrl_path, 'Robust/All_Results.csv'))
results_cddrl_ucls = pd.read_csv(os.path.join(cddrl_path, 'Dynamic_UCLs/All_Results.csv'))
results_cddrl_cb = pd.read_csv(os.path.join(cddrl_path, 'CB/All_Results.csv'))
results_cddrl_bns = pd.read_csv(os.path.join(cddrl_path, 'Ensemble_BNs/All_Results_Ensemble.csv'))
results_cddrl_methods = pd.read_csv(os.path.join(cddrl_path, 'Ensemble_Methods/weight_method0/All_Results.csv'))

plt.grid()
plt.plot(steps, results_cddrl_normal['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.plot(steps, results_cddrl_tabu10_params['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.plot(steps, results_cddrl_robust['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.plot(steps, results_cddrl_ucls['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.plot(steps, results_cddrl_cb['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.plot(steps, results_cddrl_bns['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.plot(steps, results_cddrl_methods['Multiclass_Accuracy'], ls='solid')#, color='b')
plt.grid()
plt.xlabel('Day', fontsize=14)
plt.ylabel('Multiclass Accuracies', fontsize=14)
# plt.legend(['CDDRL-CB', 'CDDRL-Ensemble_BNs', 'CDDRL-Ensemble_Methods'], loc='lower right', fontsize=14)
# plt.legend(['CDDRL-Normal', 'CDDRL-Tabu10+Params', 'CDDRL-Robust', 'CDDRL-Dynamic_UCLs', 'CDDRL-CB'], loc='lower right', fontsize=14)
plt.legend(['CDDRL-Normal', 'CDDRL-Tabu10+Params', 'CDDRL-Robust', 'CDDRL-Dynamic_UCLs', 'CDDRL-CB',
            'CDDRL-Ensemble_BNs', 'CDDRL-Ensemble_Methods0', 'CDDRL-Ensemble_Methods1'], loc='lower right', fontsize=14)
plt.title('Multiclass Accuracies CDDRL Old Water Stress', fontsize=18)
plt.xticks(np.arange(start=1, stop=time_steps + 1, step=2), fontsize=10)
plt.vlines(x=4, ymin=0.15, ymax=1, color='r', linestyles=':')
plt.text(4 - 0.7, 0.8, "Change point", rotation=360, verticalalignment='center', fontsize=14)
plt.grid()

# --------------------------------------- Comparing AUC to other algorithms ------------------------------------------ #
plt.plot(steps, results_dynamic_CDDRL_ad['AUC'], ls='solid', color='b')
plt.plot(steps, results_cddrl_bns_ad['AUC'], ls='solid', color='r')
plt.plot(steps, results_cddrl_methods_ad['AUC'], ls='solid', color='g')

plt.plot(steps, results_dynamic_CDDRL_all['AUC'], ls='dashed', color='b')
plt.plot(steps, results_cddrl_bns_all['AUC'], ls='dashed', color='r')
plt.plot(steps, results_cddrl_methods_all['AUC'], ls='dashed', color='g')

# plt.plot(steps, results_aucs_methods_ad['AUCs_methods_1'], ls='dashed')#, color='b')
# plt.plot(steps, results_aucs_methods_ad['AUCs_methods_2'], ls='dashed')#, color='g')
# plt.plot(steps, results_aucs_methods_ad['AUCs_methods_3'], ls='dashed')#, color='orange')
# plt.plot(steps, results_aucs_methods_ad['AUCs_methods_4'], ls='dashed')#, color='r')
# plt.plot(steps, results_aucs_methods_ad['AUCs_methods_5'], ls='dashed')#, color='purple')
# #
# plt.plot(steps, results_weights_methods_ad['weights_all1'], ls='dashed', color='b')
# plt.plot(steps, results_weights_methods_ad['weights_all2'], ls='dashed', color='g')
# plt.plot(steps, results_weights_methods_ad['weights_all3'], ls='dashed', color='orange')
# plt.plot(steps, results_weights_methods_ad['weights_all4'], ls='dashed', color='r')
# plt.plot(steps, results_weights_methods_ad['weights_all5'], ls='dashed', color='purple')
# plt.plot(steps, results_lstm['AUC'], ls='dotted')
# plt.plot(steps, results_adwin['AUC'], ls='dashed')
# plt.plot(steps, results_lnse['AUC'], ls=(0, (3, 3, 1, 3)))
# plt.plot(steps, results_srpc['AUC'], ls=(0, (3, 1, 1, 1, 1, 1)))
plt.grid()
plt.xlabel('Day', fontsize=14)
plt.ylabel('AUCs', fontsize=14)
# plt.legend(['CDDRL-Normal', 'CDDRL-BNs', 'CDDRL-Methods'], loc='upper left', fontsize=12)
plt.legend(['CDDRL-Normal-AD', 'CDDRL-BNs-AD', 'CDDRL-Methods-AD', 'CDDRL-Normal-All', 'CDDRL-BNs-All', 'CDDRL-Methods-All'], loc='upper left', fontsize=12)
# plt.legend(['Weights-Params', 'Weights-Tabu', 'Weights-Robust', 'Weights-UCLs', 'Weights-CB'], loc='upper left', fontsize=11)
# plt.legend(['AUC-Params', 'AUC-Tabu', 'AUC-Robust', 'AUC-UCLs', 'AUC-CB', 'Weights-Params', 'Weights-Tabu', 'Weights-Robust', 'Weights-UCLs', 'Weights-CB'], loc='upper left', fontsize=11)
# plt.legend(['CDDRL-Normal', 'CDDRL-BNs', 'CDDRL-Methods', 'LSTM', 'KNN-ADWIN', 'LNSE', 'SRP'], loc='upper left', fontsize=12)
# plt.legend(['CDDRL-Methods', 'Ensemble-Params', 'Ensemble-Tabu', 'Ensemble-Robust', 'Ensemble-UCLs', 'Ensemble-CB'], loc='upper left', fontsize=12)
plt.vlines(x=13, ymin=0.4, ymax=1, color='r', linestyles=':')
plt.title('AUCs CDDRL A D VS. All', fontsize=18)
plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=10)
plt.grid()
plt.text(13 - 2.2, 0.75, "Change point", rotation=360, verticalalignment='center', fontsize=10)
plt.savefig(graphs_path + '/AUC_graph.PNG')

# ----------------------------------- Comparing B_Accuracy to other algorithms --------------------------------------- #
plt.plot(steps, results_dynamic_CDDRL['Balanced_Accuracy'], ls='solid')
plt.plot(steps, results_cddrl_bns_ad['Balanced_Accuracy'], ls='solid')
plt.plot(steps, results_cddrl_methods_ad['Balanced_Accuracy'], ls='solid')
# plt.plot(steps, results_adwin['Balanced_Accuracy'], ls='dotted')
# plt.plot(steps, results_awe['Balanced_Accuracy'], ls='dashed')
# plt.plot(steps, results_dwm['Balanced_Accuracy'], ls='dashdot')
# plt.plot(steps, results_lnse['Balanced_Accuracy'], ls=(0, (3, 3, 1, 3)))
# plt.plot(steps, results_srpc['Balanced_Accuracy'], ls=(0, (3, 1, 1, 1, 1, 1)))
plt.grid()
plt.xlabel('Day', fontsize=14)
plt.ylabel('Balanced Accuracy', fontsize=14)
# plt.legend(['CDDRL-Dynamic', 'LSTM', 'KNN-ADWIN', 'LNSE'], loc='upper left', fontsize=12)
plt.legend(['CDDRL-Normal', 'CDDRL-BNs', 'CDDRL-Methods'], loc='upper left', fontsize=12)
# plt.legend(['CDDRL-Normal', 'CDDRL-BNs', 'CDDRL-Methods', 'KNN-ADWIN', 'AWE', 'DWM', 'LNSE', 'SRP'], loc='upper left', fontsize=12)
plt.vlines(x=13, ymin=0.4, ymax=1, color='r', linestyles=':')
plt.title('Balanced Accuracy CDDRL', fontsize=18)
plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=10)
plt.grid()
plt.text(13 - 2.2, 0.75, "Change point", rotation=360, verticalalignment='center', fontsize=10)
plt.savefig(graphs_path + '/Balanced_Accuracy_graph.PNG')

# --------------------------------------- Comparing DR FA to other algorithms ---------------------------------------- #
plt.plot(steps, results_dynamic_CDDRL['TP_Recall'], ls='solid', color='blue')
plt.plot(steps, results_cddrl_bns_ad['TP_Recall'], ls='solid', color='orange')
plt.plot(steps, results_cddrl_methods_ad['TP_Recall'], ls='solid', color='green')
# plt.plot(steps, results_adwin['TP_Recall'], ls='dotted')
# plt.plot(steps, results_awe['TP_Recall'], ls='dashed')
# plt.plot(steps, results_dwm['TP_Recall'], ls='dashdot')
# plt.plot(steps, results_lnse['TP_Recall'], ls=(0, (3, 3, 1, 3)))
# plt.plot(steps, results_srpc['TP_Recall'], ls=(0, (3, 1, 1, 1, 1, 1)))

# plt.plot(steps, results_dynamic_CDDRL['FA'], ls='dashed', color='blue')
# plt.plot(steps, results_cddrl_bns_ad['FA'], ls='dashed', color='orange')
# plt.plot(steps, results_cddrl_methods_ad['FA'], ls='dashed', color='green')
# plt.plot(steps, results_adwin['FA'], ls='dotted')
# plt.plot(steps, results_awe['FA'], ls='dashed')
# plt.plot(steps, results_dwm['FA'], ls='dashdot')
# plt.plot(steps, results_lnse['FA'], ls=(0, (3, 3, 1, 3)))
# plt.plot(steps, results_srpc['FA'], ls=(0, (3, 1, 1, 1, 1, 1)))

plt.grid()
plt.xlabel('Day', fontsize=14)
plt.ylabel('Detection Rate (%)', fontsize=14)
# plt.legend(['Normal-DR', 'BNs-DR', 'Methods-DR', 'Normal-FA', 'BNs-FA', 'Methods-FA'], loc='upper left', fontsize=12)
# plt.legend(['DR-Normal', 'DR-BNs', 'DR-Methods', 'FA-Normal', 'FA-BNs', 'FA-Methods'], loc='upper left', fontsize=12)
plt.legend(['CDDRL-Normal', 'CDDRL-BNs', 'CDDRL-Methods'], loc='upper left', fontsize=12)
plt.vlines(x=13, ymin=0, ymax=1, color='r', linestyles=':')
plt.title('DR CDDRL', fontsize=18)
plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=10)
plt.grid()
plt.text(13 - 2.2, 0.75, "Change point", rotation=360, verticalalignment='center', fontsize=10)
plt.savefig(graphs_path + '/DR_graph.PNG')

# ------------------------------------------------ Plotting CDDRL AUC ------------------------------------------------ #
cddrl_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\Graphs\Normal_BNs_Results\Results'
AUC_dynamic_CDDRL = pd.read_csv(os.path.join(cddrl_path, 'AUCs.csv'))
plt.plot(steps, AUC_dynamic_CDDRL)
plt.xlabel('Day', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.vlines(x=13, ymin=0.45, ymax=1, color='r', linestyles=':')
plt.title('AUC CDDRL', fontsize=14)
plt.xticks(np.arange(start=1, stop=42, step=2))
plt.grid()
plt.text(13 - 1.9, 0.8, "Change point", rotation=360, verticalalignment='center')

# --------------------------------------------------- DR vs. FA ------------------------------------------------------ #
path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Results\Min_leaves_PlantSeg_Results\Best_Model'
dr_cddrl = pd.read_csv(path + '/Dynamic_thresholds/0.3/DRs.csv', header=None)
fa_cddrl = pd.read_csv(path + '/Dynamic_Thresholds/0.3/FAs.csv', header=None)
plt.plot(steps, dr_cddrl * 100)
plt.plot(steps, fa_cddrl * 100)
plt.xlabel('Day', fontsize=14)
plt.ylabel('Score (%)', fontsize=14)
plt.legend(['DR', 'FA'], loc='upper left', fontsize=12)
plt.vlines(x=13, ymin=0.0, ymax=100, color='r', linestyles=':')
plt.title('Water stress', fontsize=14)
plt.xticks(np.arange(start=1, stop=42, step=2))
plt.grid()
plt.text(13 - 3, 80, "Change point", rotation=360, verticalalignment='center')

# --------------------------------------------- DR CDDRL vs. Expert -------------------------------------------------- #
path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Results\Min_leaves_PlantSeg_Results\Best_Model'
dr_expert = pd.read_csv(path + '/9ME_EBins3_dynamic_0.25/Competitors/Aviv_Results.csv')
plt.plot(steps, results_dynamic_CDDRL['TP_Recall'] * 100)
plt.plot(steps, results_expert['DR-Greenhouse'] * 100, color='g', ls='dotted')
plt.plot(steps, results_expert['DR-Image'] * 100, color='g')
# plt.plot(steps, results_expert['DR-Greenhouse'] * 100, color='g')
plt.xlabel('Day', fontsize=20)
plt.ylabel('Detection Rate (%)', fontsize=20)
plt.legend(['CDDRL', 'Expert-Greenhouse', 'Expert-Image'], loc='lower right', fontsize=20)
# plt.legend(['AgDrift', 'Expert'], loc='lower right', fontsize=20)
plt.vlines(x=13, ymin=0.0, ymax=100, color='r', linestyles=':')
plt.title('DR CDDRL vs. Expert', fontsize=20)
# plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=20)
plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=20)
plt.yticks(np.arange(start=0, stop=110, step=20), fontsize=20)
plt.grid()
plt.text(13 - 3.7, 80, "Change point", rotation=360, verticalalignment='center', fontsize=20)

# ---------------------------- Comparing CDDRL static to dynamic with different windows ------------------------------ #
path = r''
aucs_static = pd.read_csv(path + '/aucs_static_dynamics/aucs_fs_static.csv', header=None)
aucs_dynamic1 = pd.read_csv(path + '/aucs_static_dynamics/aucs_fs_dynamic1.csv', header=None)
aucs_dynamic2 = pd.read_csv(path + '/aucs_static_dynamics/aucs_fs_dynamic2.csv', header=None)
aucs_dynamic3 = pd.read_csv(path + '/aucs_static_dynamics/aucs_fs_dynamic3.csv', header=None)
aucs_dynamic4 = pd.read_csv(path + '/aucs_static_dynamics/aucs_fs_dynamic4.csv', header=None)
steps = range(1, time_steps + 1)
plt.plot(steps, aucs_static)
plt.plot(steps, aucs_dynamic1)
plt.plot(steps, aucs_dynamic2)
plt.plot(steps, aucs_dynamic3)
plt.plot(steps, aucs_dynamic4)
plt.xlabel('Day', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.legend(['CDDRL static', 'CDDRL dynamic, window=1', 'CDDRL dynamic, window=2', 'CDDRL dynamic, window=3',
            'CDDRL dynamic, window=4'], loc='upper left', fontsize=10)
plt.vlines(x=6, ymin=0.5, ymax=1, color='r', linestyles=':')
plt.title('Fertilizer stress', fontsize=14)
plt.xticks(np.arange(start=1, stop=31, step=2))
plt.grid()
plt.text(6 - 3, 0.8, "Change point", rotation=360, verticalalignment='center')

# ------------------------------------ Visualizing Distances and UCLs of CDDRL --------------------------------------- #
path = r'C:\Users\yotam\Desktop\yotam\Phenomix_\EX6_WaterStress\Results\Intersect_Patched_Results\bins_disc'
UCLs_X = pd.read_csv(path + '/static/0.3/UCLs.csv')
UCLs_V = pd.read_csv(path + '/Variational/static/0.3/UCLs.csv')
static_X = pd.read_csv(path + '/static/0.3/Distances.csv')
dynamic_X = pd.read_csv(path + '/dynamic/0.3/Distances.csv')
dynamic_X_old = pd.read_csv(path + '/oldZ_dynamic/0.3/Distances.csv')
static_V = pd.read_csv(path + '/Variational/static/0.3/Distances.csv')
dynamic_V = pd.read_csv(path + '/Variational/dynamic/0.3/Distances.csv')

i = 8
time_steps = 40
steps = range(1, time_steps + 1)
plt.plot(steps, static_X.iloc[:, i])
plt.plot(steps, static_V.iloc[:, i])
plt.plot(steps, dynamic_X.iloc[:, i])
plt.plot(steps, dynamic_X_old.iloc[:, i])
plt.plot(steps, dynamic_V.iloc[:, i])
plt.grid()
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Distance', fontsize=14)
plt.legend(['static_Chi', 'static_Var', 'dynamic_Chi', 'dynamic_Chi_old', 'dynamic_Var'], loc='upper right', fontsize=9)
plt.vlines(x=13, ymin=0, ymax=0.5, color='black', linestyles=':')
plt.hlines(y=UCLs_X.iloc[0, i], xmin=0, xmax=40, color='r', linestyles=':')
plt.hlines(y=UCLs_V.iloc[0, i], xmin=0, xmax=40, color='b', linestyles=':')
plt.title('Distances Comparison Node 9', fontsize=14)
plt.xticks(np.arange(start=1, stop=41, step=2))
plt.grid()
plt.text(9, 0.35, "Change point", rotation=360, verticalalignment='center')
plt.text(0, UCLs_X.iloc[0, i] + 0.008, "UCL_Chi", rotation=360, verticalalignment='center')
plt.text(0, UCLs_V.iloc[0, i] + 0.008, "UCL_Var", rotation=360, verticalalignment='center')

# ---------------------------------- Plot and save all variables by their treatment ---------------------------------- #
data_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Data'
save_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Features'
df_cols = pd.read_csv(data_path + '/ws_disc_bins4.csv')
smooth = pd.read_csv(data_path + '/ws_features_smooth.csv')
cols = df_cols.columns

for i in range(cols.size - 1):
    temp_col = cols[i] + '.PNG'
    plt.grid()
    plot = sns.lineplot(x='Date', y=cols[i], hue='Treatment', data=smooth)
    plt.vlines(x=13, ymin=np.min(smooth[cols[i]]), ymax=np.max(smooth[cols[i]]), color='black', linestyles=':')
    plt.text(8, np.max(smooth[cols[i]]) * 0.65, "Change Point", rotation=360, verticalalignment='center')
    plt.grid()
    plt.xticks(np.arange(start=1, stop=42, step=4))
    plt.title(label=cols[i])
    plt.grid()
    fig = plot.get_figure()
    plt.grid()
    fig.savefig(save_path + '/' + temp_col)
    plot.clear()
    print("Finished plotting feature number", i, ": ", cols[i])


save_path = r'D:\yotam\לימודים\מיתר\מחקר\תזה'
smooth = pd.read_csv(r'D:\yotam\CD_Project\Results\Alarm_New\Alarm_Accuracies.csv')
cols = smooth.columns[2]
for i in range(1):
    temp_col = cols[i] + '.PNG'
    plt.grid()
    plot = sns.lineplot(x='Time Step', y=cols, hue='Algorithm', data=smooth, err_style='bars')
    plt.legend(['Initial BN', 'CDDRL-Original', 'CDDRL-Dynamic UCLs', 'CDDRL-Tabu', 'CDDRL-Robust', 'CDDRL-Params'], loc='upper left', fontsize=10)
    plt.vlines(x=1000, ymin=0, ymax=100, color='black', linestyles='solid')
    plt.vlines(x=1100, ymin=0, ymax=100, color='black', linestyles=':')
    plt.text(955, 90, "Change Point", rotation=360, verticalalignment='center')
    plt.grid()
    plt.xticks(np.arange(start=0, stop=1600, step=100))
    plt.title(label="Alarm CDDRL Versions Accuracy Results")
    plt.grid()
    fig = plot.get_figure()
    plt.grid()
    fig.savefig(save_path + '/Alarm_Accuracies_Graph')
    # plot.clear()
    print("Finished plotting feature number", i, ": ", cols)

# --------------------------------------------- Plot A vs D temperatures --------------------------------------------- #
data_path = r''
save_path = r''
data = pd.read_csv(data_path + '/df_all_no_smooth_NoBadPlants.csv')
data = pd.DataFrame(data)
data['Plant'] = [data.at[i, 'Treatment'] + str(data.at[i, 'Num']) for i in range(0, np.size(data, 0))]

time_steps = 41
steps = range(1, time_steps + 1)
for i in range(1, 49):
    try:
        A_data = data[data['Plant'] == 'A' + str(i)]
        D_data = data[data['Plant'] == 'D' + str(i)]
        plt.figure(figsize=(25, 25))
        plt.plot(steps, A_data['Avg_temp'])
        plt.plot(steps, D_data['Avg_temp'])
        plt.plot(steps, A_data['outside_temps'])
        plt.plot(steps, D_data['outside_temps'])
        plt.plot(steps, A_data['norm_avg_temp'])
        plt.plot(steps, D_data['norm_avg_temp'])
        plt.xlabel('Day', fontsize=14)
        plt.ylabel('Temps', fontsize=14)
        plt.legend(['Avg_temp_A' + str(i), 'Avg_temp_D' + str(i), 'Outside_temp_A' + str(i), 'Outside_temp_D' + str(i),
                    'Norm_avg_temp_A' + str(i), 'Norm_avg_temp_D' + str(i)], loc='center right', fontsize=9)
        plt.title('A' + str(i) + ' D' + str(i) + ' Avg Temperatures', fontsize=14)
        plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=10)
        plt.grid()
        plt.text(10.5, 20, "Change point", rotation=360, verticalalignment='center')
        plt.vlines(x=13, ymin=-8, ymax=35, color='r', linestyles=':')
        save_name = 'A' + str(i) + ' D' + str(i) + ' Avg Temps'
        plt.savefig(save_path + '/' + save_name)
    except:
        print("Error in plants: ", i)


# -------------------------------------------- Plot BNs found by the CDDRL ------------------------------------------- #
# def markov_blanket(adjacency_matrix, node):
#     parents = np.where(adjacency_matrix[:, node] > 0)[0].tolist()
#     children = np.where(adjacency_matrix[node, :] > 0)[0].tolist()
#     parents_of_children = [np.where(adjacency_matrix[:, children] > 0)[0].tolist() for children in children]
#     parents_of_children = [item for sublist in parents_of_children for item in sublist]
#     MB = [x for x in np.unique(parents + children + parents_of_children) if x != node]
#     return MB
#
#
# def color_map_bn(adjacency_matrix, target_node, colors=None):
#     if colors is None:
#         colors = ['deepskyblue', 'lime', 'yellow']
#     mb = markov_blanket(adjacency_matrix, target_node)
#     color_map = []
#     for i in range(adjacency_matrix.shape[0]):
#         if i == target_node:
#             color_map.append(colors[0])
#         elif not mb is None:
#             if i in mb:
#                 color_map.append(colors[1])
#             else:
#                 color_map.append(colors[2])
#         else:
#             color_map.append(colors[2])
#     return color_map


def markov_blanket(adjacency_matrix, node):
    parents = np.where(adjacency_matrix[:, node] > 0)[0].tolist()
    children = np.where(adjacency_matrix[node, :] > 0)[0].tolist()
    parents_of_children = [np.where(adjacency_matrix[:, children] > 0)[0].tolist() for children in children]
    parents_of_children = [item for sublist in parents_of_children for item in sublist]
    pc = [x for x in np.unique(parents + children) if x != node]
    spouses = [x for x in np.unique(parents_of_children) if x != node]
    return [pc, spouses]


def color_map_bn(adjacency_matrix, target_node, colors=None):
    if colors is None:
        colors = ['deepskyblue', 'limegreen', 'limegreen', 'yellow']
    [pc, spouses] = markov_blanket(adjacency_matrix, target_node)
    color_map = []
    for i in range(adjacency_matrix.shape[0]):
        if i == target_node:
            color_map.append(colors[0])
        elif not pc is None:
            if i in pc:
                color_map.append(colors[1])
            elif i in spouses:
                color_map.append(colors[2])
            else:
                color_map.append(colors[3])
        else:
            color_map.append(colors[3])
    return color_map


def plot_bns(path_original, var_names, days_num):
    path_bns = path_original + '/BNs'
    path_save = path_original + '/Plot_BNs'
    df = pd.read_csv(r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Data\ws_disc.csv')
    ids = {ind: var_names[col] for ind, col in enumerate(df.columns)}
    # ids = {ind: col for ind, col in enumerate(df.columns)}

    for ii in range(1, days_num + 1):
        plt.figure(figsize=(10, 10))
        dag_i = np.array(pd.read_csv(path_bns + r'\BN' + str(ii) + '.csv', header=None))
        target_node = dag_i.shape[0] - 1
        color_map = color_map_bn(dag_i, target_node=target_node)
        g = nx.from_numpy_matrix(dag_i, create_using=nx.DiGraph)
        pos = nx.fruchterman_reingold_layout(g, k=10.0, iterations=20)
        nx.draw(g, node_color=color_map, node_size=3000, pos=pos, labels=ids, with_labels=True, arrowsize=14)
        ax = plt.gca()  # to get the current axis
        ax.collections[0].set_edgecolor("black")
        ax.margins(0.08)
        plt.tight_layout()
        plt.savefig(path_save + r'\BN' + str(ii) + '.png')
        plt.close()


# -------------------------------- Set var names and number of days and plot the BNs --------------------------------- #
days = 42
path = r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\only_stress_labels\Ensembles\Ensemble methods\All\BNs\Combined\majority_vote'
col_names_to_short_cuts = {'Mean Size': 'S1', 'Max size': 'S2', 'Std size': 'S3',
                           'Mean Mean green': 'H1', 'Mean std green': 'H2', 'std Mean green': 'H3',
                           'Mean egi': 'E1', 'std egi': 'E2', 'Mean vari': 'V1', 'std vari': 'V2',
                           'Mean ndi': 'N1', 'std ndi': 'N2', 'class': 'T', 'Depth': 'D', 'Max angle': 'A',
                           'Mean perimeter': 'P1', 'Max perimeter': 'P2', 'Std perimeter': 'P3', 'Max size diff': 'S4'}

col_names_to_short_cuts2 = {'Max size': 'M_Size', 'Num_of_Leaves': 'N_Leaves', 'std std green': 'st_green',
                            'Mean Mean green': 'M_M_green', 'Norm_avg_2ndmin_leaf_temp': '2min_temp',
                            'Mean green 2ndmin leaf': '2nd_green', 'Std_Depth_All_Leaves': 'ST_Depth',
                            'Norm_all_plant_avg_third_temp': 'temp3', 'Mean ndi_x': 'NDI', 'class': 'Treat'}

col_names_to_short_WS_Old = {'Max perimeter': 'MP', 'Max size': 'MS', 'Mean green 2ndmin leaf': 'MG2',
                             'Mean Mean green': 'MMG', 'Mean std green': 'MSG', 'Mean vari 2ndmin leaf': 'MV2',
                             'Mean vari': 'MV', 'Norm_all_plant_avg_third_temp': 't3', 'Norm_all_plant_min_temp': '1t',
                             'Norm_avg_third_2ndmin_leaf_temp': '2t3', 'Num_of_Leaves': 'NL',
                             'range vari min leaf': 'RV1', 'Std green 2ndmin leaf': 'ST2G', 'Std perimeter': 'STP',
                             'Std size': 'STS',	'std std green': 'STSG', 'class': 'T'}

col_names_to_short_WS_New = {'Max_size': 'MS', 'Avg_hue': 'AH', 'Avg_vari': 'AV',
                             'Std_std_hue': 'SSH', 'Avg_plant_angles': 'AA', 'Avg_ndi': 'AN',
                             'Avg_perimeter': 'AP', 'Max_perimeter': 'MP', 'Avg_plant_third_temp': 'ATT',
                             'Num_of_leaves': 'NL', 'class': 'T'}

col_names_to_short_FS = {'Mean std green': 'MSG', 'Max angle': 'MA', 'Mean Mean green': 'MMG',
                         'std vari': 'STV', 'Mean perimeter': 'MP', 'Std perimeter': 'STP',
                         'std ndi': 'STN', 'Mean Size': 'MS', 'Mean egi': 'ME',
                         'std Mean green': 'SMG', 'class': 'T'}

col_names_to_short_alarm = {'Max perimeter': 'MP', 'Max size': 'MS', 'Mean green 2ndmin leaf': 'MG2',
                            'Mean Mean green': 'MMG', 'Mean std green': 'MSG', 'Mean vari 2ndmin leaf': 'MV2',
                            'Mean vari': 'MV', 'Norm_all_plant_avg_third_temp': 't3', 'Norm_all_plant_min_temp': '1t',
                            'Norm_avg_third_2ndmin_leaf_temp': '2t3', 'Num_of_Leaves': 'NL',
                            'range vari min leaf': 'RV1', 'Std green 2ndmin leaf': 'ST2G', 'Std perimeter': 'STP',
                            'Std size': 'STS',	'std std green': 'STSG', 'class': 'T'}

col_names_to_short_cuts_AWS = {'day': 'Day', 'hour': 'Hour', 'minute': 'Minute', 'Instance_type': 'Inst_Type',
                               'Op_system': 'Op_Sys', 'Region': 'Region', 'Price': 'Price'}

col_names_to_short_cuts_Electricity = {'day': 'Day', 'hour': 'Hour', 'nswprice': 'NSW_price', 'nswdemand': 'NSW_demand',
                                       'vicprice': 'Vic_price', 'vicdemand': 'Vic_demand', 'transfer': 'Transfer',
                                       'class': 'Price'}

col_names_to_short_cuts_Instagram = {'Length of Username': 'Name_Len', 'Sex': 'Sex', 'Is Professional Account': 'Prof',
                                     'Number of Followers': 'Followers', 'Is Joined Recently': 'New',
                                     'Is Private': 'Private', 'Is Verified': 'Verified', 'Number of Posts': 'Posts',
                                     'Number of Mutual Followers': 'Mutual_Fol', 'Mean Post Likes': 'Likes',
                                     'Percentage of Following': 'Following', 'Is Business Account': 'Business',
                                     'Number of Video Posts': 'Videos', 'Length of Biography': 'Bio',
                                     'Class': 'Will_Follow'}

col_names_to_short_cuts_NYC_Traffic = {'Unique ID': 'Station_ID', 'Month': 'Month', 'Day': 'Day',
                                       'Hour_Interval': 'Hour', 'Line': 'Line', 'North Direction Label': 'North',
                                       'South Direction Label': 'South', 'Division': 'Division',
                                       'Structure': 'Structure', 'Borough': 'Borough', 'Neighborhood': 'Neighbor',
                                       'Homeownership rate': 'Home_Owner', 'Properties entering REO, 1-4 family': '1-4Family',
                                       'Refinance loan rate (per 1,000 properties)': 'Loan', 'Serious crime rate (per 1,000 residents)': 'Crime',
                                       'Serious crime rate, property (per 1,000 residents)': 'Crime2', 'Entries': 'Entries',
                                       'Exits': 'Exits'}

col_names_to_short_cuts_Old_Water_Stress = {'WaterLeft': 'Water', 'SPAD': 'SPAD', 'size': 'Size', '3rd_temp': 'Temp',
                                            'class': 'Treatment'}

plot_bns(path, col_names_to_short_cuts_Old_Water_Stress, days)


# ----------------------------------------------- Plot a single BN --------------------------------------------------- #
col_names = {'HISTORY': 'HIST', 'CVP': 'CVP', 'PCWP': 'PCWP', 'HYPOVOLEMIA': 'HYPV', 'LVEDVOLUME': 'LVV',
             'LVFAILURE': 'LVF', 'STROKEVOLUME': 'STV', 'ERRLOWOUTPUT': 'ERLO', 'HRBP': 'HRBP', 'HREKG': 'HRK',
             'ERRCAUTER': 'ERC', 'HRSAT': 'HRS', 'INSUFFANESTH': 'ISA', 'ANAPHYLAXIS': 'ANP', 'TPR': 'TPR',
             'EXPCO2': 'EC2', 'KINKEDTUBE': 'KDT', 'MINVOL': 'MINV', 'FIO2': 'FI2', 'PVSAT': 'PVS', 'SAO2': 'SA2',
             'PAP': 'PAP', 'PULMEMBOLUS': 'PLB', 'SHUNT': 'SHT', 'INTUBATION': 'ITB', 'PRESS': 'PRS',
             'DISCONNECT': 'DSC', 'MINVOLSET': 'MINS', 'VENTMACH': 'VTM', 'VENTTUBE': 'VTT', 'VENTLUNG': 'VTL',
             'VENTALV': 'VTA',	'ARTCO2': 'AC2', 'CATECHOL': 'CTC', 'HR': 'HR',	'CO': 'CO', 'BP': 'BP'}

dag_path = r'C:\Users\yotam\Desktop\yotam\CD_Project\DBs\Alarm_New'
data_path = r'C:\Users\yotam\Desktop\yotam\CD_Project\DBs\Alarm_New\dfs'
dag = np.array(pd.read_csv(dag_path + '/bnpost_dag.csv', header=None))
data = pd.read_csv(data_path + '/1.csv')
del data['class']
target_node = []
ids = {ind: col_names[col] for ind, col in enumerate(data.columns)}
color_map = color_map_bn(dag, target_node=target_node)
g = nx.from_numpy_matrix(dag, create_using=nx.DiGraph)
pos = nx.fruchterman_reingold_layout(g, k=10.0, iterations=20)
nx.draw(g, node_color=color_map, node_size=1000, pos=pos, labels=ids, with_labels=True, arrowsize=14)
ax = plt.gca()  # to get the current axis
ax.collections[0].set_edgecolor("black")
plt.show()
# name_to_save = 'BN_PRE'
# plt.savefig(dag_path + '/' + name_to_save + '.png')


# ---------------------------------------------- Plot windows sizes -------------------------------------------------- #
data_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\only_stress_labels\‏‏train_test_try\final window'
save_path = r'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\graphs\Window graphs'

auc2 = pd.read_csv(data_path + '/AUCs2.csv')
auc3 = pd.read_csv(data_path + '/AUCs3.csv')
auc4 = pd.read_csv(data_path + '/AUCs4.csv')
auc5 = pd.read_csv(data_path + '/AUCs5.csv')
auc6 = pd.read_csv(data_path + '/AUCs6.csv')
b_acc2 = pd.read_csv(data_path + '/All_Results2.csv')
b_acc3 = pd.read_csv(data_path + '/All_Results3.csv')
b_acc4 = pd.read_csv(data_path + '/All_Results4.csv')
b_acc5 = pd.read_csv(data_path + '/All_Results5.csv')
b_acc6 = pd.read_csv(data_path + '/All_Results6.csv')

# plt.plot(steps, b_acc2['Balanced_Accuracy'], ls='solid')
# plt.plot(steps, b_acc3['Balanced_Accuracy'], ls='dotted')
# plt.plot(steps, b_acc4['Balanced_Accuracy'], ls='dotted')
# plt.plot(steps, b_acc5['Balanced_Accuracy'], ls='dashed')
# plt.plot(steps, b_acc6['Balanced_Accuracy'], ls='dashdot')
plt.plot(steps, auc2, ls='solid')
plt.plot(steps, auc3, ls='dotted')
plt.plot(steps, auc4, ls='dotted')
plt.plot(steps, auc5, ls='dashed')
plt.plot(steps, auc6, ls='dashdot')
plt.xlabel('Day', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.legend(['Window=2', 'Window=3', 'Window=4', 'Window=5', 'Window=6'], loc='upper left', fontsize=12)
plt.vlines(x=13, ymin=0.35, ymax=1, color='r', linestyles=':')
plt.title('AUC of Changing Window Size', fontsize=18)
plt.xticks(np.arange(start=1, stop=42, step=2), fontsize=10)
plt.grid()
plt.text(13 - 2.4, 0.75, "Change point", rotation=360, verticalalignment='center', fontsize=10)
plt.savefig(save_path + '/AUC_window_graph.PNG')


# ---------------------------------------------- Plot Ensemble Methods BNs ------------------------------------------- #

method = 'majority_vote'   # Options: 'every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold'
num_of_bns = 40
weight_threshold = 0.6

path = r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\only_stress_labels\Ensembles\Ensemble methods\All\BNs\Combined'
save_path = path + '/' + method + '/plot_BNs/'

# col_names = ['Water', 'SPAD', 'Size', 'Temp', 'Treatment']
# col_names = ['day', 'hour', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer', 'class']
col_names = ["Max_size", "Avg_hue", "Avg_vari", "Std_std_hue", "Avg_plant_angles", "Avg_ndi", "Avg_perimeter",
             "Max_perimeter", "Avg_plant_third_temp", "Num_of_leaves", "Treatment"]

for bn in range(1, num_of_bns + 1):
    plt.figure(figsize=(12, 12))
    G = nx.DiGraph()
    # add nodes of the graph
    for col in range(len(col_names)):
        if col_names[col] == 'class':
            G.add_node(col_names[col])
        else:
            G.add_node(col_names[col])
    # positions for all nodes - seed for reproducibility
    pos = nx.spring_layout(G, seed=7)

    # edges
    if method == 'every_edge' or method == 'majority_vote':
        graph = pd.read_csv(path + '/' + method + '/BN' + str(bn) + '.csv', header=None)
        color_map = color_map_bn(np.array(graph), target_node=len(graph) - 1)
        for row in range(len(col_names)):
            for col in range(len(col_names)):
                if graph.iloc[row, col] > 0:
                    G.add_edge(col_names[row], col_names[col])

    if method == 'every_edge_with_weight':
        graph = pd.read_csv(path + '/' + method + '/BN' + str(bn) + '.csv', header=None)
        color_map = color_map_bn(np.array(graph), target_node=len(graph) - 1)
        for row in range(len(col_names)):
            for col in range(len(col_names)):
                if graph.iloc[row, col] > 0:
                    G.add_edge(col_names[row], col_names[col], weight=np.round(graph.iloc[row, col], 2))

    if method == 'every_edge_over_weight_threshold':
        method = 'every_edge_with_weight'
        graph = pd.read_csv(path + '/' + method + '/BN' + str(bn) + '.csv', header=None)
        method = 'every_edge_over_weight_threshold'
        graph = np.where(graph > weight_threshold, graph, 0)
        color_map = color_map_bn(np.array(graph), target_node=len(graph) - 1)
        for row in range(len(col_names)):
            for col in range(len(col_names)):
                if graph[row, col] >= weight_threshold:
                    G.add_edge(col_names[row], col_names[col], weight=np.round(graph[row, col], 2))

    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    # Draw network
    options = {
        "font_size": 14,
        "font_family": "arial-black",
        "node_size": 5500,
        "node_color": color_map,
        "arrows": True,
        "arrowsize": 16,
        "width": 2,
        "with_labels": False,
    }
    nx.draw_networkx(G, pos, **options)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path + 'BN' + str(bn) + '.png')
    plt.close()

# G.add_node("Max_size")
# G.add_node("Avg_hue")
# G.add_node("Avg_vari")
# G.add_node("Std_std_hue")
# G.add_node("Avg_plant_angles")
# G.add_node("Avg_ndi")
# G.add_node("Avg_perimeter")
# G.add_node("Max_perimeter")
# G.add_node("Avg_plant_third_temp")
# G.add_node("Num_of_leaves")
# G.add_node("class")

# G.add_node("MS")
# G.add_node("AH")
# G.add_node("AV")
# G.add_node("SSH")
# G.add_node("AA")
# G.add_node("AN")
# G.add_node("AP")
# G.add_node("MP")
# G.add_node("ATT")
# G.add_node("NL")
# G.add_node("T")

# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]
#
# pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
#
# # nodes
# nx.draw_networkx_nodes(G, pos, node_size=1500)
#
# # edges
# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
#
# # node labels
# nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")
# # edge weight labels
# edge_labels = nx.get_edge_attributes(G, "weight")
# nx.draw_networkx_edge_labels(G, pos, edge_labels)
#
# ax = plt.gca()
# ax.margins(0.08)
# plt.axis("off")
# plt.tight_layout()
# plt.show()


seed = 13648  # Seed random number generators for reproducibility
G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
pos = nx.spring_layout(G, seed=seed)

# node_sizes = [3 + 10 * i for i in range(len(G))]
# M = G.number_of_edges()
# edge_colors = range(2, M + 2)
# edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
# cmap = plt.cm.plasma

nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color="indigo")
edges = nx.draw_networkx_edges(G, pos, node_size=500, arrowstyle="->", arrowsize=10, edge_color="black", width=2)
# set alpha value for each edge
# for i in range(M):
#     edges[i].set_alpha(edge_alphas[i])
#
# pc = mpl.collections.PatchCollection(edges, cmap=cmap)
# pc.set_array(edge_colors)
# plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()



