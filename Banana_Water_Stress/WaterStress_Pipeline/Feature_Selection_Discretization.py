from random import gauss
import matplotlib.pyplot as plt
from supervised_cd_utils import Discritization, Evaluate_competitors
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
from scipy.stats import iqr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from BorutaShap import BorutaShap
import xgboost as xgb
from sklearn.linear_model import Lasso
import seaborn as sns
import ppscore as pps

warnings.filterwarnings("ignore")

# ------------------------------------------- Load data and prepare it ----------------------------------------------- #

path = r'C:\Users\User\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
save_path = r'C:\Users\User\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features\Features_Importance'
data = pd.read_csv(path + '/All_features_smoothed.csv')

data = data.sort_values(by=['Date', 'Treatment', 'Num'])
y = np.array([1 if x[0] == 'D' else 0 for x in data['Treatment'].tolist()])
X = data.drop(['Treatment', 'Date', 'Num'], axis=1)
col_names = X.columns

jump = 188
change_point = 13
break_points = np.arange(start=0, stop=y.shape[0] - jump, step=jump)

# ----------------------------------- Feature Selection by RF Feature Importance ------------------------------------- #

num_of_experiments = 50
importance = np.zeros(X.shape[1])
non_importance = np.zeros(X.shape[1])
# Go through every two days (two jumps) and fit a rf model num of experiments times, sum the features importance
for t in tqdm(break_points):
    X_temp = X[t:t + jump * 2]
    y_temp = y[t:t + jump * 2]
    for r in tqdm(range(num_of_experiments)):
        rf = RandomForestClassifier(random_state=r)
        rf.fit(X_temp, y_temp)
        if t < jump * change_point:  # We don't want the features to be important before change point
            non_importance += rf.feature_importances_
        else:
            importance += rf.feature_importances_

# Get average importance and non importance per run
importance /= (num_of_experiments * len(break_points))
non_importance /= (num_of_experiments * change_point)
# Get the overall importance - normalize by subtracting the non importance of each feature
overall_importance = importance - non_importance

# Get sorted overall importance features
overall_columns_ranks = {col: score for col, score in zip(col_names, overall_importance)}
sorted_overall_columns_ranks = dict(sorted(overall_columns_ranks.items(), key=lambda x: x[1], reverse=True))
# Get sorted importance only features
importance_columns_ranks = {col: score for col, score in zip(col_names, importance)}
sorted_importance_columns_ranks = dict(sorted(importance_columns_ranks.items(), key=lambda x: x[1], reverse=True))

# Create a dataframe with the n chosen most important features
n_features_to_choose = 10
chosen_columns = [item for ind, item in enumerate(sorted_overall_columns_ranks) if ind < n_features_to_choose]
data_chosen = data[chosen_columns]

# Save sorted feature importance (importance only and overall)
importance_df = pd.DataFrame()
importance_df['Feature'] = sorted_importance_columns_ranks.keys()
importance_df['Importance'] = sorted_importance_columns_ranks.values()
importance_df.to_csv(save_path + '/rf_importance.csv')

overall_importance_df = pd.DataFrame()
overall_importance_df['Feature'] = sorted_overall_columns_ranks.keys()
overall_importance_df['Importance'] = sorted_overall_columns_ranks.values()
overall_importance_df.to_csv(save_path + '/rf_overall importance.csv')


# ------------------------------------------ Feature Selection by Boruta --------------------------------------------- #

def boruta_selection(model, x, y):
    # Category encoder for the dataset x
    for c in range(x.shape[1]):
        if x.dtypes[c] == 'object':
            x[x.columns[c]] = x[x.columns[c]].astype('category')
            x[x.columns[c]] = x[x.columns[c]].cat.codes

    feature_names = np.array(x.columns)

    # Run Boruta using XGBoost
    if model == 'xgb':
        xgb_model = xgb.XGBClassifier()
        feat_selector_xgb = BorutaPy(xgb_model, n_estimators='auto', verbose=2, random_state=42)
        feat_selector_xgb.fit(np.array(x), y)
        support_xgb = feat_selector_xgb.support_
        ranking_xgb = feat_selector_xgb.ranking_

        accepted_vars = [col for ind, col in enumerate(x.columns) if ranking_xgb[ind] == 1]
        tentative_vars = [col for ind, col in enumerate(x.columns) if ranking_xgb[ind] == 2]
        rejected_vars = [col for ind, col in enumerate(x.columns) if ranking_xgb[ind] > 2]

        # ranks = list(zip(feature_names, ranking_xgb))
        ranks = ranking_xgb
        x_filtered = x[accepted_vars]

    elif model == 'rf':
        # Run Boruta Random Forest
        rf_model = RandomForestClassifier()
        feat_selector_rf = BorutaPy(rf_model, n_estimators='auto', verbose=2, random_state=42)
        feat_selector_rf.fit(np.array(x), y)
        support_rf = feat_selector_rf.support_
        ranking_rf = feat_selector_rf.ranking_

        accepted_vars = [col for ind, col in enumerate(x.columns) if ranking_rf[ind] == 1]
        tentative_vars = [col for ind, col in enumerate(x.columns) if ranking_rf[ind] == 2]
        rejected_vars = [col for ind, col in enumerate(x.columns) if ranking_rf[ind] > 2]

        # ranks = list(zip(feature_names, ranking_rf))
        ranks = ranking_rf
        x_filtered = x[accepted_vars]

    return x_filtered, accepted_vars, tentative_vars, rejected_vars, ranks


def save_boruta_results(df, accept, tent, reject, rank, s_path, model):
    accept = pd.DataFrame(accept)
    tent = pd.DataFrame(tent)
    reject = pd.DataFrame(reject)
    rank = pd.DataFrame(rank)
    rank = rank.sort_values(by=1)
    results = pd.concat((accept, tent, reject, rank), axis=1)
    results.columns = ['Accepted', 'Tentative', 'Rejected', 'Variable', 'Ranking']
    if model == 'rf':
        # df.to_csv(s_path + '/df_boruta_rf.csv', index=False)
        results.to_csv(s_path + '/importance_boruta_rf.csv', index=False)
    else:
        # df.to_csv(s_path + '/df_boruta_xgb.csv', index=False)
        results.to_csv(s_path + '/importance_boruta_xgb.csv', index=False)
    return


# Run boruta on every two jumps and aggregate features' ranking
overall_ranking = np.zeros(len(X.columns))

for t in tqdm(break_points):
    X_temp = X[t:t + jump * 2]
    y_temp = y[t:t + jump * 2]
    [df_boruta, accepted, tentative, rejected, ranking] = boruta_selection('rf', X_temp, y_temp)
    if t < jump * change_point:  # We don't want the features to be important before change point
        ranking = np.where(ranking == 1, -2, ranking)
        ranking = np.where(ranking == 2, -1, ranking)
        ranking = np.where(ranking >= 3, 0, ranking)
    else:
        ranking = np.where(ranking == 1, 2, ranking)
        ranking = np.where(ranking == 2, 1, ranking)
        ranking = np.where(ranking >= 3, 0, ranking)
        overall_ranking = np.add(overall_ranking, ranking)    # Use if calculating importances only after change point

    overall_ranking = np.add(overall_ranking, ranking)

features_ranking = pd.DataFrame()
features_ranking['Feature'] = X.columns
features_ranking['Overall Ranking'] = overall_ranking / len(break_points)
features_ranking = features_ranking.sort_values(by='Overall Ranking', ascending=False)
features_ranking.to_csv(save_path + '/boruta_importance_norm_rf.csv', index=False)

# save_boruta_results(df_boruta, accepted, tentative, rejected, overall_ranking, save_path, 'rf')


# ---------------------------------------- Feature Selection by Boruta-Shap ------------------------------------------ #

def boruta_shap_selection(x, y):
    # Category encoder for the dataset x
    for c in range(x.shape[1]):
        if x.dtypes[c] == 'object':
            x[x.columns[c]] = x[x.columns[c]].astype('category')
            x[x.columns[c]] = x[x.columns[c]].cat.codes

    # Run Boruta-Shap Random Forest
    feat_selector_rf = BorutaShap(importance_measure='shap', classification=True)
    feat_selector_rf.fit(x, y, n_trials=100, sample=False, verbose=True)
    accept = feat_selector_rf.accepted
    tent = feat_selector_rf.tentative
    reject = feat_selector_rf.rejected

    return accept, tent, reject


# Run boruta on every two jumps and aggregate features' ranking
boruta_shap_ranking = pd.DataFrame(columns=X.columns)
boruta_shap_ranking.loc[len(boruta_shap_ranking)] = 0

for t in tqdm(break_points):
    X_temp = X[t:t + jump * 2]
    y_temp = y[t:t + jump * 2]
    [accepted, tentative, rejected] = boruta_shap_selection(X_temp, y_temp)
    if t < jump * change_point:  # We don't want the features to be important before change point
        continue
    else:
        boruta_shap_ranking[accepted] += 2
        boruta_shap_ranking[tentative] += 1

boruta_shap_ranking.to_csv(save_path + '/boruta_shap_importance_norm_rf.csv', index=False)


# ------------------------------- Feature Selection by Predictive Power Score (PPS) ---------------------------------- #

temp_data = data.copy()
temp_data = temp_data[jump * change_point:]
del temp_data['Num']
del temp_data['Date']
pps_matrix = pps.matrix(temp_data)
pps_matrix.to_csv(save_path + '/PPS_Correlation.csv', index=False)


# ------------------------------------------ Feature Selection by ANOVA ---------------------------------------------- #

# Set features dictionary with the number of times each was selected
Features_score_dict = dict()
for i in range(len(X.columns)):
    Features_score_dict[X.columns[i]] = 0

# Run through the days and select top k features for each two jumps
for t in tqdm(break_points):
    if t >= jump * change_point:
        X_temp = X[t:t + jump * 2]
        y_temp = y[t:t + jump * 2]
        # Define feature selection
        fs = SelectKBest(score_func=f_classif, k=20)
        # Apply feature selection
        fs.fit(X_temp, y_temp)
        # Get indices of chosen columns
        cols = fs.get_support(indices=True)
        # Add a point to each selected feature
        for j in range(len(cols)):
            Features_score_dict[X.columns[cols[j]]] += 1

Features_scores = pd.DataFrame()
Features_scores['Feature'] = X.columns
Features_scores['ANOVA_Score'] = Features_score_dict.values()
Features_scores = Features_scores.sort_values(by='ANOVA_Score', ascending=False)
Features_scores.to_csv(save_path + '/ANOVA_importance.csv', index=False)


# ------------------------------------------ Feature Selection by LASSO ---------------------------------------------- #

# Set features dictionary with the number of times each was selected
Features_score_dict = dict()
for i in range(len(X.columns)):
    Features_score_dict[X.columns[i]] = 0

for t in tqdm(break_points):
    if t >= jump * change_point:
        X_temp = X[t:t + jump * 2]
        y_temp = y[t:t + jump * 2]
        model = Lasso(alpha=0.8)
        results = model.fit(X_temp, y_temp)
        important_features = np.where(results.coef_ != 0)
        for j in range(len(important_features[0])):
            Features_score_dict[X.columns[important_features[0][j]]] += 1

Lasso_scores = pd.DataFrame()
Lasso_scores['Feature'] = X.columns
Lasso_scores['LASSO_Score'] = Features_score_dict.values()
Lasso_scores = Lasso_scores.sort_values(by='LASSO_Score', ascending=False)
Lasso_scores.to_csv(save_path + '/LASSO_importance.csv', index=False)


# ------------------------------------------------- Correlation ------------------------------------------------------ #

def get_feature_correlation(df, top_n=None, corr_method='pearson',
                            remove_duplicates=True, remove_self_correlations=True):
    """
    Compute the feature correlation and sort feature pairs based on their correlation

    :param df: The dataframe with the predictor variables
    :type df: pandas.core.frame.DataFrame
    :param top_n: Top N feature pairs to be reported (if None, all of the pairs will be returned)
    :param corr_method: Correlation compuation method
    :type corr_method: str
    :param remove_duplicates: Indicates whether duplicate features must be removed
    :type remove_duplicates: bool
    :param remove_self_correlations: Indicates whether self correlations will be removed
    :type remove_self_correlations: bool

    :return: pandas.core.frame.DataFrame
    """
    corr_matrix_abs = df.corr(method=corr_method).abs()
    corr_matrix_abs_us = corr_matrix_abs.unstack()
    sorted_correlated_features = corr_matrix_abs_us \
        .sort_values(kind="quicksort", ascending=False) \
        .reset_index()

    # Remove comparisons of the same feature
    if remove_self_correlations:
        sorted_correlated_features = sorted_correlated_features[
            (sorted_correlated_features.level_0 != sorted_correlated_features.level_1)]
    # Remove duplicates
    if remove_duplicates:
        sorted_correlated_features = sorted_correlated_features.iloc[:-2:2]
    # Create meaningful names for the columns
    sorted_correlated_features.columns = ['Feature 1', 'Feature 2', 'Correlation (abs)']
    if top_n:
        return sorted_correlated_features[:top_n]
    return sorted_correlated_features


# Copy data and add treatment to the data frame
df = X.copy()
df['Treatment'] = y
df = df[jump * (change_point + 1):]

# Pair wise correlations between every two features
cor_features = get_feature_correlation(df)
cor_features.to_csv(save_path + '/Features_Pair_Correlation.csv', index=False)

# Correlation matrix between all features
correlations = df.corr()
correlations.to_csv(save_path + '/Features_Correlation.csv')

# Top features correlations
top_features = ['Max_size', 'Max_perimeter', 'Avg_size', 'Avg_hue', 'Avg_vari', 'Avg_ndi', 'Avg_perimeter', 'Std_vari',
                'Avg_std_hue', 'Std_std_hue', 'Avg_polar_second_moment_area', 'Avg_plant_angles', 'Range_vari',
                'Std_plant_angles', 'Std_avg_hue', 'Max_avg_leaves_angles', 'Avg_value_hsv', 'Std_range_hue',
                'Std_size', 'Min_leaves_temp', 'Avg_plant_third_temp', 'Avg_saturation', 'Range_hue_2min_leaf',
                'Avg_range_hue', 'Num_of_leaves', 'Treatment']
top_df = df[top_features]
top_correlations = top_df.corr()
top_correlations.to_csv(save_path + '/Top_Features_Correlation.csv')


# -------------------------------------------------- Box Plots ------------------------------------------------------- #

path_to_save = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features\Graphs\smoothed\BP2'
temp_data = data.copy()
temp_data = temp_data[jump * (change_point + 1):]
del temp_data['Num']
temp_data['Treatment'] = np.where(temp_data['Treatment'] == 'D', 'D', 'Rest')

for c in range(2, len(temp_data.columns)):
    plot = sns.boxplot(x="Date", y=temp_data.columns[c], hue="Treatment", data=temp_data)
    plt.title(label=temp_data.columns[c])
    fig = plot.get_figure()
    temp_col = temp_data.columns[c] + '.PNG'
    fig.savefig(path_to_save + '/' + temp_col)
    plot.clear()


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------ Discretization ---------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# ----- Equal Bins Disc ----- #

def robust_equal_bins_streaming(x, b_points, stable, n_bins):

    def remove_outliers(x):
        q1_quantile = np.quantile(x, 0.25)
        q3_quantile = np.quantile(x, 0.75)
        h = 1.5 * iqr(x)
        f = x.copy()
        f = f[(f > q1_quantile - h) & (f < q3_quantile + h)]
        return f

    def robust_eb(stable, all, bins):
        stable = remove_outliers(stable)
        stable = stable.reset_index(drop=True)
        disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        disc.fit(np.asarray(stable).reshape(-1, 1))
        y = disc.transform(np.asarray(all).reshape(-1, 1)).reshape(-1)
        return y

    x_discretized = np.zeros(x.shape)
    for m in range(1, len(b_points)):
        x_day_i = x[b_points[m - 1]:b_points[m], :]
        x_stable = x_day_i[:stable, :]
        for n in range(x_discretized.shape[1]):
            x_discretized[b_points[m - 1]:b_points[m], n] = \
                robust_eb(pd.Series(x_stable[:, n]), pd.Series(x_day_i[:, n]), bins=n_bins)
    return x_discretized + 1


# Read data and set parameters
path = r'C:\Users\User\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
data = pd.read_csv(path + '/All_features_smoothed.csv')
data = data.sort_values(by=['Date', 'Treatment', 'Num'])
data = data.reset_index()
chosen_columns = ['Max_size', 'Avg_hue', 'Avg_vari', 'Std_std_hue', 'Avg_plant_angles', 'Avg_ndi',
                  'Avg_perimeter', 'Max_perimeter', 'Avg_plant_third_temp', 'Num_of_leaves']
disc_data = data[chosen_columns]
disc_data = data.copy()
del disc_data['index']
del disc_data['Treatment']
del disc_data['Num']
del disc_data['Date']
Class = pd.DataFrame(data['Treatment'] == 'D') + 1

jump = 188
bins = 4
break_points = range(0, data.shape[0] + jump, jump)

# Make equal bins discretization by treatment A values in each day
stable_size = data[:jump]
stable_size = len(stable_size[stable_size['Treatment'] == 'A'])
X_disc = pd.DataFrame(robust_equal_bins_streaming(np.array(disc_data), break_points, stable_size, bins))
chosen_columns.append('class')
X_disc = pd.concat([X_disc, Class], axis=1)
# X_disc.columns = chosen_columns
cols = list(data.columns[4:])
cols.append('class')
X_disc.columns = cols
X_disc.to_csv(path + '/Discretization/Disc_4Bins_all_features.csv', index=False)


# ---------------------------------------------- Entropy Disc -------------------------------------------------------- #

chosen_columns = ['Max_size', 'Avg_hue', 'Avg_vari', 'Std_std_hue', 'Avg_plant_angles', 'Avg_ndi',
                  'Avg_perimeter', 'Max_perimeter', 'Avg_plant_third_temp', 'Num_of_leaves']
disc = Discritization(data_path=path, N_bins=4)
X_disc, c_columns = disc.mdl_static(disc_data.values, np.array(Class), col_names=chosen_columns)
X_disc, c_columns = disc.mdl_static(disc_data.values, np.array(Class), col_names=data.columns)
df_disc = pd.DataFrame(X_disc + 1, columns=c_columns)
df_disc['class'] = Class
df_disc.to_csv(path + '/Discretization/Disc_Entropy_all_features.csv', index=False)




