clear all;
Jump = 94;
Window = Jump * 2;
stable_days = 3;
stable = stable_days * Jump;
days = 41;
days_col = (1:days);
days_col = reshape(days_col,[days 1]);

% CDDRL parameters
alpha = 0.5;
lambda = 0.8;
tabu_flag = 0;
tabu_size = 10;
robust_flag = 0;
params_flag = 0;
UCL_days = 3;

data_path = 'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Data\AD_Comp';
save_path = 'C:\Users\User\Desktop\yotam\MATLAB\Stress_Experiments\Water_Stress\Results\CDDRL\A_VS_D\entropy\Entropy_disc_ucl';
df=readtable([data_path '\ws_disc_entropyAD.csv']); % Discritizied database
df_smooth = readtable([data_path '\þþws_features_smoothAD.csv']); % No discritization database

Plants = [df_smooth(:, 3), df_smooth(:, 1), df_smooth(:, 2)];

columns = {'Max_size', 'Avg_hue', 'Avg_vari', 'Std_std_hue', 'Avg_plant_angles', 'Avg_ndi', 'Avg_perimeter', 'Max_perimeter', 'Avg_plant_third_temp', 'Num_of_leaves', 'class'};
%columns = {'Max_size', 'Avg_vari', 'Std_std_hue', 'Avg_ndi', 'Avg_perimeter', 'Max_perimeter', 'Avg_plant_third_temp', 'Num_of_leaves', 'class'};

%columns = {'Max_size', 'Avg_size', 'Max_perimeter', 'Avg_perimeter', 'Avg_ndi', 'Avg_hue', 'Num_of_leaves', 'Avg_vari', 'Avg_polar_second_moment_area', 'Avg_std_hue', 'Avg_plant_angles', 'Std_std_hue', 'Max_avg_leaves_angles', 'Std_plant_angles', 'Std_size', 'class'};
%columns = {'Max_size', 'Avg_ndi', 'Num_of_leaves', 'Avg_std_hue', 'Std_std_hue', 'class'};

df = df(:, columns);
df = table2array(df);
True_array = zeros(size(Plants, 1), 1);

for i=1:size(Plants, 1)
    if table2array(Plants{i, 2}) == 'D'
        True_array(i) = 1;
    end
end

n_features = size(df, 2);
thresholds = 0.5;
initial_df = df(1:stable, :);
dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 10, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end

bnet = learn_params(bnet, initial_df');

%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_static(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_days);

feature_importance = zeros(1, size(df, 2));

for b=1:length(BNlist)
    BNs_path = [save_path '\\BNs\BN' num2str(b) '.csv'];
    dag_i = BNlist{b}.dag;
    csvwrite(BNs_path,dag_i)
    childs = find(BNlist{b}.dag(size(df, 2), :));
    feature_importance(childs) = feature_importance(childs) + 1;
end

feature_importance = array2table(feature_importance);
feature_importance.Properties.VariableNames = columns;

netowrks = (size(BNlist,2));
break_points = zeros(days,1);
break_points(1) = Jump;

for net=2:days
    break_points(net) = Jump+Jump*(net-1);
end

% add one BN0 since the first two days gets only one BN,
% add the second BN0 to ensure we predict for BN(t) with BN(t-1)
BNlist = [{BNlist{1}}, BNlist];
BNlist = [{BNlist{1}}, BNlist];


for t=1:length(thresholds)
    threshold = thresholds(t);
    aucs = zeros(days,1);
    FAs = zeros(days,1);
    TPs = zeros(days,1);
    FPs = zeros(days,1);
    FNs = zeros(days,1);
    TNs = zeros(days,1);
    preds_array = [];
    scores_array = [];
    for b=1:days
        bn = BNlist{b};
        if b==1
           data = df(1:break_points(b),:);
           true = df(1:break_points(b),n) - 1;
       else
           data = df((break_points(b-1)+1):break_points(b),:);
           true = df((break_points(b-1)+1):break_points(b),n)-1;
        end
        [preds, scores] = BN_inference(bn,data,n);
        pred = zeros(size(scores,1),1);
        pred(scores > threshold) = 1;
        preds_array = [preds_array;pred];
        scores_array = [scores_array;scores];
        [~,~,~,AUC] = perfcurve(true,scores,1);
        aucs(b) = AUC;
        [TPs(b),FPs(b),FNs(b),TNs(b)] = confusion_matrix(pred, true);
        [TP_precision(b,1),TP_recall(b,1),TN_precision(b,1),TN_recall(b,1),FA_rate(b,1),f1_p(b,1),f1_n(b,1),f1_a(b,1),gm(b,1),b_accuracy(b,1),y_score(b,1)] = score_matrices(TPs(b),FPs(b),FNs(b),TNs(b));
    end
    
    All_Results = array2table([days_col,TPs,FPs,FNs,TNs,FA_rate,TP_precision,TP_recall,TN_precision,TN_recall,f1_p,f1_n,f1_a,gm,b_accuracy,y_score,aucs]);
    All_Results.Properties.VariableNames = {'Day' 'TP' 'FP' 'FN' 'TN' 'FA' 'TP_Precision' 'TP_Recall' 'TN_Precision' 'TN_Recall' 'F1_pos' 'F1_neg' 'F1_Avg' 'Geometric_mean' 'Balanced_Accuracy' 'Y_score' 'AUC'};
    scores_array = array2table(scores_array);
    preds_array = array2table(preds_array);
    preds_scores = [Plants,scores_array,preds_array,array2table(True_array)];
    preds_scores.Properties.VariableNames = {'Day' 'Treatment' 'Plant_Num' 'Probability' 'Prediction' 'True_Value'};
    %Dist_Array = array2table(Dist_Array);
    %UCL_Array = array2table(UCL_Array);
    %Dist_Array.Properties.VariableNames = {'node_1' 'node_2' 'node_3' 'node_4' 'node_5' 'node_6' 'node_7' 'node_8' 'node_9' 'node_10' 'node_11' 'node_12' 'node_13' 'node_14' 'node_15' 'node_16' 'node_17'};
    %UCL_Array.Properties.VariableNames = {'node_1' 'node_2' 'node_3' 'node_4' 'node_5' 'node_6' 'node_7' 'node_8' 'node_9' 'node_10' 'node_11' 'node_12' 'node_13' 'node_14' 'node_15' 'node_16' 'node_17'};

    writetable(All_Results,[save_path  '\All_Results.csv'])
    writetable(array2table(aucs),[save_path  '\AUCs.csv'])
    writetable(preds_scores,[save_path  '\Predictions.csv'])
    %writetable(Dist_Array,[save_path '\Distances.csv'])
    %writetable(UCL_Array,[save_path  '\UCLs.csv'])
    writetable(array2table(C_nodes),[save_path  '\Changed_Nodes.csv'])
end


