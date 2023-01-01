clear all;

window_days = 2;
Jump = 201;
Window = Jump * window_days;
change_day = 7;
stable_days = 3;
stable = stable_days * Jump;
days = 28;
days_col = (1: days);
days_col = reshape(days_col,[days 1]);

% CDDRL parameters
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_days = 3;

data_path = 'D:\yotam\MATLAB\Stress_Experiments\Fertilizer_Stress\Data';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\Fertilizer_Stress\Results\CDDRL\stable_3\Tabu10+Params';
df =  readtable([data_path '\fs_disc_entropy_stressOnly.csv']); % Discritization data
df_smooth = readtable([data_path '\fs_features_smooth.csv']); % No discritization database for true array
df = table2array(df);
Plants = [df_smooth(:, 3), df_smooth(:, 1), df_smooth(:, 2)];
True_array = zeros(size(Plants, 1), 1);


for i=(Jump+1)*change_day:size(Plants, 1)
    if table2array(Plants{i, 2}) == 'A'
        True_array(i) = 1;
    end
end

n_features = size(df, 2);
thresholds = 0.3;

% Create stable df and build BN0 using MMHC
initial_df = df(1:stable, :);
dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 100, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end
bnet = learn_params(bnet, initial_df');

% Run CDDRL over the whole time period
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_static(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag);
[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_days);

columns = {'Mean_std_green', 'Max_angle', 'Mean_Mean_green', 'std_vari', 'Mean_perimeter', 'Std_perimeter', 'std_ndi', 'Mean_Size', 'Mean_egi', 'std_Mean_green', 'class'};

feature_importance = zeros(1, size(df, 2));

% Save all BNs found during the process
for b=1:length(BNlist)
    BNs_path = [save_path '\BNs\BN' num2str(b) '.csv'];
    dag_i = BNlist{b}.dag;
    csvwrite(BNs_path,dag_i)
    % Find markov blanket of the target variable
    mb = markov_blanket(BNlist{b}.dag, size(df, 2));
    feature_importance = feature_importance + mb;
end

feature_importance = array2table(feature_importance);
feature_importance.Properties.VariableNames = columns;
feature_importance

% Create time steps
netowrks=(size(BNlist,2));
break_points = zeros(days,1);
break_points(1) = Jump;

for net=2:days
    break_points(net) = Jump+Jump*(net-1);
end

% add one BN0 since the first two days gets only one BN,
% add the second BN0 to ensure we predict for BN(t) with BN(t-1)
for w=1:window_days
   BNlist = [{BNlist{1}}, BNlist];
end

% Predict, calculate performance and save results
for t=1:length(thresholds)
    threshold = thresholds(t);
    aucs = zeros(days,1) + 0.5;
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
        
        [preds, scores] = BN_inference(bn, data, n);
        pred = zeros(size(scores, 1), 1);
        pred(scores(:, 2) > threshold) = 1;
        preds_array = [preds_array;pred];
        scores_array = [scores_array;scores];
        
        if b > change_day
            [~,~,~,AUC] = perfcurve(true, scores(:, 2), 1);
            aucs(b) = AUC;
        end
        [TPs(b),FPs(b),FNs(b),TNs(b)] = confusion_matrix(pred, true);
        [TP_precision(b,1),TP_recall(b,1),TN_precision(b,1),TN_recall(b,1),FA_rate(b,1),f1_p(b,1),f1_n(b,1),f1_a(b,1),gm(b,1),b_accuracy(b,1),y_score(b,1)] = score_matrices(TPs(b),FPs(b),FNs(b),TNs(b));
    end

    All_Results = array2table([days_col,TPs,FPs,FNs,TNs,FA_rate,TP_precision,TP_recall,TN_precision,TN_recall,f1_p,f1_n,f1_a,gm,b_accuracy,y_score,aucs]);
    All_Results.Properties.VariableNames = {'Day' 'TP' 'FP' 'FN' 'TN' 'FA' 'TP_Precision' 'TP_Recall' 'TN_Precision' 'TN_Recall' 'F1_pos' 'F1_neg' 'F1_Avg' 'Geometric_mean' 'Balanced_Accuracy' 'Y_score' 'AUC'};
    scores_array = array2table(scores_array);
    preds_array = array2table(preds_array);
    preds_scores = [Plants, scores_array(:, 2), preds_array, array2table(True_array)];
    preds_scores.Properties.VariableNames = {'Day' 'Treatment' 'Plant_Num' 'Probability' 'Prediction' 'True_Value'};
    
    Dist_Array = array2table(Dist_Array);
    UCL_Array = array2table(UCL_Array);
    Dist_Array.Properties.VariableNames = {'node_1' 'node_2' 'node_3' 'node_4' 'node_5' 'node_6' 'node_7' 'node_8' 'node_9' 'node_10' 'node_11'};
    UCL_Array.Properties.VariableNames = {'node_1' 'node_2' 'node_3' 'node_4' 'node_5' 'node_6' 'node_7' 'node_8' 'node_9' 'node_10' 'node_11'};

    writetable(All_Results,[save_path '\All_Results.csv'])
    writetable(array2table(aucs),[save_path '\AUCs.csv'])
    writetable(preds_scores,[save_path '\Predictions.csv'])
    writetable(Dist_Array,[save_path '\Distances.csv'])
    writetable(UCL_Array,[save_path '\UCLs.csv'])
    writetable(array2table(C_nodes),[save_path '\Changed_Nodes.csv'])
    
end

