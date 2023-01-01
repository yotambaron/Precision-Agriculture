clear all;
Jump = 120;
window_days = 2;
stable_days = 3;
stable = stable_days * Jump;
Window = Jump * window_days;
days = 17;
change_day = 3;
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

data_path = 'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Data';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Results\Multiclass\CDDRL';
df = readtable([data_path '\ws_disc.csv']); % Discritizied database
df_smooth = readtable([data_path '\WS_old.csv']); % No discritization database

Plants = [df_smooth(:, 3), df_smooth(:, 1), df_smooth(:, 2)];
True_array = zeros(size(Plants, 1), 1);

columns = {'WaterLeft', 'SPAD', 'size', 'x3rd_temp', 'class'};

df = df(:, columns);
df = table2array(df);

for i=1:size(Plants, 1)
    if i < (Jump+1)*change_day
            df(i, 5) = 1;
    end
end

n_features = size(df, 2);
thresholds = 0.3;
initial_df = df(1:stable, :);
dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 10, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end

bnet = learn_params(bnet, initial_df');

tic;

[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_days);

tt=toc;

feature_importance = zeros(1, size(df, 2));

for b=1:length(BNlist)
    BNs_path = [save_path '\\BNs\BN' num2str(b) '.csv'];
    dag_i = BNlist{b}.dag;
    csvwrite(BNs_path,dag_i)
    % Find markov blanket of the target variable
    mb = markov_blanket(BNlist{b}.dag, size(df, 2));
    feature_importance = feature_importance + mb;
end

feature_importance = array2table(feature_importance);
feature_importance.Properties.VariableNames = columns;
feature_importance

netowrks = (size(BNlist,2));
break_points = zeros(days,1);
break_points(1) = Jump;

for net=2:days
    break_points(net) = Jump+Jump*(net-1);
end

% add one BN0 since the first two days gets only one BN,
% add the second BN0 to ensure we predict for data(t) with BN(t-1)
% add BN0 for the first window days to predict correctly
for w=1:window_days
   BNlist = [{BNlist{1}}, BNlist];
end

tic;
for t=1:length(thresholds)
    threshold = thresholds(t);
    Accuracies = zeros(days, 1);
    preds_array = [];
    scores_array = [];
    for b=1:days
        bn = BNlist{b};
        if b==1
           data = df(1:break_points(b),:);
           true = df(1:break_points(b),n);
       else
           data = df((break_points(b-1)+1):break_points(b),:);
           true = df((break_points(b-1)+1):break_points(b),n);
        end
        [preds, scores] = BN_inference(bn, data, n);
        preds_array = [preds_array;preds];
        scores_array = [scores_array;scores];
        Accuracies(b) = mean(preds == true);
    end
    
    ttt=toc;
    
    All_Results = array2table([days_col, Accuracies]);
    All_Results.Properties.VariableNames = {'Day' 'Multiclass_Accuracy'};
    scores_array = array2table(scores_array);
    preds_array = array2table(preds_array);
    preds_scores = [Plants, scores_array(:, 2), preds_array, array2table(True_array)];
    preds_scores.Properties.VariableNames = {'Day' 'Treatment' 'Plant_Num' 'Probability' 'Prediction' 'True_Value'};
    Dist_Array = array2table(Dist_Array);
    UCL_Array = array2table(UCL_Array);
    Dist_Array.Properties.VariableNames = {'node_1' 'node_2' 'node_3' 'node_4' 'node_5'};
    UCL_Array.Properties.VariableNames = {'node_1' 'node_2' 'node_3' 'node_4' 'node_5'};

    writetable(All_Results,[save_path  '\All_Results.csv'])
    writetable(scores_array,[save_path  '\Scores.csv'])
    writetable(preds_scores,[save_path  '\Predictions.csv'])
    writetable(array2table(C_nodes),[save_path  '\Changed_Nodes.csv'])
    writetable(Dist_Array,[save_path '\Distances.csv'])
    writetable(UCL_Array,[save_path  '\UCLs.csv'])
end


