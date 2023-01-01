clear all;

% Set general parameters
Jump = 120;
window_days = 2;
stable_days = 3;
stable = stable_days * Jump;
Window = Jump * window_days;
days = 17;
change_day = 3;
days_col = (1:days);
days_col = reshape(days_col,[days 1]);

% Normal, Tabu10+params, Robust, UCL, CB  
methods_flags = [1, 1, 1, 1, 1];
methods_num = sum(methods_flags);
tabu_size = 10;
classes = 4;

% Set CDDRL parameters
alpha = 0.5;
lambda = 0.8;
weights = ones(1, methods_num) / methods_num;
weight_method = 1;
thresholds = [0.3];

% Results variables to save
BNlist_all = [];
C_nodes_all = [];

% Set paths
data_path = 'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Data';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Results\Multiclass\CDDRL';
df = readtable([data_path '\ws_disc.csv']); % Discritizied database
df_smooth = readtable([data_path '\WS_old.csv']); % No discritization database

Plants = [df_smooth(:, 3), df_smooth(:, 1), df_smooth(:, 2)];
True_array = zeros(size(Plants, 1), 1);

columns = {'WaterLeft', 'SPAD', 'size', 'x3rd_temp', 'class'};

df = df(:, columns);
df = table2array(df);

%for i=1:size(Plants, 1)
%    if i < (Jump+1)*change_day
%            df(i, 5) = 1;
%    end
%end

n_features = size(df, 2);
initial_df = df(1:stable, :);
dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 10, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end

bnet = learn_params(bnet, initial_df');

for method=1:length(methods_flags)
   
    % Normal CDDRL
    if method == 1 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 0;
        [C_nodes_normal, BNlist_normal, ~, ~] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        %BNlist_normal = save_bns(BNlist_normal, save_path, 'Normal', 0);
        BNlist_normal = adjust_BNlist(BNlist_normal, window_days, 0);
        BNlist_all = [BNlist_all; BNlist_normal];
        C_nodes_all = [C_nodes_all, C_nodes_normal];
    end
    
    % Tabu size 10 + params CDDRL
    if method == 2 && methods_flags(method) == 1
        tabu_flag = 1;
        params_flag = 1;
        robust_flag = 0;
        [C_nodes_tabu_params, BNlist_tabu_params, ~, ~] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        %BNlist_tabu_params = save_bns(BNlist_tabu_params, save_path, 'Tabu_Params', 0);
        BNlist_tabu_params = adjust_BNlist(BNlist_tabu_params, window_days, 0);
        BNlist_all = [BNlist_all; BNlist_tabu_params];
        C_nodes_all = [C_nodes_all, C_nodes_tabu_params];
    end
    
    % Robust CDDRL
    if method == 3 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 1;
        [C_nodes_robust, BNlist_robust, ~, ~] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        %BNlist_robust = save_bns(BNlist_robust, save_path, 'Robust', 0);
        BNlist_robust = adjust_BNlist(BNlist_robust, window_days, 0);
        BNlist_all = [BNlist_all; BNlist_robust];
        C_nodes_all = [C_nodes_all, C_nodes_robust];
    end
    
    % Dynamic UCLs CDDRL
    if method == 4 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 0;
        UCL_days = 3;
        [C_nodes_UCLs, BNlist_UCLs, ~, ~] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_days);
        %BNlist_UCLs = save_bns(BNlist_UCLs, save_path, 'Dynamic_UCLs', 0); 
        BNlist_UCLs = adjust_BNlist(BNlist_UCLs, window_days, 0);
        BNlist_all = [BNlist_all; BNlist_UCLs];
        C_nodes_all = [C_nodes_all, C_nodes_UCLs];
    end
    
    % Constraint-Based CDDRL
    if method == 5 && methods_flags(method) == 1
        [C_nodes_CB, BNlist_CB, test_num_CB] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
        %BNlist_CB = save_bns(BNlist_CB, save_path, 'CB', 0);
        BNlist_CB = adjust_BNlist(BNlist_CB, window_days, 0);
        BNlist_all = [BNlist_all; BNlist_CB];
        C_nodes_all = [C_nodes_all, C_nodes_CB];
    end
end

netowrks = (size(BNlist_all, 2)) - 1;
break_points = zeros(days, 1);

for net=1:days
    break_points(net) = Jump + Jump * (net-1);
end

for t=1:length(thresholds)
    threshold = thresholds(t);
    Accuracies = zeros(days, 1);
    preds_array = [];
    scores_array = [];
    weights_all = [];
    
    for b=1:days
        display('predicting day: ')
        b
        weights_all = [weights_all; weights];
        bns = [];
        if b==1
            data = df(1:break_points(b),:);
            true = df(1:break_points(b),n);
        else
            data = df((break_points(b-1)+1):break_points(b),:);
            true = df((break_points(b-1)+1):break_points(b),n);
        end
        
        for m=1:size(BNlist_all, 1)
            bns = [bns; BNlist_all{m, b}];
        end
        
        % Infere by the weighted predictions of all methods
        [preds, scores] = BN_inference_ensemble(bns, data, n, methods_num);
        probabilities = zeros(size(scores,1), classes);
        for s=1:length(weights)
            probabilities = probabilities + weights(s) * reshape(cell2mat(scores(:, s)), [classes, size(scores, 1)])';
        end
        pred = zeros(size(scores,1), 1);
        for pp=1:length(pred)
           ppp = find(probabilities(pp, :) == max(probabilities(pp, :)));
           if length(ppp) > 1
               random_pred = randperm(length(ppp), 1);
               pred(pp) = ppp(random_pred);
           else
               pred(pp) = ppp(1);
           end
        end
        %pred(probabilities(:, 2) > threshold) = 1;
        
        % Update the previous weights with the current methods' performance
        [weights, ~] = update_ensemble_weights(weights, scores, true - 1, weight_method, 0, classes);
        preds_array = [preds_array; pred];
        scores_array = [scores_array; probabilities];
        Accuracies(b) = mean(pred == true);
        
    end
    
end

% Combine all BNs from the different methods by a certain combine method
weight_threshold = 0.5;
bns_all = BNlist_all(:, (1 + window_days):days+1);

combine_method = 'every_edge'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);


combine_method = 'majority_vote'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);


combine_method = 'every_edge_with_weight'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);


combine_method = 'every_edge_over_weight_threshold'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);

% Save results
All_Results = array2table([days_col, Accuracies]);
All_Results.Properties.VariableNames = {'Day' 'Multiclass_Accuracy'};
writetable(All_Results,[save_path '\All_Results.csv'])
writetable(array2table(C_nodes_all),[save_path '\Changed_Nodes.csv'])
writetable(array2table(weights_all),[save_path '\weights.csv'])
writetable(array2table(scores_array),[save_path  '\Scores.csv'])
writetable(array2table(preds_array),[save_path  '\Predictions.csv'])

