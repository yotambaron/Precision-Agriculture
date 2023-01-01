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

% Set CDDRL parameters
threshold = 0.5;
classes = 4;
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_days = 3;
bns_ensemble = 3;
weight_flag = 0;
weights = [];
curr_bn_weight = 0.5;
previous_bns_weight = 1 - curr_bn_weight;
ensemble_list = [];
bns_inds = cell(days, 1);

% Results variables to save
C_nodes_all = [];
bns_inds_list = [];

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

[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_days);

%for b=1:length(BNlist)
%    BNs_path = [save_path '\BNs\BN' num2str(b) '.csv'];
%    dag_i = BNlist{b}.dag;
%    csvwrite(BNs_path,dag_i)
%end
    
 netowrks = (size(BNlist, 2));
 break_points = zeros(days, 1);

 for net=1:days
     break_points(net) = Jump + Jump * (net-1);
 end

% add BN0 for the first window days to predict correctly
BNlist = adjust_BNlist(BNlist, window_days, 0);

Accuracies = zeros(days, 1);
preds_array = [];
scores_array = [];

for b=1:days
   display('predicting day: ')
   b
   bn = BNlist{b};
      
   if b==1
       data = df(1:break_points(b),:);
       true = df(1:break_points(b),n);
       % Normal inference only with the current BN        
       [preds, scores] = BN_inference(bn, data, n);
       pred = preds;
       % Add the current BN to the BN list and update weights
       ensemble_list = [ensemble_list; bn];
       weights = [weights; 1];
       bns_inds_list = [bns_inds_list; b];
      
   else
       data = df((break_points(b-1)+1):break_points(b),:);
       true = df((break_points(b-1)+1):break_points(b),n);
           
       % Infere with all BNs in the list + current BN - current BN's
       % weight is decided by 'curr_bn_weight' and the past BNs
       % contribute the rest of the weight by their previous weights
       temp_ensemble_list = [ensemble_list; bn];
       [preds, scores] = BN_inference_ensemble(temp_ensemble_list, data, n, length(temp_ensemble_list));
       temp_weights = weights * previous_bns_weight;
       temp_weights = [temp_weights; curr_bn_weight];
       probabilities = zeros(size(scores,1), classes);
       for s=1:length(temp_weights)
           probabilities = probabilities + temp_weights(s) * reshape(cell2mat(scores(:, s)), [classes, size(scores, 1)])';
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
        
       % Update current weights by the current performance
       new_weights = update_ensemble_weights(ones(1, length(temp_ensemble_list)), scores, true - 1, weight_flag, 0, classes);
       % Normalize the previous weights with a non real BN weight
       if b > 2
           weights = [weights; 1/length(weights)];
           weights = weights / sum(weights);
           weights(end) = [];
       end
       curr_weight = new_weights(end);
       new_weights(end) = [];
       % New weights of the past BNs is the average between previous
       % and normalized current weights
       weights = (weights + new_weights') / 2;
       if b <= bns_ensemble
           weights = [weights; curr_weight];
           ensemble_list = [ensemble_list; bn];
           bns_inds_list = [bns_inds_list; b];
       else
           % If current BN weight is bigger than any of the previous
           % BNs then replace them in BN list
           min_weight_pos = find(weights == min(weights));
           if weights(min_weight_pos(1)) <= curr_weight
               weights(min_weight_pos(1)) = curr_weight;
               ensemble_list(min_weight_pos(1)) = bn;
               bns_inds_list(min_weight_pos(1)) = b;
           end
       end
       % Normalize the updated BN list weights
       weights = weights / sum(weights);
       scores = probabilities;
   end
      
   bns_inds{b, 1} = bns_inds_list;
   
   preds_array = [preds_array; pred];
   scores_array = [scores_array; scores];
   Accuracies(b) = mean(pred == true);
   
end

All_Results = array2table([days_col, Accuracies]);
All_Results.Properties.VariableNames = {'Day' 'Multiclass_Accuracy'};
writetable(All_Results,[save_path '\All_Results_Ensemble.csv'])
writetable(array2table(C_nodes),[save_path '\Changed_Nodes_Ensemble.csv'])
writetable(cell2table(bns_inds),[save_path '\BNs_In_Ensemble.csv'])
writetable(array2table(scores_array),[save_path  '\Scores.csv'])
writetable(array2table(preds_array),[save_path  '\Predictions.csv'])
    