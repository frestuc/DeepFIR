% data: array to compute average value
% epsilon: to get the confidence interval, if epsilon = 0.05 -> you get 95% confidence intervals
% n_sim: number of elements in data
function [mean_value, yCI95] = compute_confidence(data,epsilon)

n_sim = numel(data);

% Mean Value
mean_value = mean(data);

% Confidence Intervals
ySEM = std(data)/sqrt(n_sim);
CI95 = tinv([0+epsilon 1-epsilon], n_sim-1);
yCI95 = bsxfun(@times, ySEM, CI95(:)); 

return