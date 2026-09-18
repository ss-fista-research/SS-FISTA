clear all
close all
clc
code_root = fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(fullfile(code_root, 'toolbox'));
addpath(fullfile(code_root, 'menthod'));
formal_method_names = {'func_FISTA', 'func_FISTA_Mod', ...
    'func_GAPGA', 'func_SSFISTA'};
formal_method_dir = fullfile(code_root, 'menthod');
for method_index = 1:numel(formal_method_names)
    method_name = formal_method_names{method_index};
    resolved_method = which(method_name);
    expected_method = fullfile(formal_method_dir, [method_name '.m']);
    if isempty(resolved_method) || ~strcmpi(resolved_method, expected_method)
        error('SSFISTA:FormalMethodPath', ...
            '%s must resolve to %s (resolved: %s).', ...
            method_name, expected_method, resolved_method);
    end
end
set(groot,'defaultLineLineWidth',1.5);

%% load data
datasets = {'splice', 'ionosphere', 'australian', 'diabetes'};
for dataset_index = 1:numel(datasets)
dataset = datasets{dataset_index};
load(fullfile(code_root,'data','logistic', [dataset '.mat']));
h_train = full(features_train);
l_train = labels_train;
h_test = full(features_test);
l_test = labels_test;

%% preprocessing: fit scaling on training data only
train_min = zeros(1,size(h_train,2));
train_max = zeros(1,size(h_train,2));
for j=1:size(h_train,2)
    train_min(j) = min(h_train(:,j));
    train_max(j) = max(h_train(:,j));
    if train_max(j) == train_min(j)
        h_train(:,j) = 0;
        h_test(:,j) = 0;
    else
        h_train(:,j) = 2*(h_train(:,j)-train_min(j))/(train_max(j)-train_min(j))-1;
        h_test(:,j) = 2*(h_test(:,j)-train_min(j))/(train_max(j)-train_min(j))-1;
    end
end

%% parameters
[m, n] = size(h_train);
para.m = m;
para.n = n;
para.W = h_train;
para.y = l_train;
para.mu = 1e-2;

Li = zeros(m, 1);
for i=1:m
    Wi = para.W(i,:);
    Li(i) = norm(Wi)^2 /4;
end
para.beta_fi = 1 /max(Li);
para.tol = 1e-17;
para.maxits = 1e4;
para.beta = 4*m/norm(para.W)^2;
para.gamma = para.beta;

gradF = @(x) grad_logistic(x, para.W, para.y) /m;
proxJ = @(x, t) wthresh(x, 's', t);
objPhi = @(x) 1; % Original placeholder retained; fk is not the objective value.
para.x0 = zeros(n, 1);
para.verbose = 1;
para.a = 40;

%% reference xsol: separate FISTA run, excluded from its1 and t1
fprintf('computing reference with FISTA...\n');
xsol = func_FISTA(para, proxJ,gradF, objPhi, 0);
fprintf('\n');

%% FISTA-Mod
fprintf('performing FISTA-Mod...\n');
p = 1/20;
q = 1/2;
r = 4;
[x2, its2, dk2, ek2, fk2, t2] = func_FISTA_Mod(p,q,r, para, proxJ,gradF, objPhi, xsol);
fprintf('\n');

%% FISTA
fprintf('performing original FISTA...\n');
[x1, its1, dk1, ek1, fk1, t1] = func_FISTA(para, proxJ,gradF, objPhi, xsol);
fprintf('\n');

%% GAPGA
fprintf('performing GAPGA...\n');
[x4, its4, dk4, ek4, fk4, t4] = func_GAPGA(para, proxJ,gradF, objPhi, xsol);
fprintf('\n');

%% SS-FISTA
fprintf('performing SS-FISTA...\n');
[x5, its5, dk5, ek5, fk5, t5] = func_SSFISTA(para, proxJ,gradF, objPhi, xsol);
fprintf('\n');

%% plotting
hh = parula;
linewidth = 1;
axesFontSize = 8;
labelFontSize = 8;
legendFontSize = 8;
resolution = 300;
output_size = 300 *[10, 8];

figure(101), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.85 0.4]);

p1d = semilogy(dk1, 'color', [1/1.75,1/1/1.75,1/1/1.75], 'LineWidth',linewidth);
hold on,
p2d = semilogy(dk2, 'color', [1/6,1/6,1/6], 'LineWidth',linewidth);
p4d = semilogy(dk4, 'color', [1/6,1/4,1/1.2], 'LineWidth',linewidth);
p5d = semilogy(dk5, 'color', [1,0,0], 'LineWidth',linewidth);
uistack(p1d, 'bottom');
grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1, 1000, min(dk2), 2*max(dk1)]);
if strcmp(dataset, 'splice')
    xlim([1, 350]);
elseif strcmp(dataset, 'ionosphere')
    xlim([1, 2000]);
end
ytick = [1e-12, 1e-8, 1e-4, 1e-0, 1e4];
set(gca, 'yTick', ytick);
ylb = ylabel({'$\|x_{k}-x^\star\|$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.1, 0.5, 0]);
xlb = xlabel({'\vspace{-1.0mm}';'$k$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.075, 0]);

lg = legend([p1d,p2d,p4d,p5d], ...
    'FISTA',...
    'FISTA-Mod, $p = \frac{1}{20}, q = \frac{1}{2}$',...
    'GAPGA',...
    'SS-FISTA');
set(lg,'FontSize', legendFontSize);
set(lg, 'Interpreter', 'latex');
legend('boxoff');

%% learning metrics: testing data only, no regularization in log-loss
score1 = h_test*x1;
score2 = h_test*x2;
score4 = h_test*x4;
score5 = h_test*x5;

prediction1 = 2*(score1 >= 0)-1;
prediction2 = 2*(score2 >= 0)-1;
prediction4 = 2*(score4 >= 0)-1;
prediction5 = 2*(score5 >= 0)-1;
accuracy1 = mean(prediction1 == l_test);
accuracy2 = mean(prediction2 == l_test);
accuracy4 = mean(prediction4 == l_test);
accuracy5 = mean(prediction5 == l_test);

z1 = -l_test .* score1;
z2 = -l_test .* score2;
z4 = -l_test .* score4;
z5 = -l_test .* score5;
logloss1 = mean(max(z1,0) + log1p(exp(-abs(z1))));
logloss2 = mean(max(z2,0) + log1p(exp(-abs(z2))));
logloss4 = mean(max(z4,0) + log1p(exp(-abs(z4))));
logloss5 = mean(max(z5,0) + log1p(exp(-abs(z5))));

threshold = 1e-8;
nnz1 = sum(abs(x1) > threshold);
nnz2 = sum(abs(x2) > threshold);
nnz4 = sum(abs(x4) > threshold);
nnz5 = sum(abs(x5) > threshold);

Method = {'FISTA'; 'FISTA-Mod'; 'GAPGA'; 'SS-FISTA'};
Iterations = [its1; its2; its4; its5];
CPU_time = [t1; t2; t4; t5];
Test_Accuracy = [accuracy1; accuracy2; accuracy4; accuracy5];
Test_LogLoss = [logloss1; logloss2; logloss4; logloss5];
NNZ = [nnz1; nnz2; nnz4; nnz5];
metrics = table(Method, Iterations, CPU_time, Test_Accuracy, Test_LogLoss, NNZ);

%% display/save results
fprintf('Dataset: %s\n', dataset);
disp(metrics)
if ~exist(fullfile(code_root,'results','figures'), 'dir')
    mkdir(fullfile(code_root,'results','figures'));
end
if ~exist(fullfile(code_root,'results','raw'), 'dir')
    mkdir(fullfile(code_root,'results','raw'));
end
if ~exist(fullfile(code_root,'results','tables'), 'dir')
    mkdir(fullfile(code_root,'results','tables'));
end
filename = fullfile(code_root,'results','figures', sprintf('logistic_%s_convergence', dataset));
print([filename '.eps'], '-depsc', '-r600');
print([filename '.pdf'], '-dpdf');
save(fullfile(code_root,'results','raw', sprintf('logistic_%s_results.mat', dataset)), ...
    'dataset', 'para', 'p', 'q', 'r', 'xsol', ...
    'train_min', 'train_max', 'threshold', 'metrics', ...
    'x1', 'its1', 'dk1', 'ek1', 'fk1', 't1', ...
    'x2', 'its2', 'dk2', 'ek2', 'fk2', 't2', ...
    'x4', 'its4', 'dk4', 'ek4', 'fk4', 't4', ...
    'x5', 'its5', 'dk5', 'ek5', 'fk5', 't5');
writetable(metrics, fullfile(code_root,'results','tables', sprintf('logistic_%s_metrics.csv', dataset)));
end
