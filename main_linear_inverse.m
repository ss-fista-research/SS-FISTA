clear all
close all
clc
code_root = fileparts(mfilename('fullpath'));
rng(42,'twister');
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

%% problem set up
J = 'lasso';
% J = 'tv';

[para, gradF, proxJ, objPhi] = problem_FB(J);

%% parameters
para.J = J;
para.tol = 1e-15;
para.maxits = 1e4;
para.gamma = para.beta;
para.x0 = zeros(para.n, 1);
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

axis([1, 1000, min(dk5), 2*max(dk1)]);
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

%% display/save results
disp([its1, its2, its4, its5])
disp([t1, t2, t4, t5])
if ~exist(fullfile(code_root,'results','figures'), 'dir')
    mkdir(fullfile(code_root,'results','figures'));
end
if ~exist(fullfile(code_root,'results','raw'), 'dir')
    mkdir(fullfile(code_root,'results','raw'));
end
filename = fullfile(code_root,'results','figures', sprintf('linear_%s_convergence', J));
print([filename '.eps'], '-depsc', '-r600');
print([filename '.pdf'], '-dpdf');
save(fullfile(code_root,'results','raw', sprintf('linear_%s_results.mat', J)), ...
    'J', 'para', 'p', 'q', 'r', 'xsol', ...
    'x1', 'its1', 'dk1', 'ek1', 'fk1', 't1', ...
    'x2', 'its2', 'dk2', 'ek2', 'fk2', 't2', ...
    'x4', 'its4', 'dk4', 'ek4', 'fk4', 't4', ...
    'x5', 'its5', 'dk5', 'ek5', 'fk5', 't5');
