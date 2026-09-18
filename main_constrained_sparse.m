%% Section 4.2: feasibility-constrained sparse recovery

clear;
clc;
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

%% problem setup
rng(42, 'twister');
m = 512;
n = 1024;
s = 64;
lambda = 1;
A = randn(m, n) / sqrt(m);
support = randperm(n, s);
x_true = zeros(n, 1);
x_true(support) = 1 + rand(s, 1);
b_clean = A * x_true;
noise_std = 1e-3 * std(b_clean);
b = b_clean + noise_std * randn(m, 1);

%% parameters
L = norm(A)^2;
para.m = m;
para.n = n;
para.mu = lambda;
para.gamma = 1 / L;
para.tol = 1e-17;
para.maxits = 10000;
para.x0 = ones(n, 1) / n;
para.verbose = 1;
para.a = 40;

p = 1/20;
q = 1/2;
r = 4;
gradF = @(x) A' * (A * x - b);
proxJ = @(v, t) max(v - t * lambda, 0);
objPhi = @(x) 0.5 * norm(A * x - b)^2 + lambda * norm(x, 1);

%% reference FISTA
fprintf('----------------------------------------\n');
fprintf('Computing FISTA numerical reference...\n');
fprintf('----------------------------------------\n');
[x_ref, ref_its, ref_dk, ref_ek, ref_fk, ref_cpu] = ...
    func_FISTA(para, proxJ, gradF, objPhi, zeros(n, 1));
fprintf('Reference completed: iterations = %d, CPU = %.4f s\n', ref_its, ref_cpu);

%% FISTA-Mod
fprintf('\n----------------------------------------\n');
fprintf('Performing FISTA-Mod...\n');
fprintf('----------------------------------------\n');
[x2, its2, dk2, ek2, fk2, t2] = ...
    func_FISTA_Mod(p, q, r, para, proxJ, gradF, objPhi, x_ref);
fprintf('FISTA-Mod completed: iterations = %d, CPU = %.4f s\n', its2, t2);

%% FISTA
fprintf('\n----------------------------------------\n');
fprintf('Performing FISTA...\n');
fprintf('----------------------------------------\n');
[x1, its1, dk1, ek1, fk1, t1] = ...
    func_FISTA(para, proxJ, gradF, objPhi, x_ref);
fprintf('FISTA completed: iterations = %d, CPU = %.4f s\n', its1, t1);

%% GAPGA
fprintf('\n----------------------------------------\n');
fprintf('Performing GAPGA...\n');
fprintf('----------------------------------------\n');
[x4, its4, dk4, ek4, fk4, t4] = ...
    func_GAPGA(para, proxJ, gradF, objPhi, x_ref);
fprintf('GAPGA completed: iterations = %d, CPU = %.4f s\n', its4, t4);

%% SS-FISTA
fprintf('\n----------------------------------------\n');
fprintf('Performing SS-FISTA...\n');
fprintf('----------------------------------------\n');
[x5, its5, dk5, ek5, fk5, t5] = ...
    func_SSFISTA(para, proxJ, gradF, objPhi, x_ref);
fprintf('SS-FISTA completed: iterations = %d, CPU = %.4f s\n', its5, t5);

%% results
Method = {'FISTA'; 'FISTA-Mod'; 'GAPGA'; 'SS-FISTA'};
Iterations = [its1; its2; its4; its5];
CPUTime = [t1; t2; t4; t5];
FinalDistanceToReference = [norm(x1 - x_ref, 2); norm(x2 - x_ref, 2); ...
    norm(x4 - x_ref, 2); norm(x5 - x_ref, 2)];
FinalObjective = [objPhi(x1); objPhi(x2); objPhi(x4); objPhi(x5)];
RelativeRecoveryError = [norm(x1 - x_true, 2); norm(x2 - x_true, 2); ...
    norm(x4 - x_true, 2); norm(x5 - x_true, 2)] / norm(x_true, 2);
NNZ = [nnz(abs(x1) > 1e-8); nnz(abs(x2) > 1e-8); ...
    nnz(abs(x4) > 1e-8); nnz(abs(x5) > 1e-8)];
GradientEvaluations = [its1; its2; 1 + 2*its4; its5];
ProxEvaluations = GradientEvaluations;
results_table = table(Method, Iterations, CPUTime, FinalDistanceToReference, ...
    FinalObjective, RelativeRecoveryError, NNZ, GradientEvaluations, ProxEvaluations);

%% independent feasibility diagnostics (excluded from formal CPU timing)
diagnostics = run_constrained_sparse_diagnostics(para, proxJ, gradF, objPhi, x_ref, p, q, r, [its1; its2; its4; its5], {x1; x2; x4; x5});

disp([its1, its2, its4, its5])
disp([t1, t2, t4, t5])

%% plotting and save
% Cross-check the static method-call counts against the query counts from the
% independent diagnostic pass before producing any official outputs.
DiagnosticGradientEvaluations = [ ...
    diagnostics(1).NumberOfGradientEvaluations; ...
    diagnostics(2).NumberOfGradientEvaluations; ...
    diagnostics(3).NumberOfGradientEvaluations; ...
    diagnostics(4).NumberOfGradientEvaluations];
DiagnosticProxEvaluations = [ ...
    diagnostics(1).NumberOfProximalEvaluations; ...
    diagnostics(2).NumberOfProximalEvaluations; ...
    diagnostics(3).NumberOfProximalEvaluations; ...
    diagnostics(4).NumberOfProximalEvaluations];
if ~isequal(GradientEvaluations, DiagnosticGradientEvaluations)
    error('SSFISTA:GradientEvaluationCountMismatch', ...
        'Static gradient-evaluation counts do not match the diagnostic pass.');
end
if ~isequal(ProxEvaluations, DiagnosticProxEvaluations)
    error('SSFISTA:ProximalEvaluationCountMismatch', ...
        'Static proximal-evaluation counts do not match the diagnostic pass.');
end

InfeasibleGradientQueries = [ ...
    diagnostics(1).InfeasibleGradientPointCount; ...
    diagnostics(2).InfeasibleGradientPointCount; ...
    diagnostics(3).InfeasibleGradientPointCount; ...
    diagnostics(4).InfeasibleGradientPointCount];
MaxFeasibilityViolation = [ ...
    diagnostics(1).MaxFeasibilityViolation; ...
    diagnostics(2).MaxFeasibilityViolation; ...
    diagnostics(3).MaxFeasibilityViolation; ...
    diagnostics(4).MaxFeasibilityViolation];
MinimumGradientPointCoordinate = [ ...
    diagnostics(1).MinimumGradientPointCoordinate; ...
    diagnostics(2).MinimumGradientPointCoordinate; ...
    diagnostics(3).MinimumGradientPointCoordinate; ...
    diagnostics(4).MinimumGradientPointCoordinate];

% Detailed Section 4.2 record and the compact manuscript-facing table.
results_table = table(Method, Iterations, CPUTime, FinalDistanceToReference, ...
    FinalObjective, RelativeRecoveryError, NNZ, GradientEvaluations, ...
    ProxEvaluations, MaxFeasibilityViolation, MinimumGradientPointCoordinate, ...
    InfeasibleGradientQueries);
main_table = table(Method, Iterations, GradientEvaluations, ProxEvaluations, ...
    CPUTime, InfeasibleGradientQueries);

disp(main_table)

raw_dir = fullfile(code_root, 'results', 'raw');
figure_dir = fullfile(code_root, 'results', 'figures');
table_dir = fullfile(code_root, 'results', 'tables');
if ~exist(raw_dir, 'dir'), mkdir(raw_dir); end
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end
if ~exist(table_dir, 'dir'), mkdir(table_dir); end

writetable(results_table, fullfile(table_dir, 'constrained_sparse_results.csv'));
writetable(main_table, fullfile(table_dir, 'constrained_sparse_main_table.csv'));

problem = struct('m', m, 'n', n, 's', s, 'lambda', lambda, 'A', A, ...
    'x_true', x_true, 'support', support, 'b_clean', b_clean, ...
    'noise_std', noise_std, 'b', b, 'L', L, 'gamma', para.gamma, ...
    'x0', para.x0, 'tol', para.tol, 'maxits', para.maxits, ...
    'rng_seed', 42);
reference = struct('x', x_ref, 'Iterations', ref_its, 'CPUTime', ref_cpu, ...
    'DistanceTrace', ref_dk, 'ResidualTrace', ref_ek, ...
    'ObjectiveTrace', ref_fk);
save(fullfile(raw_dir, 'constrained_sparse_results.mat'), ...
    'problem', 'para', 'p', 'q', 'r', 'x_ref', 'reference', ...
    'x1', 'its1', 'dk1', 'ek1', 'fk1', 't1', ...
    'x2', 'its2', 'dk2', 'ek2', 'fk2', 't2', ...
    'x4', 'its4', 'dk4', 'ek4', 'fk4', 't4', ...
    'x5', 'its5', 'dk5', 'ek5', 'fk5', 't5', ...
    'results_table', 'main_table', 'diagnostics', '-v7.3');

% The two official Section 4.2 figures remain visible after saving.
linewidth = 1;
axesFontSize = 8;
labelFontSize = 8;
legendFontSize = 8;
colors = {[1/1.75, 1/1.75, 1/1.75], [1/6, 1/6, 1/6], ...
    [1/6, 1/4, 1/1.2], [1, 0, 0]};

%% convergence figure
figure(101), clf;
set(gcf, 'PaperUnits', 'centimeters', ...
    'PaperPosition', [-0.1, 0, 10, 8], 'PaperSize', [9.15, 7.6]);
set(gca, 'FontSize', axesFontSize);
p1d = semilogy(1:its1, dk1, 'Color', colors{1}, 'LineWidth', linewidth);
hold on;
p2d = semilogy(1:its2, dk2, 'Color', colors{2}, 'LineWidth', linewidth);
p4d = semilogy(1:its4, dk4, 'Color', colors{3}, 'LineWidth', linewidth);
p5d = semilogy(1:its5, dk5, 'Color', colors{4}, 'LineWidth', linewidth);
grid on;
ax = gca;
ax.GridLineStyle = '--';
xlim([1, max([its1, its2, its4, its5])]);
ylim([1e-12, 2*max(dk1)]);
xlabel('$k$', 'FontSize', labelFontSize, 'Interpreter', 'latex');
ylabel('$\|x_k-x^\star\|_2$', 'FontSize', labelFontSize, ...
    'Interpreter', 'latex');
lg = legend([p1d, p2d, p4d, p5d], 'FISTA', ...
    'FISTA-Mod, $p = \frac{1}{20}, q = \frac{1}{2}$', 'GAPGA', ...
    'SS-FISTA', 'Location', 'northeast', 'Interpreter', 'latex');
set(lg, 'FontSize', legendFontSize);
legend('boxoff');
convergence_filename = fullfile(figure_dir, 'constrained_sparse_convergence');
print([convergence_filename, '.eps'], '-depsc', '-r600');
print([convergence_filename, '.pdf'], '-dpdf');

%% minimum-coordinate feasibility figure
figure(102), clf;
set(gcf, 'PaperUnits', 'centimeters', ...
    'PaperPosition', [-0.1, 0, 10, 8], 'PaperSize', [9.15, 7.6]);
set(gca, 'FontSize', axesFontSize);
hold on;
keep1 = diagnostics(1).IterationIndex >= 1;
keep2 = diagnostics(2).IterationIndex >= 1;
keep4 = diagnostics(3).IterationIndex >= 1;
keep5 = diagnostics(4).IterationIndex >= 1;
p1m = plot(diagnostics(1).IterationIndex(keep1), ...
    diagnostics(1).IterationMinimumCoordinate(keep1), ...
    'Color', colors{1}, 'LineWidth', linewidth);
p2m = plot(diagnostics(2).IterationIndex(keep2), ...
    diagnostics(2).IterationMinimumCoordinate(keep2), ...
    'Color', colors{2}, 'LineWidth', linewidth);
gapga_marker_indices = 1:5:nnz(keep4);
p4m = plot(diagnostics(3).IterationIndex(keep4), ...
    diagnostics(3).IterationMinimumCoordinate(keep4), '-', ...
    'Color', colors{3}, 'LineWidth', linewidth, 'Marker', 'o', ...
    'MarkerIndices', gapga_marker_indices, 'MarkerSize', 3);
ssfista_marker_indices = 1:5:nnz(keep5);
p5m = plot(diagnostics(4).IterationIndex(keep5), ...
    diagnostics(4).IterationMinimumCoordinate(keep5), '--', ...
    'Color', colors{4}, 'LineWidth', linewidth, 'Marker', 's', ...
    'MarkerIndices', ssfista_marker_indices, 'MarkerSize', 3);
yline(0, 'k--', 'LineWidth', linewidth);
grid on;
ax = gca;
ax.GridLineStyle = '--';
xlim([1, 40]);
xlabel('$k$', 'FontSize', labelFontSize, 'Interpreter', 'latex');
ylabel('Minimum coordinate', 'FontSize', labelFontSize);
box on;
lg = legend([p1m, p2m, p4m, p5m], 'FISTA', ...
    'FISTA-Mod, $p = \frac{1}{20}, q = \frac{1}{2}$', 'GAPGA', ...
    'SS-FISTA', 'Location', 'southeast', 'Interpreter', 'latex');
set(lg, 'FontSize', legendFontSize);
legend('boxoff');
minimum_coordinate_filename = fullfile(figure_dir, 'constrained_sparse_min_coordinate');
print([minimum_coordinate_filename, '.eps'], '-depsc', '-r600');
print([minimum_coordinate_filename, '.pdf'], '-dpdf');
