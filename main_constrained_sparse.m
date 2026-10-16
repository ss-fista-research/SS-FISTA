%% Section 4.2: feasibility-constrained sparse recovery

clear;
clc;
code_root = fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(fullfile(code_root, 'toolbox'));
addpath(fullfile(code_root, 'menthod'));
formal_method_names = {'func_FISTA', 'func_FISTA_Mod', ...
    'func_MGFISTA', 'func_NAGEL', 'func_GAPGA', 'func_SSFISTA'};
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

%% FISTA
fprintf('\n----------------------------------------\n');
fprintf('Performing FISTA...\n');
fprintf('----------------------------------------\n');
[x1, its1, dk1, ek1, fk1, t1] = ...
    func_FISTA(para, proxJ, gradF, objPhi, x_ref);
fprintf('FISTA completed: iterations = %d, CPU = %.4f s\n', its1, t1);

%% FISTA-Mod
fprintf('\n----------------------------------------\n');
fprintf('Performing FISTA-Mod...\n');
fprintf('----------------------------------------\n');
[x2, its2, dk2, ek2, fk2, t2] = ...
    func_FISTA_Mod(p, q, r, para, proxJ, gradF, objPhi, x_ref);
fprintf('FISTA-Mod completed: iterations = %d, CPU = %.4f s\n', its2, t2);

%% MG-FISTA
fprintf('Performing MG-FISTA...\n');
[x6, its6, dk6, ek6, fk6, t6] = func_MGFISTA(para, proxJ, gradF, objPhi, x_ref);
fprintf('\n');

%% NAGEL
fprintf('Performing NAGEL...\n');
[x7, its7, dk7, ek7, fk7, t7] = func_NAGEL(para, proxJ, gradF, objPhi, x_ref);
fprintf('\n');

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
Method = {'FISTA'; 'FISTA-Mod'; 'GAPGA'; 'MG-FISTA'; 'NAGEL'; 'SS-FISTA'};
Iterations = [its1; its2; its4; its6; its7; its5];
CPUTime = [t1; t2; t4; t6; t7; t5];
FinalDistanceToReference = [norm(x1 - x_ref, 2); norm(x2 - x_ref, 2); ...
    norm(x4 - x_ref, 2); norm(x6 - x_ref, 2); norm(x7 - x_ref, 2); norm(x5 - x_ref, 2)];
FinalObjective = [objPhi(x1); objPhi(x2); objPhi(x4); objPhi(x6); objPhi(x7); objPhi(x5)];
RelativeRecoveryError = [norm(x1 - x_true, 2); norm(x2 - x_true, 2); ...
    norm(x4 - x_true, 2); norm(x6 - x_true, 2); norm(x7 - x_true, 2); norm(x5 - x_true, 2)] / norm(x_true, 2);
NNZ = [nnz(abs(x1) > 1e-8); nnz(abs(x2) > 1e-8); ...
    nnz(abs(x4) > 1e-8); nnz(abs(x6) > 1e-8); nnz(abs(x7) > 1e-8); nnz(abs(x5) > 1e-8)];
GradientEvaluations = [its1; its2; 1 + 2*its4; its6; 2*its7; its5];
ProxEvaluations = GradientEvaluations;
results_table = table(Method, Iterations, CPUTime, FinalDistanceToReference, ...
    FinalObjective, RelativeRecoveryError, NNZ, GradientEvaluations, ProxEvaluations);

%% independent feasibility diagnostics (excluded from formal CPU timing)
diagnostics = run_constrained_sparse_diagnostics(para, proxJ, gradF, objPhi, x_ref, p, q, r, [its1; its2; its4; its6; its7; its5], {x1; x2; x4; x6; x7; x5});

disp([its1, its2, its4, its6, its7, its5])
disp([t1, t2, t4, t6, t7, t5])

%% plotting and save
% Cross-check the static method-call counts against the query counts from the
% independent diagnostic pass before producing any official outputs.
DiagnosticGradientEvaluations = [ ...
    diagnostics(1).NumberOfGradientEvaluations; ...
    diagnostics(2).NumberOfGradientEvaluations; ...
    diagnostics(3).NumberOfGradientEvaluations; ...
    diagnostics(4).NumberOfGradientEvaluations; ...
    diagnostics(5).NumberOfGradientEvaluations; ...
    diagnostics(6).NumberOfGradientEvaluations];
DiagnosticProxEvaluations = [ ...
    diagnostics(1).NumberOfProximalEvaluations; ...
    diagnostics(2).NumberOfProximalEvaluations; ...
    diagnostics(3).NumberOfProximalEvaluations; ...
    diagnostics(4).NumberOfProximalEvaluations; ...
    diagnostics(5).NumberOfProximalEvaluations; ...
    diagnostics(6).NumberOfProximalEvaluations];
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
    diagnostics(4).InfeasibleGradientPointCount; ...
    diagnostics(5).InfeasibleGradientPointCount; ...
    diagnostics(6).InfeasibleGradientPointCount];
MaxFeasibilityViolation = [ ...
    diagnostics(1).MaxFeasibilityViolation; ...
    diagnostics(2).MaxFeasibilityViolation; ...
    diagnostics(3).MaxFeasibilityViolation; ...
    diagnostics(4).MaxFeasibilityViolation; ...
    diagnostics(5).MaxFeasibilityViolation; ...
    diagnostics(6).MaxFeasibilityViolation];
MinimumGradientPointCoordinate = [ ...
    diagnostics(1).MinimumGradientPointCoordinate; ...
    diagnostics(2).MinimumGradientPointCoordinate; ...
    diagnostics(3).MinimumGradientPointCoordinate; ...
    diagnostics(4).MinimumGradientPointCoordinate; ...
    diagnostics(5).MinimumGradientPointCoordinate; ...
    diagnostics(6).MinimumGradientPointCoordinate];

% Detailed Section 4.2 record and the compact manuscript-facing table.
results_table = table(Method, Iterations, CPUTime, FinalDistanceToReference, ...
    FinalObjective, RelativeRecoveryError, NNZ, GradientEvaluations, ...
    ProxEvaluations, MaxFeasibilityViolation, MinimumGradientPointCoordinate, ...
    InfeasibleGradientQueries);
main_table = table(Method, Iterations, GradientEvaluations, ProxEvaluations, ...
    CPUTime, InfeasibleGradientQueries);

results_table = results_table([1,2,4,5,3,6],:);
main_table = main_table([1,2,4,5,3,6],:);
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
    'x6', 'x7', 'x5', 'its6', 'its7', 'its5', 'dk6', 'dk7', 'dk5', 'ek6', 'ek7', 'ek5', 'fk6', 'fk7', 'fk5', 't6', 't7', 't5', ...
    'results_table', 'main_table', 'diagnostics', '-v7.3');

% The two official Section 4.2 figures remain visible after saving.
linewidth = 1;
axesFontSize = 8;
labelFontSize = 8;
legendFontSize = 8;
colors = {[1/1.75, 1/1.75, 1/1.75], [1/6, 1/6, 1/6], ...
    [1/6, 1/4, 1/1.2], [0.494, 0.184, 0.556], [0.929, 0.694, 0.125], [1, 0, 0]};

%% convergence figure
figure(101), clf;
set(gcf, 'PaperUnits', 'centimeters', ...
    'PaperPosition', [-0.1, 0, 10, 8], 'PaperSize', [9.15, 7.6]);
set(gca, 'FontSize', axesFontSize);
p1d = semilogy(1:its1, dk1, 'Color', colors{1}, 'LineWidth', linewidth);
hold on;
p2d = semilogy(1:its2, dk2, 'Color', colors{2}, 'LineWidth', linewidth);
p4d = semilogy(1:its4, dk4, 'Color', colors{3}, 'LineWidth', linewidth);
p6d = semilogy(1:its6, dk6, 'Color', colors{4}, 'LineWidth', linewidth);
p7d = semilogy(1:its7, dk7, 'Color', colors{5}, 'LineWidth', linewidth);
p5d = semilogy(1:its5, dk5, 'Color', colors{6}, 'LineWidth', linewidth);
grid on;
ax = gca;
ax.GridLineStyle = '--';
xlim([1, max([its1, its2, its4, its6, its7, its5])]);
ylim([1e-12, 2*max(dk1)]);
xlabel('Iteration', 'FontSize', labelFontSize, 'Interpreter', 'latex');
ylabel('$\|x_k-x^\star\|_2$', 'FontSize', labelFontSize, ...
    'Interpreter', 'latex');
lg = legend([p1d, p2d, p6d, p7d, p4d, p5d], ...
    'FISTA', 'FISTA-Mod', 'MG-FISTA', 'NAGEL', 'GAPGA', 'SS-FISTA', ...
    'Location', 'northeast', 'Interpreter', 'latex');
set(lg, 'FontSize', legendFontSize);
legend('boxoff');
convergence_filename = fullfile(figure_dir, 'constrained_sparse_convergence');
print([convergence_filename, '.eps'], '-depsc', '-r600');
print([convergence_filename, '.pdf'], '-dpdf');
print([convergence_filename, '.png'], '-dpng', '-r180');

%% minimum-coordinate feasibility figure
figure(102), clf;
set(gcf, 'PaperUnits', 'centimeters', ...
    'PaperPosition', [-0.1, 0, 10, 8], 'PaperSize', [9.15, 7.6]);
set(gca, 'FontSize', axesFontSize);
hold on;
keep1 = diagnostics(1).IterationIndex >= 1;
keep2 = diagnostics(2).IterationIndex >= 1;
keep4 = diagnostics(3).IterationIndex >= 1;
keep6 = diagnostics(4).IterationIndex >= 1;
keep7 = diagnostics(5).IterationIndex >= 1;
keep5 = diagnostics(6).IterationIndex >= 1;
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
p6m = plot(diagnostics(4).IterationIndex(keep6), diagnostics(4).IterationMinimumCoordinate(keep6), ...
    'Color',colors{4},'LineWidth',linewidth);
p7m = plot(diagnostics(5).IterationIndex(keep7), diagnostics(5).IterationMinimumCoordinate(keep7), ...
    'Color',colors{5},'LineWidth',linewidth);
ssfista_marker_indices = 1:5:nnz(keep5);
p5m = plot(diagnostics(6).IterationIndex(keep5), ...
    diagnostics(6).IterationMinimumCoordinate(keep5), '--', ...
    'Color', colors{6}, 'LineWidth', linewidth, 'Marker', 's', ...
    'MarkerIndices', ssfista_marker_indices, 'MarkerSize', 3);
yline(0, 'k--', 'LineWidth', linewidth);
grid on;
ax = gca;
ax.GridLineStyle = '--';
xlim([1, 40]);
xlabel('Iteration', 'FontSize', labelFontSize, 'Interpreter', 'latex');
ylabel('Minimum coordinate', 'FontSize', labelFontSize);
box on;
lg = legend([p1m, p2m, p6m, p7m, p4m, p5m], ...
    'FISTA', 'FISTA-Mod', 'MG-FISTA', 'NAGEL', 'GAPGA', 'SS-FISTA', ...
    'Location', 'southeast', 'Interpreter', 'latex');
set(lg, 'FontSize', legendFontSize);
legend('boxoff');
minimum_coordinate_filename = fullfile(figure_dir, 'constrained_sparse_min_coordinate');
print([minimum_coordinate_filename, '.eps'], '-depsc', '-r600');
print([minimum_coordinate_filename, '.pdf'], '-dpdf');
print([minimum_coordinate_filename, '.png'], '-dpng', '-r180');

function diagnostics = run_constrained_sparse_diagnostics(para, proxJ, baseGradF, objPhi, x_ref, p, q, r, formal_iterations, formal_iterates)
%RUN_CONSTRAINED_SPARSE_DIAGNOSTICS Independent query-level feasibility audit.
%   This helper reruns the six current formal methods using an exact logging
%   wrapper around baseGradF.  It is deliberately a diagnostic-only pass: its
%   timings are neither returned nor used by the formal Section 4.2 results.
%   No files are written here.

method_names = {'FISTA'; 'FISTA-Mod'; 'GAPGA'; 'MG-FISTA'; 'NAGEL'; 'SS-FISTA'};
infeasibility_threshold = -1e-12;

if ~isequal(size(formal_iterations), [6, 1])
    error('SSFISTA:InvalidFormalIterations', ...
        'formal_iterations must be a 6-by-1 vector in FISTA, FISTA-Mod, GAPGA, MG-FISTA, NAGEL, SS-FISTA order.');
end
if ~iscell(formal_iterates) || numel(formal_iterates) ~= 6
    error('SSFISTA:InvalidFormalIterates', ...
        'formal_iterates must be a six-element cell array in formal method order.');
end

% Verify before recording that the wrapper returns the unchanged gradient.
record_queries = false;
query_count = 0;
query_capacity = 0;
query_points = [];
query_violation = [];
query_min_coordinate = [];
test_points = {para.x0, zeros(para.n, 1), ones(para.n, 1) / para.n};
for test_index = 1:numel(test_points)
    v = test_points{test_index};
    if ~isequal(logged_grad(v), baseGradF(v))
        error('SSFISTA:LoggedGradientMismatch', ...
            'The diagnostic gradient differs from baseGradF at test point %d.', test_index);
    end
end

diagnostics = repmat(empty_diagnostic(), 6, 1);

begin_trace(para.maxits);
[x, its] = run_silently(@() func_FISTA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(1) = complete_trace(method_names{1}, its, x, formal_iterations(1), formal_iterates{1});

begin_trace(para.maxits);
[x, its] = run_silently(@() func_FISTA_Mod(p, q, r, para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(2) = complete_trace(method_names{2}, its, x, formal_iterations(2), formal_iterates{2});

begin_trace(para.maxits);
[x, its] = run_silently(@() func_MGFISTA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(4) = complete_trace(method_names{4}, its, x, formal_iterations(4), formal_iterates{4});

begin_trace(2*para.maxits);
[x, its] = run_silently(@() func_NAGEL(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(5) = complete_trace(method_names{5}, its, x, formal_iterations(5), formal_iterates{5});

begin_trace(1 + 2 * para.maxits);
[x, its] = run_silently(@() func_GAPGA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(3) = complete_trace(method_names{3}, its, x, formal_iterations(3), formal_iterates{3});

begin_trace(para.maxits);
[x, its] = run_silently(@() func_SSFISTA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(6) = complete_trace(method_names{6}, its, x, formal_iterations(6), formal_iterates{6});

    function grad = logged_grad(v)
        grad = baseGradF(v);
        if record_queries
            query_count = query_count + 1;
            if query_count > query_capacity
                error('SSFISTA:TraceCapacityExceeded', ...
                    'The diagnostic trace exceeded its allocated capacity.');
            end
            query_points(:, query_count) = v;
            query_violation(query_count, 1) = norm(min(v, 0), 2);
            query_min_coordinate(query_count, 1) = min(v);
        end
    end

    function begin_trace(capacity)
        record_queries = false;
        query_count = 0;
        query_capacity = capacity;
        query_points = zeros(para.n, capacity);
        query_violation = zeros(capacity, 1);
        query_min_coordinate = zeros(capacity, 1);
        record_queries = true;
    end

    function trace = complete_trace(name, iterations, final_x, expected_iterations, formal_x)
        record_queries = false;
        expected_queries = iterations;
        if strcmp(name, 'GAPGA')
            expected_queries = 1 + 2 * iterations;
        elseif strcmp(name,'NAGEL')
            expected_queries = 2*iterations;
        end
        if query_count ~= expected_queries
            error('SSFISTA:UnexpectedGradientCount', ...
                '%s made %d gradient calls; expected %d.', name, query_count, expected_queries);
        end
        if iterations ~= expected_iterations
            error('SSFISTA:DiagnosticIterationMismatch', ...
                '%s diagnostic pass used %d iterations; formal pass used %d.', ...
                name, iterations, expected_iterations);
        end

        trace = empty_diagnostic();
        trace.Method = name;
        trace.Iterations = iterations;
        trace.GradientPoints = query_points(:, 1:query_count);
        trace.FeasibilityViolation = query_violation(1:query_count);
        trace.MinimumCoordinate = query_min_coordinate(1:query_count);
        trace.InfeasibleMask = trace.MinimumCoordinate < infeasibility_threshold;
        trace.InfeasibilityThreshold = infeasibility_threshold;
        trace.NumberOfGradientEvaluations = query_count;
        trace.NumberOfProximalEvaluations = query_count;
        trace.MaxFeasibilityViolation = max(trace.FeasibilityViolation);
        trace.MinimumGradientPointCoordinate = min(trace.MinimumCoordinate);
        trace.InfeasibleGradientPointCount = nnz(trace.InfeasibleMask);
        trace.FinalXDifference = norm(final_x - formal_x, 2);
        trace.FinalXMatchTolerance = 100 * eps(max([1, norm(final_x, 2), norm(formal_x, 2)]));
        trace.FinalXMatchesFormal = trace.FinalXDifference <= trace.FinalXMatchTolerance;
        if ~trace.FinalXMatchesFormal
            error('SSFISTA:DiagnosticFinalPointMismatch', ...
                '%s diagnostic pass does not reproduce the formal final point.', name);
        end

        if strcmp(name, 'GAPGA')
            trace.IterationIndex = [0; (1:iterations)'];
            trace.IterationFeasibilityViolation = [trace.FeasibilityViolation(1); ...
                max(reshape(trace.FeasibilityViolation(2:end), 2, []), [], 1)'];
            trace.IterationMinimumCoordinate = [trace.MinimumCoordinate(1); ...
                min(reshape(trace.MinimumCoordinate(2:end), 2, []), [], 1)'];
        elseif strcmp(name,'NAGEL')
            trace.IterationIndex = (1:iterations)';
            trace.IterationFeasibilityViolation = max(reshape(trace.FeasibilityViolation,2,[]),[],1)';
            trace.IterationMinimumCoordinate = min(reshape(trace.MinimumCoordinate,2,[]),[],1)';
        else
            trace.IterationIndex = (1:iterations)';
            trace.IterationFeasibilityViolation = trace.FeasibilityViolation;
            trace.IterationMinimumCoordinate = trace.MinimumCoordinate;
        end
    end
end

function trace = empty_diagnostic()
trace = struct( ...
    'Method', '', ...
    'Iterations', NaN, ...
    'GradientPoints', [], ...
    'FeasibilityViolation', [], ...
    'MinimumCoordinate', [], ...
    'InfeasibleMask', [], ...
    'InfeasibilityThreshold', NaN, ...
    'NumberOfGradientEvaluations', NaN, ...
    'NumberOfProximalEvaluations', NaN, ...
    'MaxFeasibilityViolation', NaN, ...
    'MinimumGradientPointCoordinate', NaN, ...
    'InfeasibleGradientPointCount', NaN, ...
    'FinalXDifference', NaN, ...
    'FinalXMatchTolerance', NaN, ...
    'FinalXMatchesFormal', false, ...
    'IterationIndex', [], ...
    'IterationFeasibilityViolation', [], ...
    'IterationMinimumCoordinate', []);
end

function varargout = run_silently(f)
% Preserve verbose distance/residual traces inside the frozen methods while
% suppressing their command-window progress during the diagnostic-only pass.
captured_output = evalc('[varargout{1:nargout}] = f();'); %#ok<NASGU>
end
