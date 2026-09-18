%% PCP Lobby example
% This entry point preserves the paper's Lobby formulation and parameters.
% It computes the historical PCP numerical reference locally; no external
% reference MAT file is required or loaded.

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

%% load data
type = 'Lobby';
source_path = fullfile(code_root, 'data', 'pcp', 'Lobby.mat');
data = load(source_path, 'I', 'imSize');
% Lobby.mat is pre-cropped to the original sequence frames 1:300.
frame_window = [1, size(data.I, 2)];
f = data.I;
imSize = data.imSize;
preview_local = 90;
preview_status = 'formal Lobby preview';

n1 = imSize(1);
n2 = imSize(2);
n3 = size(f, 2);
source_identifier = source_path;
window_start = frame_window(1);
window_end = frame_window(2);
frame_count = n3;
assert(size(f, 1) == prod(imSize), 'Input rows do not match imSize for %s.', type);
assert(preview_local >= 1 && preview_local <= n3, ...
    'Preview index is outside the selected window for %s.', type);
preview_global = frame_window(1) + preview_local - 1;

f = double(f);
f = scale_to_interval(f, 0, 1);
n = size(f);

%% parameters
para.x0 = zeros(size(f));
para.n = n;
para.mu = 1 / sqrt(max(size(f)));
para.nu = 2;
para.gamma = 1;
para.tol = 1e-18;
para.maxits = 1000;
para.a = 40;
para.verbose = 1;

p = 1/20;
q = 1/2;
r = 4;
GradF = @(S) -((f - S) - svt(f - S, para.nu));
ProxJ = @(S, t) wthresh(S, 's', t);
ObjPhi = @(S) 1;

window_metadata = struct('frame_window', frame_window, ...
    'window_start', window_start, 'window_end', window_end, ...
    'numFrames', n3, 'source_path', source_path, ...
    'source_imSize', imSize, 'matrix_size', n, ...
    'rescaled_to_unit_interval', true);
preview_metadata = struct('preview_local', preview_local, ...
    'preview_global', preview_global, 'status', preview_status);
problem = struct('dataset', type, 'type', type, 'imSize', imSize, ...
    'source_identifier', source_identifier, 'source_path', source_path, ...
    'frame_window', frame_window, 'window_start', window_start, ...
    'window_end', window_end, 'numFrames', n3, 'frame_count', frame_count, ...
    'matrix_size', n, 'preview_local', preview_local, ...
    'preview_global', preview_global, 'preview_status', preview_status, ...
    'rescaled_to_unit_interval', true);
parameters = para;

results_dir = fullfile(code_root, 'results');
raw_dir = fullfile(results_dir, 'raw');
table_dir = fullfile(results_dir, 'tables');
figures_dir = fullfile(results_dir, 'figures');
figure_dir = fullfile(figures_dir, 'pcp');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
if ~exist(raw_dir, 'dir'), mkdir(raw_dir); end
if ~exist(table_dir, 'dir'), mkdir(table_dir); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end
stem = ['pcp_' lower(type)];
checkpoint_file = fullfile(raw_dir, [stem '_checkpoint.mat']);
result_file = fullfile(raw_dir, [stem '_results.mat']);

%% reference xsol
% This local routine is used only to construct the numerical reference for
% the PCP error curves. It is not one of the four benchmark algorithms.
% Its CPU time and iterations are excluded from the formal comparison.
fprintf('Computing numerical reference with historical PCP recurrence...\n');
reference_para = para;
[x_ref, ref_its, ref_dk, ref_ek, ref_fk, ref_cpu] = ...
    pcp_historical_reference(p, q, r, reference_para, ProxJ, GradF, ObjPhi, 0);
ref_status = 'computed_historical_pcp_reference';
reference = struct('description', ...
    ['In-driver historical PCP SS-FISTA reference; local routine only, ' ...
     'not a benchmark algorithm; no external reference MAT file.'], ...
    'status', ref_status, 'iterations', ref_its, 'cpu_time', ref_cpu, ...
    'tol', reference_para.tol, 'maxits', reference_para.maxits);
assert(isequal(size(x_ref), size(f)), ...
    'Computed reference size mismatch for %s.', type);

%% FISTA-Mod, FISTA, GAPGA, SS-FISTA
Method = {'FISTA-Mod'; 'FISTA'; 'GAPGA'; 'SS-FISTA'};
x1 = []; x2 = []; x4 = []; x5 = [];
L1 = []; L2 = []; L4 = []; L5 = [];
its1 = NaN; its2 = NaN; its4 = NaN; its5 = NaN;
dk1 = []; dk2 = []; dk4 = []; dk5 = [];
ek1 = []; ek2 = []; ek4 = []; ek5 = [];
fk1 = []; fk2 = []; fk4 = []; fk5 = [];
t1 = NaN; t2 = NaN; t4 = NaN; t5 = NaN;
iterations = NaN(4, 1);
CPU = NaN(4, 1);
objective = NaN(4, 1);
gradient_counts = NaN(4, 1);
prox_counts = NaN(4, 1);
status = repmat({'pending'}, 4, 1);
has_nan_inf = false(4, 1);
is_divergent = false(4, 1);

for method_index = 1:4
    if numel(status) >= method_index && ~strcmp(status{method_index}, 'pending') && ...
            isfinite(iterations(method_index))
        fprintf('Skipping completed %s (%s).\n', Method{method_index}, status{method_index});
        continue;
    end
    switch method_index
        case 1
            fprintf('Performing %s...\n', Method{method_index});
            [x2, its2, dk2, ek2, fk2, t2] = ...
                func_FISTA_Mod(p, q, r, para, ProxJ, GradF, ObjPhi, x_ref);
            S_current = x2; its_current = its2; dk_current = dk2;
            ek_current = ek2; fk_current = fk2; cpu_current = t2;
            L2 = svt(f - x2, para.nu); L_current = L2;
        case 2
            fprintf('Performing %s...\n', Method{method_index});
            [x1, its1, dk1, ek1, fk1, t1] = ...
                func_FISTA(para, ProxJ, GradF, ObjPhi, x_ref);
            S_current = x1; its_current = its1; dk_current = dk1;
            ek_current = ek1; fk_current = fk1; cpu_current = t1;
            L1 = svt(f - x1, para.nu); L_current = L1;
        case 3
            fprintf('Performing %s...\n', Method{method_index});
            [x4, its4, dk4, ek4, fk4, t4] = ...
                func_GAPGA(para, ProxJ, GradF, ObjPhi, x_ref);
            S_current = x4; its_current = its4; dk_current = dk4;
            ek_current = ek4; fk_current = fk4; cpu_current = t4;
            L4 = svt(f - x4, para.nu); L_current = L4;
        case 4
            fprintf('Performing %s...\n', Method{method_index});
            [x5, its5, dk5, ek5, fk5, t5] = ...
                func_SSFISTA(para, ProxJ, GradF, ObjPhi, x_ref);
            S_current = x5; its_current = its5; dk_current = dk5;
            ek_current = ek5; fk_current = fk5; cpu_current = t5;
            L5 = svt(f - x5, para.nu); L_current = L5;
    end

    iterations(method_index) = its_current;
    CPU(method_index) = cpu_current;
    objective(method_index) = 0.5 * norm(f - L_current - S_current, 'fro')^2 + ...
        para.mu * sum(abs(S_current(:))) + para.nu * sum(svd(L_current, 'econ'));
    gradient_counts(method_index) = its_current;
    if method_index == 3, gradient_counts(method_index) = 1 + 2 * its_current; end
    prox_counts(method_index) = gradient_counts(method_index);
    status{method_index} = 'maxits';
    if ek_current(end) / prod(n) < para.tol, status{method_index} = 'tol'; end
    if ek_current(end) > 1e10, status{method_index} = 'divergence'; end
    if any(~isfinite(S_current(:))) || any(~isfinite(dk_current)) || ...
            any(~isfinite(ek_current)) || ...
            any(~isfinite(fk_current))
        status{method_index} = 'nan_inf';
    end
    has_nan_inf(method_index) = strcmp(status{method_index}, 'nan_inf');
    is_divergent(method_index) = strcmp(status{method_index}, 'divergence');

    % Every completed method updates the same recoverable checkpoint.
    save(checkpoint_file, 'problem', 'window_metadata', 'preview_metadata', ...
        'type', 'source_identifier', 'source_path', 'imSize', 'f', ...
        'frame_window', 'window_start', 'window_end', 'frame_count', ...
        'preview_local', 'preview_global', ...
        'para', 'parameters', 'reference_para', 'reference', 'x_ref', ...
        'ref_its', 'ref_cpu', 'ref_dk', 'ref_ek', 'ref_fk', 'ref_status', ...
        'Method', 'iterations', 'CPU', 'objective', 'gradient_counts', ...
        'prox_counts', 'status', 'has_nan_inf', 'is_divergent', ...
        'x1', 'x2', 'x4', 'x5', 'L1', 'L2', 'L4', 'L5', ...
        'its1', 'its2', 'its4', 'its5', 'dk1', 'dk2', 'dk4', 'dk5', ...
        'ek1', 'ek2', 'ek4', 'ek5', 'fk1', 'fk2', 'fk4', 'fk5', ...
        't1', 't2', 't4', 't5', '-v7.3');
end

%% results
results_table = table(Method, iterations, CPU, objective, gradient_counts, ...
    prox_counts, status, has_nan_inf, is_divergent);
save(result_file, 'problem', 'window_metadata', 'preview_metadata', ...
    'type', 'source_identifier', 'source_path', 'imSize', 'f', ...
    'frame_window', 'window_start', 'window_end', 'frame_count', ...
    'preview_local', 'preview_global', ...
    'para', 'parameters', 'reference_para', 'reference', 'x_ref', ...
    'ref_its', 'ref_cpu', 'ref_dk', 'ref_ek', 'ref_fk', 'ref_status', ...
    'Method', 'iterations', 'CPU', 'objective', 'gradient_counts', ...
    'prox_counts', 'status', 'has_nan_inf', 'is_divergent', ...
    'results_table', 'x1', 'x2', 'x4', 'x5', 'L1', 'L2', 'L4', 'L5', ...
    'its1', 'its2', 'its4', 'its5', 'dk1', 'dk2', 'dk4', 'dk5', ...
    'ek1', 'ek2', 'ek4', 'ek5', 'fk1', 'fk2', 'fk4', 'fk5', ...
    't1', 't2', 't4', 't5', '-v7.3');

disp(results_table);

%% validate results
% The validation is intentionally based on variable identity (x1/x2/x4/x5),
% rather than the internal Method row order.
final_required = {'Method', 'x_ref', 'x1', 'x2', 'x4', 'x5', 'L1', 'L2', 'L4', 'L5', ...
    'dk1', 'dk2', 'dk4', 'dk5', 'ek1', 'ek2', 'ek4', 'ek5', ...
    'fk1', 'fk2', 'fk4', 'fk5', 'its1', 'its2', 'its4', 'its5', ...
    't1', 't2', 't4', 't5', 'iterations', 'CPU', 'objective', 'status', ...
    'results_table'};
final_check = load(result_file, final_required{:});
for field_index = 1:numel(final_required)
    field_name = final_required{field_index};
    field_value = final_check.(field_name);
    if isnumeric(field_value) && ~all(isfinite(field_value(:)))
        error('Final PCP validation failed: non-finite field %s.', field_name);
    end
end
if any(strcmp(final_check.status, 'pending'))
    error('Final PCP validation failed: pending method status remains.');
end
if any(strcmp(final_check.status, 'nan_inf')) || ...
        any(strcmp(final_check.status, 'divergence'))
    error('Final PCP validation failed: invalid method status remains.');
end
if ~istable(final_check.results_table)
    error('Final PCP validation failed: results_table is missing or invalid.');
end
fprintf('Final PCP reload validation PASS for %s.\n', type);

% A completed final package supersedes its large recovery checkpoint.
if exist(checkpoint_file, 'file') == 2
    delete(checkpoint_file);
    fprintf('Deleted successful checkpoint: %s\n', checkpoint_file);
end

%% save summary table

formal_method = {'FISTA'; 'FISTA-Mod'; 'GAPGA'; 'SS-FISTA'};
formal_iterations = [final_check.its1; final_check.its2; ...
    final_check.its4; final_check.its5];
formal_cpu = [final_check.t1; final_check.t2; final_check.t4; final_check.t5];
formal_error = [final_check.dk1(end); final_check.dk2(end); ...
    final_check.dk4(end); final_check.dk5(end)];
formal_gradient = [final_check.its1; final_check.its2; ...
    1 + 2 * final_check.its4; final_check.its5];
formal_prox = formal_gradient;

% Status/count fields in the MAT retain the execution order. Extract them by
% method name so that the CSV order remains independent of that implementation
% detail.
idx_fista = find(strcmp(final_check.Method, 'FISTA'), 1);
idx_mod = find(strcmp(final_check.Method, 'FISTA-Mod'), 1);
idx_gapga = find(strcmp(final_check.Method, 'GAPGA'), 1);
idx_ssfista = find(strcmp(final_check.Method, 'SS-FISTA'), 1);
formal_status = final_check.status([idx_fista; idx_mod; idx_gapga; idx_ssfista]);

pcp_table = table(formal_method, formal_iterations, formal_cpu, formal_error, ...
    formal_gradient, formal_prox, formal_status, ...
    'VariableNames', {'Method', 'Iterations', 'CPU', 'Error', ...
    'GradientEvaluations', 'ProxEvaluations', 'Status'});
csv_file = fullfile(table_dir, [stem '_results.csv']);
writetable(pcp_table, csv_file);
disp(pcp_table);

%% convergence figure
linewidth = 1;
axesFontSize = 8;
labelFontSize = 8;
legendFontSize = 8;
h_conv = figure('Visible', 'on', 'Units', 'centimeters', ...
    'Position', [2 2 10 8], 'Color', 'w');
p1d = semilogy(final_check.dk1, 'Color', ...
    [1/1.75, 1/1/1.75, 1/1/1.75], 'LineWidth', linewidth);
hold on;
p2d = semilogy(final_check.dk2, 'Color', [1/6, 1/6, 1/6], ...
    'LineWidth', linewidth);
p4d = semilogy(final_check.dk4, 'Color', [1/6, 1/4, 1/1.2], ...
    'LineWidth', linewidth);
p5d = semilogy(final_check.dk5, 'Color', [1, 0, 0], ...
    'LineWidth', linewidth);
uistack(p1d, 'bottom');
grid on;
ax = gca;
ax.GridLineStyle = '--';
axis([1, final_check.its1, min(final_check.dk5), ...
    2 * max(final_check.dk1)]);
set(gca, 'YTick', [1e-12, 1e-8, 1e-4, 1e-0, 1e4], ...
    'FontSize', axesFontSize);
ylabel({'$\|x_{k}-x^\star\|$'}, 'Interpreter', 'latex', ...
    'FontSize', labelFontSize);
xlabel({'$k$'}, 'Interpreter', 'latex', 'FontSize', labelFontSize);
lg = legend([p1d, p2d, p4d, p5d], ...
    'FISTA', ...
    'FISTA-Mod, $p = \frac{1}{20}, q = \frac{1}{2}$', ...
    'GAPGA', 'SS-FISTA');
set(lg, 'FontSize', legendFontSize, 'Interpreter', 'latex');
legend('boxoff');
set(h_conv, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 10 8], ...
    'PaperSize', [10 8]);
conv_prefix = fullfile(figure_dir, [stem '_convergence']);
print(h_conv, [conv_prefix '.eps'], '-depsc', '-r300');
print(h_conv, [conv_prefix '.pdf'], '-dpdf');

%% preview figures
if ~isnan(preview_local)
    preview_names = {'original', 'sparse', 'low_rank'};
    preview_values = {f(:, preview_local), ...
        final_check.x5(:, preview_local), final_check.L5(:, preview_local)};
    for preview_index = 1:numel(preview_values)
        h_preview = figure('Visible', 'on', 'Units', 'centimeters', ...
            'Position', [2 2 8 8], 'Color', 'w');
        % imgsc is the historical helper: imagesc, gray colormap,
        % axis image, and axis off, with no titles or numerical overlays.
        imgsc(reshape(preview_values{preview_index}, imSize));
        set(h_preview, 'PaperUnits', 'centimeters', ...
            'PaperPosition', [0 0 8 8], 'PaperSize', [8 8]);
        preview_prefix = fullfile(figure_dir, ...
            [stem '_' preview_names{preview_index}]);
        print(h_preview, [preview_prefix '.eps'], '-deps', '-r300');
        print(h_preview, [preview_prefix '.pdf'], '-dpdf');
    end
else
    fprintf('%s visual preview pending.\n', type);
end

function [x, its, dk, ek, fk, time] = ...
    pcp_historical_reference(p, q, r, para, ProxJ, GradF, ObjPhi, xsol)
% PCP historical numerical-reference generator only.
%
% This recurrence reproduces the reference protocol used by the original
% PCP driver.  It must not be confused with the current corrected
% func_SSFISTA benchmark implementation.  The p/q/r arguments are retained
% solely to preserve the historical call signature; the recurrence itself
% uses a = 40 as in the original reference run.
 %#ok<INUSD>
clock_id = tic;
n = para.n;
mu = para.mu;
gamma = para.gamma;
tol = para.tol;
maxits = para.maxits;
verbose = para.verbose;
a = 40;

x = para.x0;
W = x;
dk = zeros(maxits, 1);
ek = zeros(maxits, 1);
fk = zeros(maxits, 1);

if verbose
    itsprint(sprintf('        step %08d: residual = %.3e', 1, 1), 1);
end
for its = 1:maxits
    fk(its) = ObjPhi(x);
    dk(its) = norm(x - xsol, 'fro');

    x_old = x;
    thetak = a / (its + a);
    ak = 1 - 1 / (a * (its + a)^2);
    y = thetak * W + (1 - thetak) * x;
    W = ProxJ(W - gamma / thetak * GradF(y), ...
        mu * gamma / thetak);
    x = (1 - ak * thetak) * x_old + ak * thetak * W;

    res = norm(x_old - x, 'fro');
    ek(its) = res;
    if verbose && mod(its, 10) == 0
        itsprint(sprintf('        step %08d: residual = %.3e', its, res), its);
    end
    if (res / prod(n) < tol) || (res > 1e10)
        break;
    end
end
fprintf('\n');
dk = dk(1:its);
ek = ek(1:its);
fk = fk(1:its);
time = toc(clock_id);
end
