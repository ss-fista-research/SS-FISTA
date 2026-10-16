%% PCP video example
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
clear data;
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
result_file = fullfile(raw_dir, [stem '_results.mat']);

%% reference xsol
% This local routine is used only to construct the numerical reference for
% the PCP error curves. It is not one of the six benchmark algorithms.
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

%% FISTA
fprintf('Performing FISTA...\n');
[x1, its1, dk1, ek1, fk1, t1] = ...
    func_FISTA(para, ProxJ, GradF, ObjPhi, x_ref);
L1 = svt(f - x1, para.nu);
fprintf('FISTA completed: iterations = %d, CPU = %.4f s\n', its1, t1);

%% FISTA-Mod
fprintf('Performing FISTA-Mod...\n');
[x2, its2, dk2, ek2, fk2, t2] = ...
    func_FISTA_Mod(p, q, r, para, ProxJ, GradF, ObjPhi, x_ref);
L2 = svt(f - x2, para.nu);
fprintf('FISTA-Mod completed: iterations = %d, CPU = %.4f s\n', its2, t2);

%% MG-FISTA
fprintf('Performing MG-FISTA...\n');
[x6, its6, dk6, ek6, fk6, t6] = ...
    func_MGFISTA(para, ProxJ, GradF, ObjPhi, x_ref);
L6 = svt(f - x6, para.nu);
fprintf('MG-FISTA completed: iterations = %d, CPU = %.4f s\n', its6, t6);

%% NAGEL
fprintf('Performing NAGEL...\n');
[x7, its7, dk7, ek7, fk7, t7] = ...
    func_NAGEL(para, ProxJ, GradF, ObjPhi, x_ref);
L7 = svt(f - x7, para.nu);
fprintf('NAGEL completed: iterations = %d, CPU = %.4f s\n', its7, t7);

%% GAPGA
fprintf('Performing GAPGA...\n');
[x4, its4, dk4, ek4, fk4, t4] = ...
    func_GAPGA(para, ProxJ, GradF, ObjPhi, x_ref);
L4 = svt(f - x4, para.nu);
fprintf('GAPGA completed: iterations = %d, CPU = %.4f s\n', its4, t4);

%% SS-FISTA
fprintf('Performing SS-FISTA...\n');
[x5, its5, dk5, ek5, fk5, t5] = ...
    func_SSFISTA(para, ProxJ, GradF, ObjPhi, x_ref);
L5 = svt(f - x5, para.nu);
fprintf('SS-FISTA completed: iterations = %d, CPU = %.4f s\n', its5, t5);

%% results
Method = {'FISTA'; 'FISTA-Mod'; 'MG-FISTA'; 'NAGEL'; 'GAPGA'; 'SS-FISTA'};
iterations = [its1; its2; its6; its7; its4; its5];
CPU = [t1; t2; t6; t7; t4; t5];
solution_error = [dk1(end); dk2(end); dk6(end); dk7(end); dk4(end); dk5(end)];
gradient_counts = [its1; its2; its6; 2*its7; 1+2*its4; its5];
prox_counts = gradient_counts;
last_changes = [ek1(end); ek2(end); ek6(end); ek7(end); ek4(end); ek5(end)];
status = repmat({'maxits'},6,1);
has_nan_inf = ~isfinite(solution_error) | ~isfinite(last_changes);
is_divergent = last_changes>1e10;
for j = 1:6
    if last_changes(j)/prod(n)<para.tol, status{j}='tol'; end
    if is_divergent(j), status{j}='divergence'; end
    if has_nan_inf(j), status{j}='nan_inf'; end
end
pcp_table = table(Method,iterations,CPU,solution_error,gradient_counts,prox_counts,status, ...
    'VariableNames',{'Method','Iterations','CPU','Error','GradientEvaluations','ProxEvaluations','Status'});
disp(pcp_table);

%% save final results once; do not reload all result matrices
writetable(pcp_table,fullfile(table_dir,[stem '_results.csv']));
save(result_file, 'problem', 'window_metadata', 'preview_metadata', ...
    'type', 'source_identifier', 'source_path', 'imSize', 'f', ...
    'frame_window', 'window_start', 'window_end', 'frame_count', ...
    'preview_local', 'preview_global', ...
    'para', 'parameters', 'reference_para', 'reference', 'x_ref', ...
    'ref_its', 'ref_cpu', 'ref_dk', 'ref_ek', 'ref_fk', 'ref_status', ...
    'Method', 'iterations', 'CPU', 'gradient_counts', ...
    'prox_counts', 'status', 'has_nan_inf', 'is_divergent', ...
    'x1', 'x2', 'x6', 'x7', 'x4', 'x5', ...
    'L1', 'L2', 'L6', 'L7', 'L4', 'L5', ...
    'its1', 'its2', 'its6', 'its7', 'its4', 'its5', ...
    'dk1', 'dk2', 'dk6', 'dk7', 'dk4', 'dk5', ...
    'ek1', 'ek2', 'ek6', 'ek7', 'ek4', 'ek5', ...
    'fk1', 'fk2', 'fk6', 'fk7', 'fk4', 'fk5', ...
    't1', 't2', 't6', 't7', 't4', 't5', '-v7.3');
fprintf('Saved final PCP result: %s\n',result_file);

%% plotting (numeric results have already been saved)
figure_dir = fullfile(code_root, 'results', 'figures', 'pcp');
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end
%% convergence figure
linewidth = 1;
axesFontSize = 8;
labelFontSize = 8;
legendFontSize = 8;
h_conv = figure('Visible', 'on', 'Units', 'centimeters', ...
    'Position', [2 2 10 8], 'Color', 'w', 'Renderer', 'painters');
p1d = semilogy(dk1, 'Color', ...
    [1/1.75, 1/1/1.75, 1/1/1.75], 'LineWidth', linewidth);
hold on;
p2d = semilogy(dk2, 'Color', [1/6, 1/6, 1/6], ...
    'LineWidth', linewidth);
p4d = semilogy(dk4, 'Color', [1/6, 1/4, 1/1.2], ...
    'LineWidth', linewidth);
p6d = semilogy(dk6,'Color',[0.494 0.184 0.556],'LineWidth',linewidth);
p7d = semilogy(dk7,'Color',[0.929 0.694 0.125],'LineWidth',linewidth);
p5d = semilogy(dk5, 'Color', [1, 0, 0], ...
    'LineWidth', linewidth);
uistack(p1d, 'bottom');
grid on;
ax = gca;
ax.GridLineStyle = '--';
axis([1, its1, min(dk5), ...
    2 * max(dk1)]);
set(gca, 'YTick', [1e-12, 1e-8, 1e-4, 1e-0, 1e4], ...
    'FontSize', axesFontSize);
ylabel({'$\|x_{k}-x^\star\|$'}, 'Interpreter', 'latex', ...
    'FontSize', labelFontSize);
xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', labelFontSize);
lg = legend([p1d, p2d, p6d, p7d, p4d, p5d], ...
    'FISTA', 'FISTA-Mod', 'MG-FISTA', 'NAGEL', 'GAPGA', 'SS-FISTA');
set(lg, 'FontSize', legendFontSize, 'Interpreter', 'latex');
legend('boxoff');
disableDefaultInteractivity(gca);
set(h_conv, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 10 8], ...
    'PaperSize', [10 8]);
conv_prefix = fullfile(figure_dir, [stem '_convergence']);
print(h_conv, [conv_prefix '.eps'], '-depsc', '-r300');
print(h_conv, [conv_prefix '.pdf'], '-dpdf');
print(h_conv, [conv_prefix '.png'], '-dpng', '-r180');

%% preview figures
if ~isnan(preview_local)
    preview_names = {'original', 'sparse', 'low_rank'};
    preview_values = {f(:, preview_local), ...
        x5(:, preview_local), L5(:, preview_local)};
    for preview_index = 1:numel(preview_values)
        h_preview = figure('Visible', 'on', 'Units', 'centimeters', ...
            'Position', [2 2 8 8], 'Color', 'w', 'Renderer', 'painters');
        % imgsc is the historical helper: imagesc, gray colormap,
        % axis image, and axis off, with no titles or numerical overlays.
        imgsc(reshape(preview_values{preview_index}, imSize));
        disableDefaultInteractivity(gca);
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

fprintf('PCP figures saved for %s.\n', type);

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
