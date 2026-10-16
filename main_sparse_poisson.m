%% Nonnegative sparse Poisson recovery: one problem, two recorded quantities.
clear;
close all;
clc;
code_root = fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(fullfile(code_root,'toolbox'));
addpath(fullfile(code_root,'menthod'));
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

%% load data / problem setup
rng(2026,'twister');
n = 256;
blur_weight = 0.95;
poisson_background = 5;
poisson_lambda = 0.5;
recovery_initial_level = 100; % Common feasible start used to examine boundary behavior.
shift = circshift(speye(n),1,1);
A = blur_weight*speye(n)+(1-blur_weight)/2*(shift+shift');
truth = zeros(n,1);
support = randperm(n,24);
truth(support) = 8+8*rand(24,1);
background = poisson_background*ones(n,1);
counts = poisson_sample(A*truth+background);
lambda = poisson_lambda;
L = max(counts./background.^2); % norm(A)=1; valid smoothness bound on x>=0.

%% parameters
para.m = n;
para.n = n;
para.mu = lambda;
para.gamma = 1/L;
para.tol = 1e-17;
para.maxits = 10000;
para.x0 = recovery_initial_level*ones(n,1);
para.verbose = 1;
para.a = 40;
p = 1/20;
q = 1/2;
r = 4;
gradF = @(x) sparse_poisson_gradient(x,A,counts,background);
proxJ = @(v,t) max(v-t,0);
objPhi = @(x) sparse_poisson_value(x,A,counts,background)+lambda*sum(x(:));

%% numerical reference: independent proximal gradient from x0=ones(n,1)
reference_para = para;
reference_para.x0 = ones(n,1);
reference = poisson_reference(reference_para,proxJ,gradF);
xsol = reference.x;

%% gradient-domain record for the Poisson feasibility figure only
% The original solvers keep their six-output interface and internal timer.
% Stop an invalid gradient query; do not clip intensities or alter the method.
gradF = @(x) poisson_trace('gradient',x);
objPhi = @(x) poisson_trace('objective',x);

%% FISTA
fprintf('Performing FISTA...\n');
poisson_trace('start',para.x0,A,counts,background,lambda,xsol);
try
    [x1, its1, dk1, ek1, fk1, t1] = ...
        func_FISTA(para, proxJ, gradF, objPhi, xsol);
catch err
    if ~strcmp(err.identifier,'SSFISTA:PoissonDomain'), rethrow(err); end
    [x1, its1, dk1, ek1, fk1, t1] = poisson_trace('failed');
    fprintf('FISTA: stopped at a nonpositive-intensity query.\n');
end
trace1 = poisson_trace('get');

%% FISTA-Mod
fprintf('Performing FISTA-Mod...\n');
poisson_trace('start',para.x0,A,counts,background,lambda,xsol);
try
    [x2, its2, dk2, ek2, fk2, t2] = ...
        func_FISTA_Mod(p,q,r, para, proxJ, gradF, objPhi, xsol);
catch err
    if ~strcmp(err.identifier,'SSFISTA:PoissonDomain'), rethrow(err); end
    [x2, its2, dk2, ek2, fk2, t2] = poisson_trace('failed');
    fprintf('FISTA-Mod: stopped at a nonpositive-intensity query.\n');
end
trace2 = poisson_trace('get');

%% MG-FISTA
fprintf('Performing MG-FISTA...\n');
poisson_trace('start',para.x0,A,counts,background,lambda,xsol);
try
    [x6, its6, dk6, ek6, fk6, t6] = ...
        func_MGFISTA(para, proxJ, gradF, objPhi, xsol);
catch err
    if ~strcmp(err.identifier,'SSFISTA:PoissonDomain'), rethrow(err); end
    [x6, its6, dk6, ek6, fk6, t6] = poisson_trace('failed');
    fprintf('MG-FISTA: stopped at a nonpositive-intensity query.\n');
end
trace6 = poisson_trace('get');

%% NAGEL
fprintf('Performing NAGEL...\n');
poisson_trace('start',para.x0,A,counts,background,lambda,xsol);
try
    [x7, its7, dk7, ek7, fk7, t7] = ...
        func_NAGEL(para, proxJ, gradF, objPhi, xsol);
catch err
    if ~strcmp(err.identifier,'SSFISTA:PoissonDomain'), rethrow(err); end
    [x7, its7, dk7, ek7, fk7, t7] = poisson_trace('failed');
    fprintf('NAGEL: stopped at a nonpositive-intensity query.\n');
end
trace7 = poisson_trace('get');

%% GAPGA
fprintf('Performing GAPGA...\n');
poisson_trace('start',para.x0,A,counts,background,lambda,xsol);
try
    [x4, its4, dk4, ek4, fk4, t4] = ...
        func_GAPGA(para, proxJ, gradF, objPhi, xsol);
catch err
    if ~strcmp(err.identifier,'SSFISTA:PoissonDomain'), rethrow(err); end
    [x4, its4, dk4, ek4, fk4, t4] = poisson_trace('failed');
    fprintf('GAPGA: stopped at a nonpositive-intensity query.\n');
end
trace4 = poisson_trace('get');

%% SS-FISTA
fprintf('Performing SS-FISTA...\n');
poisson_trace('start',para.x0,A,counts,background,lambda,xsol);
try
    [x5, its5, dk5, ek5, fk5, t5] = ...
        func_SSFISTA(para, proxJ, gradF, objPhi, xsol);
catch err
    if ~strcmp(err.identifier,'SSFISTA:PoissonDomain'), rethrow(err); end
    [x5, its5, dk5, ek5, fk5, t5] = poisson_trace('failed');
    fprintf('SS-FISTA: stopped at a nonpositive-intensity query.\n');
end
trace5 = poisson_trace('get');

%% results
Method = {'FISTA'; 'FISTA-Mod'; 'MG-FISTA'; 'NAGEL'; 'GAPGA'; 'SS-FISTA'};
Iterations = [its1; its2; its6; its7; its4; its5];
CPUTime = [t1; t2; t6; t7; t4; t5];
% A failed solver has no returned runtime; report NaN rather than add a timer.
results_table = table(Method,Iterations,CPUTime);
disp(results_table);

%% save results
raw_dir = fullfile(code_root,'results','raw');
table_dir = fullfile(code_root,'results','tables');
figure_dir = fullfile(code_root,'results','figures');
if ~exist(raw_dir,'dir'), mkdir(raw_dir); end
if ~exist(table_dir,'dir'), mkdir(table_dir); end
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end
result_name = 'sparse_poisson_same_case';
writetable(results_table,fullfile(table_dir,[result_name '_results.csv']));
save(fullfile(raw_dir,[result_name '_results.mat']), ...
    'A','counts','background','truth','lambda','L','para','p','q','r','xsol','reference', ...
    'Method','Iterations','CPUTime','results_table', ...
    'x1','its1','dk1','ek1','fk1','t1','trace1', ...
    'x2','its2','dk2','ek2','fk2','t2','trace2', ...
    'x6','its6','dk6','ek6','fk6','t6','trace6', ...
    'x7','its7','dk7','ek7','fk7','t7','trace7', ...
    'x4','its4','dk4','ek4','fk4','t4','trace4', ...
    'x5','its5','dk5','ek5','fk5','t5','trace5');

%% plotting
methods = Method';
distances = {dk1,dk2,dk6,dk7,dk4,dk5};
traces = {trace1,trace2,trace6,trace7,trace4,trace5};
colors = [0.57 0.57 0.57; 1/6 1/6 1/6; 0.494 0.184 0.556; ...
    0.929 0.694 0.125; 1/6 1/4 1/1.2; 1 0 0];
intensity_styles = {'-','-','-','-','-','--'};
intensity_markers = {'none','none','none','none','o','s'};
fig_convergence = figure(201); clf;
set(fig_convergence,'Position',[100 100 560 420]);
set(fig_convergence,'PaperUnits','centimeters', ...
    'PaperPosition',[0 0 10 8],'PaperSize',[10 8]);
window = 100*ceil(max(Iterations)/100);
ax_convergence = axes('Parent',fig_convergence);
set(ax_convergence,'FontSize',8);
disableDefaultInteractivity(ax_convergence);
hold on;
for j = 1:6
    semilogy(1:Iterations(j),max(distances{j},realmin), ...
        'Color',colors(j,:),'LineStyle','-', ...
        'LineWidth',1.3,'DisplayName',methods{j});
end
set(gca,'YScale','log'); grid on; box on;
xlabel('Iteration'); ylabel('$\|x_k-x^\star\|_2$','Interpreter','latex');
xlim([1 window]);
positive_errors = vertcat(distances{:});
positive_errors = positive_errors(positive_errors>0);
ylim([10^floor(log10(min(positive_errors))),10^ceil(log10(max(positive_errors)))]);
yticks(10.^(-16:4:4));
legend('Location','northeast','FontSize',8,'Box','off');

fig_intensity = figure(202); clf;
set(fig_intensity,'Position',[680 100 560 420]);
set(fig_intensity,'PaperUnits','centimeters', ...
    'PaperPosition',[0 0 10 8],'PaperSize',[10 8]);
ax_intensity = axes('Parent',fig_intensity);
set(ax_intensity,'FontSize',8);
disableDefaultInteractivity(ax_intensity);
hold on;
intensity_axis = @(v) sign(v).*log10(1+abs(v));
for j = 1:6
    trace = traces{j};
    keep = trace.IterationIndex>=1;
    plot(trace.IterationIndex(keep),intensity_axis(trace.MinimumIntensity(keep)), ...
        'Color',colors(j,:),'LineStyle',intensity_styles{j}, ...
        'Marker',intensity_markers{j},'MarkerIndices',1:5:nnz(keep),'MarkerSize',3, ...
        'LineWidth',1.3,'DisplayName',methods{j});
    if ~isnan(trace.FailedIteration)
        plot(trace.FailedIteration,intensity_axis(trace.FailedIntensity),'x', ...
            'Color',colors(j,:),'MarkerSize',9,'LineWidth',1.5,'HandleVisibility','off');
    end
end
yline(0,'k--','HandleVisibility','off'); grid on; box on;
xlabel('Iteration'); ylabel('Minimum Poisson intensity'); xlim([1 window]);
minimum_intensity = min(cellfun(@(v) min(v.MinimumIntensity),traces));
maximum_intensity = max(cellfun(@(v) max(v.MinimumIntensity),traces));
lower_limit = min(-5,5*floor(minimum_intensity/5));
intensity_ticks = unique([lower_limit,-40,-5,-1,0,1,5,20,100,200]);
intensity_ticks = intensity_ticks(intensity_ticks>=lower_limit);
yticks(intensity_axis(intensity_ticks)); yticklabels(string(intensity_ticks));
ylim(intensity_axis([lower_limit,1.05*maximum_intensity]));
legend('Location','northeast','FontSize',8,'Box','off');

%% inset: first 60 actual gradient-query iterations

drawnow;
main_position = ax_intensity.Position;
main_limits = ax_intensity.YLim;
rectangle(ax_intensity,'Position',[1,main_limits(1),59,diff(main_limits)], ...
    'EdgeColor',[0.45 0.45 0.45],'LineStyle',':','LineWidth',0.7, ...
    'HandleVisibility','off');
ax_inset = axes(fig_intensity,'Position',[main_position(1)+0.28*main_position(3), ...
    main_position(2)+0.16*main_position(4),0.68*main_position(3), ...
    0.30*main_position(4)],'Color','w','FontSize',8,'Box','on');
disableDefaultInteractivity(ax_inset);
hold(ax_inset,'on');
for j = 1:6
    trace = traces{j}; keep = trace.IterationIndex>=1 & trace.IterationIndex<=60;
    plot(ax_inset,trace.IterationIndex(keep),intensity_axis(trace.MinimumIntensity(keep)), ...
        'Color',colors(j,:),'LineStyle',intensity_styles{j}, ...
        'Marker',intensity_markers{j},'MarkerIndices',1:5:nnz(keep),'MarkerSize',3, ...
        'LineWidth',1.3);
    if ~isnan(trace.FailedIteration) && trace.FailedIteration<=60
        plot(ax_inset,trace.FailedIteration,intensity_axis(trace.FailedIntensity),'x', ...
            'Color',colors(j,:),'MarkerSize',8,'LineWidth',1.5);
        text(ax_inset,trace.FailedIteration+2,intensity_axis(trace.FailedIntensity), ...
            sprintf('%d',trace.FailedIteration),'FontSize',8, ...
            'Color',colors(j,:),'VerticalAlignment','middle');
    end
end
yline(ax_inset,0,'k--'); xlim(ax_inset,[0 60]); ylim(ax_inset,main_limits);
xticks(ax_inset,[0 20 40 60]);
inset_ticks = [-5 0 5 20 100];
yticks(ax_inset,intensity_axis(inset_ticks)); yticklabels(ax_inset,string(inset_ticks));
grid(ax_inset,'on');
title(ax_inset,'First 60 iterations','FontSize',8,'FontWeight','normal');

%% save the two figures separately
print(fig_convergence,fullfile(figure_dir,'sparse_poisson_convergence.png'),'-dpng','-r180');
print(fig_convergence,fullfile(figure_dir,'sparse_poisson_convergence.pdf'),'-dpdf');
print(fig_convergence,fullfile(figure_dir,'sparse_poisson_convergence.eps'),'-depsc','-loose','-r300');
print(fig_intensity,fullfile(figure_dir,'sparse_poisson_min_intensity.png'),'-dpng','-r180');
print(fig_intensity,fullfile(figure_dir,'sparse_poisson_min_intensity.pdf'),'-dpdf');
print(fig_intensity,fullfile(figure_dir,'sparse_poisson_min_intensity.eps'),'-depsc','-loose','-r300');

function varargout = poisson_trace(action,x,A,counts,background,lambda,xsol)
% Minimal domain trace for the two Poisson panels; no timing or accuracy targets.
persistent matrix observations offset penalty reference state iteration last_x
switch action
    case 'start'
        matrix=A; observations=counts; offset=background; penalty=lambda; reference=xsol;
        iteration=0; last_x=[];
        state=struct('x',x,'dk',[],'ek',[],'fk',[], ...
            'QueryIteration',[],'QueryIntensity',[], ...
            'FailedIteration',NaN,'FailedIntensity',NaN);
    case 'objective'
        iteration=iteration+1;
        if ~isempty(last_x), state.ek(end+1,1)=norm(x-last_x); end
        state.x=x; last_x=x;
        value=sparse_poisson_value(x,matrix,observations,offset)+penalty*sum(x(:));
        state.dk(end+1,1)=norm(x-reference);
        state.fk(end+1,1)=value;
        varargout={value};
    case 'gradient'
        intensity=matrix*x+offset;
        margin=min(intensity);
        state.QueryIteration(end+1,1)=iteration;
        state.QueryIntensity(end+1,1)=margin;
        if margin<=0
            state.FailedIteration=iteration; state.FailedIntensity=margin;
            error('SSFISTA:PoissonDomain','The Poisson gradient needs positive intensities.');
        end
        varargout={matrix'*(1-observations./intensity)};
    case 'failed'
        completed=max(0,iteration-1);
        varargout={state.x,completed,state.dk(1:completed), ...
            state.ek(1:completed),state.fk(1:completed),NaN};
    case 'get'
        [index,~,group]=unique(state.QueryIteration);
        minimum=accumarray(group,state.QueryIntensity,[],@min);
        varargout={struct('IterationIndex',index,'MinimumIntensity',minimum, ...
            'FailedIteration',state.FailedIteration,'FailedIntensity',state.FailedIntensity)};
end
end

function reference = poisson_reference(para,ProxJ,GradF)
% Independent non-accelerated proximal gradient, used only as a reference.
% Refine below plotting accuracy; never use any comparison method as reference.
x = para.x0; residual = Inf; clock_id = tic;
for k = 1:100000
    next = ProxJ(x-para.gamma*GradF(x),para.mu*para.gamma);
    residual = norm((x-next)/para.gamma)/max(1,norm(x));
    x = next;
    if residual <= 5e-17, break; end
end
runtime = toc(clock_id);
point = ProxJ(x-para.gamma*GradF(x),para.mu*para.gamma);
residual = norm((x-point)/para.gamma)/max(1,norm(x));
assert(isfinite(residual) && residual <= 1e-15,'Poisson reference is not accurate enough.');
reference = struct('x',x,'Iterations',k,'Time',runtime, ...
    'Method','Independent proximal gradient','NormalizedGradientMapping',residual, ...
    'Status','validated_reference','ValidationTolerance',1e-15,'Validated',true);
end

function counts = poisson_sample(intensity)
% Knuth sampler; reproducible and independent of Statistics Toolbox.
counts = zeros(size(intensity));
for j = 1:numel(intensity)
    threshold = exp(-intensity(j)); product = 1; k = 0;
    while product > threshold, k = k+1; product = product*rand; end
    counts(j) = k-1;
end
end

function gradient = sparse_poisson_gradient(x, A, counts, background)
intensity = A*x+background;
if any(intensity <= 0)
    error('SSFISTA:PoissonDomain','The Poisson gradient needs strictly positive intensities.');
end
gradient = A'*(1-counts./intensity);
end

function value = sparse_poisson_value(x, A, counts, background)
% Negative log likelihood, with the data-only constants omitted.
intensity = A*x+background;
if any(intensity <= 0), value = Inf; return; end
value = sum(intensity-counts.*log(intensity));
end
