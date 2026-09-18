function diagnostics = run_constrained_sparse_diagnostics(para, proxJ, baseGradF, objPhi, x_ref, p, q, r, formal_iterations, formal_iterates)
%RUN_CONSTRAINED_SPARSE_DIAGNOSTICS Independent query-level feasibility audit.
%   This helper reruns the four current formal methods using an exact logging
%   wrapper around baseGradF.  It is deliberately a diagnostic-only pass: its
%   timings are neither returned nor used by the formal Section 4.2 results.
%   No files are written here.

method_names = {'FISTA'; 'FISTA-Mod'; 'GAPGA'; 'SS-FISTA'};
infeasibility_threshold = -1e-12;

if ~isequal(size(formal_iterations), [4, 1])
    error('SSFISTA:InvalidFormalIterations', ...
        'formal_iterations must be a 4-by-1 vector in FISTA, FISTA-Mod, GAPGA, SS-FISTA order.');
end
if ~iscell(formal_iterates) || numel(formal_iterates) ~= 4
    error('SSFISTA:InvalidFormalIterates', ...
        'formal_iterates must be a four-element cell array in formal method order.');
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

diagnostics = repmat(empty_diagnostic(), 4, 1);

begin_trace(para.maxits);
[x, its] = run_silently(@() func_FISTA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(1) = complete_trace(method_names{1}, its, x, formal_iterations(1), formal_iterates{1});

begin_trace(para.maxits);
[x, its] = run_silently(@() func_FISTA_Mod(p, q, r, para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(2) = complete_trace(method_names{2}, its, x, formal_iterations(2), formal_iterates{2});

begin_trace(1 + 2 * para.maxits);
[x, its] = run_silently(@() func_GAPGA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(3) = complete_trace(method_names{3}, its, x, formal_iterations(3), formal_iterates{3});

begin_trace(para.maxits);
[x, its] = run_silently(@() func_SSFISTA(para, proxJ, @logged_grad, objPhi, x_ref));
diagnostics(4) = complete_trace(method_names{4}, its, x, formal_iterations(4), formal_iterates{4});

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
