%% run_examples.m
% Run all fredholm_nn_matlab examples and report pass/fail.
%
% Usage (from the fredholm_nn_matlab root):
%   addpath(pwd);
%   cd examples
%   run_examples
%
% Or run from any directory after adding the package root to path:
%   addpath('/path/to/fredholm_nn_matlab');
%   run('/path/to/fredholm_nn_matlab/examples/run_examples.m');

% Add package root to path
pkg_root = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(pkg_root);

fprintf('\n========================================\n');
fprintf('  fredholm_nn_matlab  —  Example Suite  \n');
fprintf('========================================\n\n');

results = struct('name', {}, 'mse', {}, 'passed', {});

% -------------------------------------------------------------------------
%% 1. Linear FIE (Picard)
% -------------------------------------------------------------------------
name   = 'Linear FIE (Picard)';
sol1   = [];
mse1   = NaN;
try
    kernel1   = @(z, x) sin(x) .* cos(z);
    additive1 = @(x) sin(x);
    exact1    = @(x) 2 * sin(x);

    sol1  = fredholm_nn.solvers.solve_linear_fie(kernel1, additive1, [0, pi/2], ...
        'NGrid', 300, 'NIterations', 10);
    mse1  = sol1.mse(exact1);
    ok    = report(name, mse1, 1e-5);
    results(end+1) = struct('name', name, 'mse', mse1, 'passed', ok);
catch ME
    fprintf('[FAIL] %s: %s\n', name, ME.message);
    results(end+1) = struct('name', name, 'mse', NaN, 'passed', false);
end
if ~isempty(sol1)
    fig_solution(sol1.x, sol1.f, exact1(sol1.x), name, 'FNN', 'Exact');
    fig_error(sol1.x, sol1.error(exact1), name, mse1);
end

% -------------------------------------------------------------------------
%% 2. Linear FIE (KM)
% -------------------------------------------------------------------------
name    = 'Linear FIE (KM)';
sol_km  = [];
sol_pi  = [];
diff_mse = NaN;
try
    lambda2   = 0.3;
    kernel2   = @(z, x) lambda2 * (cos(25*(x - z)) + cos(7*(x - z)));
    additive2 = @(x) sin(25*x) + sin(7*x);

    sol_km = fredholm_nn.solvers.solve_linear_fie(kernel2, additive2, [0, 1], ...
        'NGrid', 300, 'NIterations', 10, 'KMConstant', 0.5);
    sol_pi = fredholm_nn.solvers.solve_linear_fie(kernel2, additive2, [0, 1], ...
        'NGrid', 300, 'NIterations', 10);

    diff_mse = mean((sol_km.f - sol_pi.f).^2);
    ok = report(name, diff_mse, 1e-2);
    results(end+1) = struct('name', name, 'mse', diff_mse, 'passed', ok);
catch ME
    fprintf('[FAIL] %s: %s\n', name, ME.message);
    results(end+1) = struct('name', name, 'mse', NaN, 'passed', false);
end
if ~isempty(sol_km)
    fig_solution(sol_km.x, sol_km.f, sol_pi.f, name, 'KM (k=0.5)', 'Picard');
    fig_error(sol_km.x, abs(sol_km.f - sol_pi.f), name, diff_mse);
end

% -------------------------------------------------------------------------
%% 3. BVP ODE
% -------------------------------------------------------------------------
name  = 'BVP ODE';
sol3  = [];
mse3  = NaN;
try
    p0     = 3.2;
    p_func = @(x) 3*p0 ./ (p0 + x.^2).^2;
    q_func = @(x) zeros(size(x));
    alpha  = 0.0;
    beta   = 1.0 / sqrt(p0 + 1.0);
    exact3 = @(x) x ./ sqrt(p0 + x.^2);

    sol3  = fredholm_nn.solvers.solve_bvp_ode(p_func, q_func, alpha, beta, ...
        'NGrid', 1000, 'NIterations', 10, ...
        'PredictAt', linspace(0, 1, 200)');
    mse3  = sol3.mse(exact3);
    ok    = report(name, mse3, 1e-5);
    results(end+1) = struct('name', name, 'mse', mse3, 'passed', ok);
catch ME
    fprintf('[FAIL] %s: %s\n', name, ME.message);
    results(end+1) = struct('name', name, 'mse', NaN, 'passed', false);
end
if ~isempty(sol3)
    fig_solution(sol3.x, sol3.y, exact3(sol3.x), name, 'FNN', 'Exact');
    fig_error(sol3.x, sol3.error(exact3), name, mse3);
end

% -------------------------------------------------------------------------
%% Summary
% -------------------------------------------------------------------------
fprintf('\n----------------------------------------\n');
fprintf('  Results summary\n');
fprintf('----------------------------------------\n');
n_pass = 0;
for i = 1:length(results)
    status = 'PASS';
    if ~results(i).passed, status = 'FAIL'; end
    if isnan(results(i).mse)
        fprintf('  [%s] %s — ERROR\n', status, results(i).name);
    else
        fprintf('  [%s] %s — MSE = %.3e\n', status, results(i).name, results(i).mse);
    end
    n_pass = n_pass + results(i).passed;
end
fprintf('----------------------------------------\n');
fprintf('  %d / %d passed\n\n', n_pass, length(results));

% =========================================================================
%% Local helper functions
% =========================================================================

function ok = report(name, mse, tol)
    if mse <= tol
        fprintf('[PASS] %s — MSE = %.3e  (tol = %.1e)\n', name, mse, tol);
        ok = true;
    else
        fprintf('[WARN] %s — MSE = %.3e > tol = %.1e\n', name, mse, tol);
        ok = false;
    end
end


function fig_solution(x, f_pred, f_ref, name, pred_lbl, ref_lbl)
% FIG_SOLUTION  Overlay of predicted vs reference solution.
    if nargin < 5, pred_lbl = 'FNN';   end
    if nargin < 6, ref_lbl  = 'Exact'; end

    figure('Name', [name ' - solution']);
    plot(x, f_pred, 'b-',  'LineWidth', 2,   'DisplayName', pred_lbl);
    hold on;
    plot(x, f_ref,  'r--', 'LineWidth', 1.5, 'DisplayName', ref_lbl);
    xlabel('x'); ylabel('f(x)');
    title([name ': predicted vs reference']);
    legend('Location', 'best'); grid on;
end


function fig_error(x, err, name, mse)
% FIG_ERROR  Absolute error plot.
    figure('Name', [name ' — error']);
    plot(x, err, 'k-', 'LineWidth', 1.5);
    xlabel('x'); ylabel('Absolute error');
    title(sprintf('%s: absolute error  (MSE = %.2e)', name, mse));
    grid on;
end
