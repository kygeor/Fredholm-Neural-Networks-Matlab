%% example_bvp_ode.m
% BVP ODE solved via reduction to a Fredholm IE.
%
%   y''(x) + p(x) y(x) = 0,   x ∈ [0, 1],
%   y(0) = 0,   y(1) = 1/√(p₀ + 1),
%
%   p(x) = 3 p₀ / (p₀ + x²)²,   p₀ = 3.2
%
%   Exact solution:  y(x) = x / √(p₀ + x²)

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

p0     = 3.2;
p_func = @(x) 3*p0 ./ (p0 + x.^2).^2;
q_func = @(x) zeros(size(x));
alpha  = 0.0;
beta   = 1.0 / sqrt(p0 + 1.0);
exact  = @(x) x ./ sqrt(p0 + x.^2);

sol = fredholm_nn.solvers.solve_bvp_ode(p_func, q_func, alpha, beta, ...
    'Domain',      [0, 1], ...
    'NGrid',       1000,   ...
    'NIterations', 10,     ...
    'PredictAt',   linspace(0, 1, 200)');

mse = sol.mse(exact);
fprintf('BVP ODE — MSE vs exact: %.4e\n', mse);

% --- Overlay plot ---
figure;
plot(sol.x, sol.y,          'b-',  'LineWidth', 2, 'DisplayName', 'FNN');
hold on;
plot(sol.x, exact(sol.x),   'r--', 'LineWidth', 1.5, 'DisplayName', 'Exact');
xlabel('x'); ylabel('y(x)');
title('BVP ODE: predicted vs exact');
legend('Location', 'best'); grid on;

% --- Error plot ---
figure;
plot(sol.x, sol.error(exact), 'k-', 'LineWidth', 1.5);
xlabel('x'); ylabel('|ŷ(x) - y(x)|');
title(sprintf('BVP ODE: absolute error  (MSE = %.2e)', mse));
grid on;
