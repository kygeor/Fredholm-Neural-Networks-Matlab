%% example_linear_fie.m
% Linear Fredholm IE of the second kind — contractive (Picard) case.
%
%   f(x) = sin(x) + ∫_0^{π/2} sin(x) cos(z) f(z) dz
%
%   Exact solution:  f(x) = 2 sin(x)

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

kernel   = @(z, x) sin(x) .* cos(z);    % M×N via implicit expansion
additive = @(x) sin(x);
domain   = [0, pi/2];

sol = fredholm_nn.solvers.solve_linear_fie(kernel, additive, domain, ...
    'NGrid', 300, 'NIterations', 10);

exact = @(x) 2 * sin(x);
mse   = sol.mse(exact);
fprintf('Linear FIE — MSE vs exact: %.4e\n', mse);

% --- Overlay plot ---
figure;
plot(sol.x, sol.f,          'b-',  'LineWidth', 2, 'DisplayName', 'FNN');
hold on;
plot(sol.x, exact(sol.x),   'r--', 'LineWidth', 1.5, 'DisplayName', 'Exact');
xlabel('x'); ylabel('f(x)');
title('Linear FIE: predicted vs exact');
legend('Location', 'best'); grid on;

% --- Error plot ---
figure;
plot(sol.x, sol.error(exact), 'k-', 'LineWidth', 1.5);
xlabel('x'); ylabel('|f̂(x) - f(x)|');
title(sprintf('Linear FIE: absolute error  (MSE = %.2e)', mse));
grid on;
