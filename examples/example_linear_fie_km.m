%% example_linear_fie_km.m
% Linear Fredholm IE — Krasnoselskii-Mann (KM) relaxed iteration.
%
%   f(x) = sin(25x) + sin(7x)
%          + λ ∫_0^1 [cos(25(x-z)) + cos(7(x-z))] f(z) dz,   λ = 0.3
%
%   No closed-form exact solution is known for this case.
%   The KM constant κ < 1 is used to handle the near-non-expansive operator.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

lambda   = 0.3;
kernel   = @(z, x) lambda * (cos(25*(x - z)) + cos(7*(x - z)));
additive = @(x) sin(25*x) + sin(7*x);
domain   = [0, 1];

sol_km = fredholm_nn.solvers.solve_linear_fie(kernel, additive, domain, ...
    'NGrid',       300,  ...
    'NIterations', 10,   ...
    'KMConstant',  0.5);

% Also run standard Picard for comparison
sol_pi = fredholm_nn.solvers.solve_linear_fie(kernel, additive, domain, ...
    'NGrid',       300,  ...
    'NIterations', 10);

fprintf('KM iteration completed (%d grid points, κ = 0.5).\n', 300);

% --- Overlay plot (KM vs Picard) ---
figure;
plot(sol_km.x, sol_km.f, 'b-',  'LineWidth', 2, 'DisplayName', 'KM (κ=0.5)');
hold on;
plot(sol_pi.x, sol_pi.f, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Picard');
xlabel('x'); ylabel('f(x)');
title('Linear FIE KM: KM vs Picard iteration');
legend('Location', 'best'); grid on;

% --- Difference plot ---
figure;
plot(sol_km.x, abs(sol_km.f - sol_pi.f), 'k-', 'LineWidth', 1.5);
xlabel('x'); ylabel('|f_{KM} - f_{Picard}|');
title('Linear FIE KM: absolute difference between KM and Picard');
grid on;
