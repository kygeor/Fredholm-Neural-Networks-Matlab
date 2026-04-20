%% example_inverse_kernel.m
% Inverse kernel problem: learn K_θ(x,z) from observed solution data.
%
%   Forward equation:  f(x) = sin(x) + ∫_0^{π/2} sin(x) cos(z) f(z) dz
%   True exact solution used as "observed" data: f(x) = 2 sin(x).
%
%   A shallow neural network (20 hidden units, tansig activation) is trained
%   to recover the kernel using the Fredholm NN as a differentiable forward
%   model inside lsqnonlin (Levenberg-Marquardt).
%
%   Requires: Optimization Toolbox + Deep Learning / Neural Network Toolbox.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

% ------------------------------------------------------------------
% 1. Generate "observed" data from the known forward model
% ------------------------------------------------------------------
N        = 100;
domain   = [0, pi/2];
x_train  = linspace(domain(1), domain(2), N)';
dx       = (domain(2) - domain(1)) / (N - 1);
f_target = 2 * sin(x_train);   % exact solution used as training data

additive = @(x) sin(x);        % known free term g(x)

% ------------------------------------------------------------------
% 2. Solve the inverse problem
% ------------------------------------------------------------------
fprintf('Training kernel network (this may take a minute)...\n');
sol = fredholm_nn.solvers.solve_inverse_kernel( ...
    additive, f_target, x_train, dx, ...
    'NIterations',   15,   ...
    'NNeurons',      20,   ...
    'MaxIterations', 200,  ...
    'FuncTol',       1e-8, ...
    'LambdaPhys',    0,    ...
    'LambdaKernel',  1e-5, ...
    'LambdaReg',     1e-6, ...
    'NInstances',    1,    ...
    'Verbose',       true);

fprintf('Inverse kernel — MSE (f_hat vs f_target): %.4e\n', sol.mse);

% ------------------------------------------------------------------
% 3. Visualise results
% ------------------------------------------------------------------

% Solution overlay
figure;
plot(x_train, f_target,  'r--', 'LineWidth', 1.5, 'DisplayName', 'Target f(x)');
hold on;
plot(x_train, sol.f_hat, 'b-',  'LineWidth', 2,   'DisplayName', 'FNN (learned K)');
xlabel('x'); ylabel('f(x)');
title('Inverse kernel: solution overlay');
legend('Location', 'best'); grid on;

% Absolute error
figure;
plot(x_train, abs(sol.f_hat - f_target), 'k-', 'LineWidth', 1.5);
xlabel('x'); ylabel('|f̂(x) - f(x)|');
title(sprintf('Inverse kernel: absolute error  (MSE = %.2e)', sol.mse));
grid on;

% Learned kernel vs true kernel heatmaps
[X, Z] = meshgrid(x_train, x_train);   % N×N
K_learned = sol.kernel_fn(x_train(:)', x_train(:));   % N×N
K_true    = sin(x_train) .* cos(x_train)';             % outer product (true kernel ∝ sin(x)cos(z))

figure;
subplot(1,2,1);
imagesc(x_train, x_train, K_learned);
colorbar; axis xy;
xlabel('z'); ylabel('x');
title('Learned K_θ(x,z)');

subplot(1,2,2);
imagesc(x_train, x_train, K_true);
colorbar; axis xy;
xlabel('z'); ylabel('x');
title('True K(x,z) = sin(x)cos(z)');
