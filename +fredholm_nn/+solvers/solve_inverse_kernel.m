function sol = solve_inverse_kernel(additive, f_target, x_train, dx, varargin)
% SOLVE_INVERSE_KERNEL  Learn an unknown kernel from forward FIE data.
%
%   Given observed solution data f_target on x_train and the known additive
%   term g(x), learn a kernel K_θ(x,z) — parameterised by a shallow neural
%   network with N_neurons hidden units — such that
%
%       f(x) ≈ g(x) + ∫ K_θ(x,z) f(z) dz.
%
%   The optimisation problem is solved with MATLAB's lsqnonlin using the
%   Levenberg-Marquardt algorithm (requires the Optimization Toolbox).
%   The objective combines a data residual with optional physics and
%   kernel-norm regularisation terms.
%
%   sol = fredholm_nn.solvers.solve_inverse_kernel(additive, f_target, x_train, dx)
%   sol = fredholm_nn.solvers.solve_inverse_kernel(..., Name, Value, ...)
%
%   Required arguments
%   ------------------
%   additive : function handle  g(x).
%   f_target : column vector — target solution values on x_train.
%   x_train  : column vector — training grid.
%   dx       : scalar — quadrature step (= (x_train(end)-x_train(1))/(N-1)).
%
%   Name-Value options
%   ------------------
%   'NIterations'    : integer — FNN hidden layers K (default 15).
%   'NNeurons'       : integer — hidden units in kernel network (default 20).
%   'MaxIterations'  : integer — lsqnonlin max iterations (default 300).
%   'FuncTol'        : scalar  — lsqnonlin FunctionTolerance (default 1e-6).
%   'LambdaPhys'     : scalar  — physics residual weight λ_phys (default 0).
%   'LambdaKernel'   : scalar  — kernel-norm weight λ_K (default 1e-5).
%   'LambdaReg'      : scalar  — weight L₂ regularisation λ_wb (default 1e-6).
%   'NInstances'     : integer — independent random restarts (default 1).
%   'Verbose'        : logical — print lsqnonlin progress (default true).
%
%   Returns
%   -------
%   sol : struct with fields
%     .net_kernel  — best-found kernel network (Neural Network Toolbox object)
%     .f_hat       — final forward prediction on x_train
%     .mse         — mean squared error of f_hat vs f_target
%     .all_mse     — vector of final MSEs for every restart
%     .kernel_fn   — @(z_row, x_col) handle for the learned K_θ
%
%   Notes
%   -----
%   Requires the Optimization Toolbox (lsqnonlin) and the Deep Learning
%   (or Neural Network) Toolbox (feedforwardnet, getwb, setwb).
%
%   Reference
%   ---------
%   Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
%   Fredholm neural networks. SIAM J. Sci. Comput., 47(4).

    p = inputParser();
    addRequired(p, 'additive');
    addRequired(p, 'f_target');
    addRequired(p, 'x_train');
    addRequired(p, 'dx');
    addParameter(p, 'NIterations',   15,    @(x) isnumeric(x) && x >= 1);
    addParameter(p, 'NNeurons',      20,    @(x) isnumeric(x) && x >= 1);
    addParameter(p, 'MaxIterations', 300,   @(x) isnumeric(x) && x >= 1);
    addParameter(p, 'FuncTol',       1e-6,  @(x) isnumeric(x) && x > 0);
    addParameter(p, 'LambdaPhys',    0,     @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'LambdaKernel',  1e-5,  @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'LambdaReg',     1e-6,  @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'NInstances',    1,     @(x) isnumeric(x) && x >= 1);
    addParameter(p, 'Verbose',       true,  @islogical);
    parse(p, additive, f_target, x_train, dx, varargin{:});
    opts = p.Results;

    % ------------------------------------------------------------------
    % Build Fredholm NN skeleton (kernel replaced during optimisation)
    % ------------------------------------------------------------------
    x_train  = x_train(:);
    f_target = f_target(:);

    gd = fredholm_nn.utils.make_grid_dictionary(x_train, opts.NIterations);
    dummy_kernel = @(z_row, x_col) zeros(length(x_col), length(z_row));
    fred_model   = fredholm_nn.models.FredholmNN( ...
        gd, dummy_kernel, additive, dx, opts.NIterations);

    display_opt = 'off';
    if opts.Verbose
        display_opt = 'iter-detailed';
    end

    lm_options = optimoptions('lsqnonlin', ...
        'Display',             display_opt, ...
        'Algorithm',           'levenberg-marquardt', ...
        'FunctionTolerance',   opts.FuncTol, ...
        'MaxIterations',       opts.MaxIterations);

    % ------------------------------------------------------------------
    % Multiple random restarts
    % ------------------------------------------------------------------
    all_mse     = zeros(opts.NInstances, 1);
    all_nets    = cell(opts.NInstances, 1);

    for inst = 1:opts.NInstances
        if opts.Verbose
            fprintf('\n--- Instance %d / %d ---\n', inst, opts.NInstances);
        end

        % Fresh kernel network for each restart
        net_k = feedforwardnet(opts.NNeurons, 'trainlm');
        net_k = configure(net_k, rand(2, 10), rand(1, 10));
        net_k.layers{1}.transferFcn = 'tansig';
        net_k.layers{2}.transferFcn = 'purelin';

        % Residual function
        residual_fn = @(wb) local_residual( ...
            net_k, wb, fred_model, x_train, f_target, dx, additive, ...
            opts.LambdaPhys, opts.LambdaKernel, opts.LambdaReg);

        wb0     = getwb(net_k);
        wb_opt  = lsqnonlin(residual_fn, wb0, [], [], lm_options);
        net_k   = setwb(net_k, wb_opt);

        f_hat_i   = local_forward(net_k, fred_model, x_train);
        mse_i     = mean((f_hat_i - f_target).^2);
        all_mse(inst)  = mse_i;
        all_nets{inst} = net_k;

        if opts.Verbose
            fprintf('Instance %d: MSE = %e\n', inst, mse_i);
        end
    end

    % ------------------------------------------------------------------
    % Best restart
    % ------------------------------------------------------------------
    [best_mse, best_idx] = min(all_mse);
    best_net  = all_nets{best_idx};
    f_hat_best = local_forward(best_net, fred_model, x_train);

    sol.net_kernel = best_net;
    sol.f_hat      = f_hat_best;
    sol.mse        = best_mse;
    sol.all_mse    = all_mse;
    sol.kernel_fn  = @(z_row, x_col) local_eval_kernel(best_net, z_row, x_col);
end

% =========================================================================
%  Local helpers
% =========================================================================

function err = local_residual(net_k, wb, fred_model, x_train, f_target, ...
                               dx, additive, lam_phys, lam_K, lam_wb)
    net_k         = setwb(net_k, wb);
    fred_model.kernel = @(z_row, x_col) local_eval_kernel(net_k, z_row, x_col);

    f_fnn    = fred_model.forward(x_train);
    err_data = f_fnn - f_target;

    % Physics residual
    Kmat         = local_eval_kernel(net_k, x_train(:)', x_train(:));
    integral_prt = (Kmat * f_target) * dx;
    err_phys     = sqrt(lam_phys) * (f_target - integral_prt - additive(x_train));

    % Kernel-norm regularisation
    row_l2 = sqrt(sum(Kmat.^2, 2)) * sqrt(dx);
    err_kn = sqrt(lam_K) * row_l2;

    % Weight L₂ regularisation
    err_wb = sqrt(lam_wb) * wb(:);

    err = [err_data(:); err_phys(:); err_kn(:); err_wb(:)];
end


function f_hat = local_forward(net_k, fred_model, x_train)
    fred_model.kernel = @(z_row, x_col) local_eval_kernel(net_k, z_row, x_col);
    f_hat = fred_model.forward(x_train);
end


function out = local_eval_kernel(net_k, z_row, x_col)
% LOCAL_EVAL_KERNEL  Evaluate the kernel NN at all (x, z) pairs.
%
%   z_row : 1×N row vector
%   x_col : M×1 column vector
%   Returns M×N matrix.

    z_row = z_row(:).';   % ensure 1×N
    x_col = x_col(:);     % ensure M×1

    [X, Z] = ndgrid(x_col, z_row);   % X, Z are M×N
    nn_input = [X(:)'; Z(:)'];        % 2×(M*N)

    out_vec = net_k(nn_input);        % 1×(M*N)
    out     = reshape(out_vec, size(X));   % M×N
end
