function sol = solve_bvp_ode(p_func, q_func, alpha, beta, varargin)
% SOLVE_BVP_ODE  Solve a second-order BVP ODE via reduction to a Fredholm IE.
%
%   The ODE must be of the form
%
%       y''(x) + p(x) y(x) = q(x),   x ∈ [a, b],
%       y(a) = α,   y(b) = β.
%
%   The substitution u(x) = y''(x) yields the linear FIE
%
%       u(x) = f(x) + ∫_a^b K(x,t) u(t) dt,
%
%   where
%
%       f(x)    = q(x) − α p(x) − (β − α)·(x − a)/(b − a)·p(x),
%
%       K(x, t) = p(x) · G(x, t),   with Green's function
%
%                 ⎧ (t−a)(b−x)/(b−a),   t ≤ x
%       G(x, t) = ⎨
%                 ⎩ (x−a)(b−t)/(b−a),   t > x.
%
%   Once u(x) is found the solution is recovered via
%
%       y(x) = (q(x) − u(x)) / p(x).
%
%   sol = fredholm_nn.solvers.solve_bvp_ode(p_func, q_func, alpha, beta)
%   sol = fredholm_nn.solvers.solve_bvp_ode(..., Name, Value, ...)
%
%   Required arguments
%   ------------------
%   p_func : function handle  p(x).  Accepts and returns a column vector.
%   q_func : function handle  q(x).  Accepts and returns a column vector.
%   alpha  : scalar  — boundary value y(a).
%   beta   : scalar  — boundary value y(b).
%
%   Name-Value options
%   ------------------
%   'Domain'       : [a, b] — integration interval (default [0, 1]).
%   'NGrid'        : integer — quadrature points (default 1000).
%   'NIterations'  : integer — FNN hidden layers K (default 15).
%   'PredictAt'    : column vector — query points for ŷ(x).
%                   Defaults to the integration grid.
%   'KMConstant'   : scalar κ ∈ (0,1] (default 1.0 = Picard).
%
%   Returns
%   -------
%   sol : struct with fields
%     .x             — query points (column vector)
%     .y             — ODE solution ŷ(x) (column vector)
%     .u             — auxiliary u(x) = y''(x) from the FIE solve
%     .fie_solution  — sol struct returned by solve_linear_fie
%     .error(fn)     — function handle: pointwise |ŷ − fn(x)|
%     .mse(fn)       — function handle: mean squared error vs fn(x)
%
%   Examples
%   --------
%   % y''(x) + 3p/(p+x²)² y(x) = 0,  y(0)=0,  y(1)=1/√(p+1)
%   p = 3.2;
%   sol = fredholm_nn.solvers.solve_bvp_ode( ...
%       @(x) 3*p ./ (p + x.^2).^2, ...
%       @(x) zeros(size(x)), ...
%       0.0, 1.0/sqrt(p+1), ...
%       'NGrid', 1000, 'NIterations', 10);
%   mse = sol.mse(@(x) 1./sqrt(p + x.^2));

    p = inputParser();
    addRequired(p, 'p_func');
    addRequired(p, 'q_func');
    addRequired(p, 'alpha');
    addRequired(p, 'beta');
    addParameter(p, 'Domain',      [0, 1], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'NGrid',       1000,   @(x) isnumeric(x) && x >= 2);
    addParameter(p, 'NIterations', 15,     @(x) isnumeric(x) && x >= 1);
    addParameter(p, 'PredictAt',   [],     @isnumeric);
    addParameter(p, 'KMConstant',  1.0,    @(x) isnumeric(x) && x > 0 && x <= 1);
    parse(p, p_func, q_func, alpha, beta, varargin{:});
    opts = p.Results;

    a = opts.Domain(1);
    b = opts.Domain(2);
    L = b - a;
    scale = (beta - alpha) / L;

    % ------------------------------------------------------------------
    % Build FIE kernel K(t_row, x_col) and free term f(x)
    % ------------------------------------------------------------------
    % Derivation: u = y'' satisfies u = f + ∫K u dt where
    % K(x,t) = +p(x)·G(x,t)  (positive sign).
    % Obtained by substituting y = y_BC - ∫G u dt into u = q - p·y.
    %
    % Convention: kernel(z(:)', x(:)) — z is 1×N row, x is M×1 col.
    % Must return M×N via implicit expansion.
    kernel_fie = @(z_row, x_col) ...
        p_func(x_col) .* green_fn(z_row, x_col, a, b, L);

    additive_fie = @(x) ...
        q_func(x(:)) - alpha * p_func(x(:)) - scale * (x(:) - a) .* p_func(x(:));

    % ------------------------------------------------------------------
    % Solve the FIE
    % ------------------------------------------------------------------
    [grid, dz] = fredholm_nn.utils.make_uniform_grid(a, b, opts.NGrid);
    gd         = fredholm_nn.utils.make_grid_dictionary(grid, opts.NIterations);

    if isempty(opts.PredictAt)
        x_query = grid;
    else
        x_query = opts.PredictAt(:);
    end

    if opts.KMConstant < 1.0
        model = fredholm_nn.models.FredholmNN_KM( ...
            gd, kernel_fie, additive_fie, dz, opts.NIterations, opts.KMConstant);
    else
        model = fredholm_nn.models.FredholmNN( ...
            gd, kernel_fie, additive_fie, dz, opts.NIterations);
    end

    u = model.forward(x_query);   % u(x) = y''(x)

    % Build a fie_solution struct for diagnostics
    fie_sol.x     = x_query;
    fie_sol.f     = u;
    fie_sol.model = model;

    % ------------------------------------------------------------------
    % Recover y(x) = (q(x) − u(x)) / p(x)
    % ------------------------------------------------------------------
    p_vals = p_func(x_query);
    q_vals = q_func(x_query);

    % Guard against division by zero
    tol_zero = 1e-14;
    y_hat = zeros(size(u));
    safe  = abs(p_vals) > tol_zero;
    y_hat(safe)  = (q_vals(safe) - u(safe)) ./ p_vals(safe);
    y_hat(~safe) = NaN;

    % ------------------------------------------------------------------
    % Assemble output
    % ------------------------------------------------------------------
    sol.x            = x_query;
    sol.y            = y_hat;
    sol.u            = u;
    sol.fie_solution = fie_sol;
    sol.error        = @(exact_fn) abs(y_hat - exact_fn(x_query));
    sol.mse          = @(exact_fn) mean((y_hat - exact_fn(x_query)).^2);
end

% -------------------------------------------------------------------------
function G = green_fn(z_row, x_col, a, b, L)
% GREEN_FN  Green's function for second-order BVP on [a,b].
%
%   G(x,t) = (t−a)(b−x)/L   for t ≤ x
%           = (x−a)(b−t)/L   for t > x
%
%   z_row : 1×N
%   x_col : M×1
%   Returns M×N via implicit expansion.

    G = (z_row <= x_col) .* (z_row - a) .* (b - x_col) / L + ...
        (z_row >  x_col) .* (x_col - a) .* (b - z_row) / L;
end
