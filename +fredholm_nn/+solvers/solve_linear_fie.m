function sol = solve_linear_fie(kernel, additive, domain, varargin)
% SOLVE_LINEAR_FIE  Solve a linear Fredholm IE of the second kind.
%
%   f(x) = g(x) + integral_a^b K(x,z) f(z) dz
%
%   sol = fredholm_nn.solvers.solve_linear_fie(kernel, additive, domain)
%   sol = fredholm_nn.solvers.solve_linear_fie(..., Name, Value, ...)
%
%   Required arguments
%   ------------------
%   kernel   : function handle  K(z_row, x_col).
%              Called as kernel(z(:)', x(:)) — z is 1×N row, x is M×1 col;
%              must return an M×N matrix via implicit expansion.
%   additive : function handle  g(x).
%              Accepts and returns a column vector.
%   domain   : [a, b] — integration interval.
%
%   Name-Value options
%   ------------------
%   'NGrid'        : integer — quadrature points (default 500).
%   'NIterations'  : integer — Picard / KM iterations K (default 50).
%   'PredictAt'    : column vector — query points for f̂(x).
%                   Defaults to the integration grid.
%   'KMConstant'   : scalar κ ∈ (0,1] — use KM iteration for non-expansive
%                   operators (default 1.0 = standard Picard).
%
%   Returns
%   -------
%   sol : struct with fields
%     .x          — query points (column vector)
%     .f          — predicted solution values (column vector)
%     .model      — FredholmNN or FredholmNN_KM object
%     .error(fn)  — function handle: pointwise |f̂ - fn(x)|
%     .mse(fn)    — function handle: mean squared error vs fn(x)
%
%   Examples
%   --------
%   % f(x) = sin(x) + int_0^{pi/2} sin(x)cos(z) f(z) dz
%   % Exact: f(x) = 2*sin(x)
%   sol = fredholm_nn.solvers.solve_linear_fie( ...
%       @(z, x) sin(x) .* cos(z), ...
%       @(x) sin(x), [0, pi/2], ...
%       'NGrid', 300, 'NIterations', 10);
%   mse = sol.mse(@(x) 2*sin(x));

    p = inputParser();
    addRequired(p, 'kernel');
    addRequired(p, 'additive');
    addRequired(p, 'domain');
    addParameter(p, 'NGrid',       500,  @(x) isnumeric(x) && x >= 2);
    addParameter(p, 'NIterations', 50,   @(x) isnumeric(x) && x >= 1);
    addParameter(p, 'PredictAt',   [],   @isnumeric);
    addParameter(p, 'KMConstant',  1.0,  @(x) isnumeric(x) && x > 0 && x <= 1);
    parse(p, kernel, additive, domain, varargin{:});
    opts = p.Results;

    a = domain(1);  b = domain(2);

    [grid, dz] = fredholm_nn.utils.make_uniform_grid(a, b, opts.NGrid);
    gd         = fredholm_nn.utils.make_grid_dictionary(grid, opts.NIterations);

    if isempty(opts.PredictAt)
        x_query = grid;
    else
        x_query = opts.PredictAt(:);
    end

    if opts.KMConstant < 1.0
        model = fredholm_nn.models.FredholmNN_KM( ...
            gd, kernel, additive, dz, opts.NIterations, opts.KMConstant);
    else
        model = fredholm_nn.models.FredholmNN( ...
            gd, kernel, additive, dz, opts.NIterations);
    end

    f_hat = model.forward(x_query);

    sol.x     = x_query;
    sol.f     = f_hat;
    sol.model = model;
    sol.error = @(exact_fn) abs(f_hat - exact_fn(x_query));
    sol.mse   = @(exact_fn) mean((f_hat - exact_fn(x_query)).^2);
end
