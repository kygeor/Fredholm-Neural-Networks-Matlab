classdef FredholmNN < handle
% FREDHOLMNN  Fredholm Neural Network for linear FIEs (contractive operator).
%
%   Implements the FNN architecture for linear Fredholm integral equations
%   of the second kind:
%
%       f(x) = g(x) + integral_a^b K(x,z) f(z) dz
%
%   The network has K+1 hidden layers whose weights and biases are
%   determined analytically from the kernel and free term, discretised on
%   the integration grid.  No gradient-based training is performed.
%
%   Usage
%   -----
%   model = fredholm_nn.models.FredholmNN(grid_dict, kernel, additive, dz, K)
%   f_hat = model.forward(x_query)
%
%   Constructor parameters
%   ----------------------
%   grid_dict : struct  — fields 'layer_0' ... 'layer_K', each a column
%               vector of grid points (see make_grid_dictionary).
%   kernel    : function handle  — K(x, z).  Called as kernel(z_col, x_row)
%               where z_col is N×1 and x_row is 1×M; must return N×M via
%               implicit expansion.
%   additive  : function handle  — g(x).  Called with a column vector,
%               must return a column vector of the same length.
%   dz        : scalar  — quadrature step size.
%   K         : integer — number of hidden layers (Picard iterations).
%
%   Reference
%   ---------
%   Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
%   Fredholm neural networks. SIAM J. Sci. Comput., 47(4).

    properties
        grid_dict   % struct: layer_0 ... layer_K (column vectors)
        kernel      % function handle: K(z_col, x_row) -> N x M
        additive    % function handle: g(x_col) -> column vector
        dz          % scalar: quadrature step
        K           % integer: number of iterations / hidden layers
    end

    methods

        function obj = FredholmNN(grid_dict, kernel, additive, dz, K)
            obj.grid_dict = grid_dict;
            obj.kernel    = kernel;
            obj.additive  = additive;
            obj.dz        = dz;
            obj.K         = K;
        end

        % ------------------------------------------------------------------
        function [weights, biases] = build_weights_and_biases(obj)
        % BUILD_WEIGHTS_AND_BIASES  Construct the K+1 weight matrices and
        % bias vectors from the kernel and additive term.
        %
        % Layer 0:  W0 = diag(g(z)),  b0 = 0
        % Layer i>0: Wi = K(z_{i-1}, z_i) * dz,  bi = g(z_i)

            weights = cell(obj.K + 1, 1);
            biases  = cell(obj.K + 1, 1);

            for i = 0:obj.K
                grid_i = obj.grid_dict.(sprintf('layer_%d', i));

                if i == 0
                    weights{1} = diag(obj.additive(grid_i));
                    biases{1}  = zeros(length(grid_i), 1);
                else
                    grid_prev = obj.grid_dict.(sprintf('layer_%d', i-1));
                    % grid_prev as row (1×N), grid_i as column (M×1)
                    W = obj.kernel(grid_prev(:)', grid_i(:)) * obj.dz;
                    weights{i+1} = W;
                    biases{i+1}  = obj.additive(grid_i(:));
                end
            end
        end

        % ------------------------------------------------------------------
        function f_hat = forward(obj, x_query)
        % FORWARD  Evaluate the FNN at query points x_query.
        %
        %   f_hat = model.forward(x_query)
        %
        %   x_query : column vector of query points (need not be on the grid).
        %   f_hat   : column vector of predicted solution values.

            x_query = x_query(:);   % ensure column

            [weights, biases] = obj.build_weights_and_biases();

            % Propagate through hidden layers starting from ones vector
            h = ones(length(obj.grid_dict.layer_0), 1);
            for i = 1:obj.K + 1
                h = weights{i} * h + biases{i};
            end
            % h is now the last hidden-layer output (N_grid x 1)

            % Output layer: for each query point xq,
            %   f̂(xq) = sum_j K(z_j, xq)*dz * h_j  +  g(xq)
            last_grid = obj.grid_dict.(sprintf('layer_%d', obj.K));
            z = last_grid(:);   % N_grid x 1

            % K_out: N_grid x n_query  (z as col, x_query as row)
            K_out = obj.kernel(z, x_query(:)') * obj.dz;  % N x Q
            f_hat = K_out' * h + obj.additive(x_query);   % Q x 1
        end

    end % methods
end % classdef
