classdef FredholmNN_KM < handle
% FREDHOLMNN_KM  Fredholm Neural Network — Krasnoselskii-Mann variant.
%
%   Solves linear FIEs whose integral operator is non-expansive (rather
%   than strictly contractive) using the KM relaxation scheme:
%
%       f_{n+1}(x) = (1-κ) f_n(x) + κ T[f_n](x)
%
%   where κ ∈ (0,1] is the relaxation parameter.  Setting κ = 1 reduces
%   to standard Picard iteration (identical to FredholmNN).
%
%   The weight matrix at each hidden layer i > 0 becomes:
%
%       W_i = κ · K(z_{i-1}, z_i) · dz  +  (1-κ) · I
%
%   and the bias becomes:
%
%       b_i = κ · g(z_i)
%
%   Usage
%   -----
%   model = fredholm_nn.models.FredholmNN_KM(grid_dict, kernel, additive, dz, K, km_constant)
%   f_hat = model.forward(x_query)
%
%   Constructor parameters
%   ----------------------
%   grid_dict   : struct  — see FredholmNN.
%   kernel      : function handle — same broadcasting convention as FredholmNN.
%   additive    : function handle — g(x).
%   dz          : scalar — quadrature step.
%   K           : integer — number of hidden layers.
%   km_constant : scalar κ ∈ (0,1] — KM relaxation parameter.
%
%   Reference
%   ---------
%   Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
%   Fredholm neural networks. SIAM J. Sci. Comput., 47(4).

    properties
        grid_dict
        kernel
        additive
        dz
        K
        km_constant   % κ ∈ (0, 1]
    end

    methods

        function obj = FredholmNN_KM(grid_dict, kernel, additive, dz, K, km_constant)
            if km_constant <= 0 || km_constant > 1
                error('fredholm_nn:FredholmNN_KM', ...
                      'km_constant must be in (0, 1].');
            end
            obj.grid_dict   = grid_dict;
            obj.kernel      = kernel;
            obj.additive    = additive;
            obj.dz          = dz;
            obj.K           = K;
            obj.km_constant = km_constant;
        end

        % ------------------------------------------------------------------
        function [weights, biases] = build_weights_and_biases(obj)
        % BUILD_WEIGHTS_AND_BIASES  Construct KM-modified weight matrices.

            kappa   = obj.km_constant;
            weights = cell(obj.K + 1, 1);
            biases  = cell(obj.K + 1, 1);

            for i = 0:obj.K
                grid_i = obj.grid_dict.(sprintf('layer_%d', i));

                if i == 0
                    weights{1} = diag(obj.additive(grid_i));
                    biases{1}  = zeros(length(grid_i), 1);
                else
                    grid_prev = obj.grid_dict.(sprintf('layer_%d', i-1));
                    N = length(grid_prev);
                    M = length(grid_i);

                    % Core kernel matrix scaled by κ·dz
                    W = obj.kernel(grid_prev(:)', grid_i(:)) * obj.dz * kappa;

                    % KM correction: add (1-κ)·I on the diagonal
                    if N == M
                        W = W + (1 - kappa) * eye(M);
                    end

                    weights{i+1} = W;
                    biases{i+1}  = obj.additive(grid_i(:)) * kappa;
                end
            end
        end

        % ------------------------------------------------------------------
        function f_hat = forward(obj, x_query)
        % FORWARD  Evaluate the KM FNN at query points x_query.

            x_query = x_query(:);

            [weights, biases] = obj.build_weights_and_biases();

            h = ones(length(obj.grid_dict.layer_0), 1);
            for i = 1:obj.K + 1
                h = weights{i} * h + biases{i};
            end

            % Output layer (same as FredholmNN — no KM scaling here)
            last_grid = obj.grid_dict.(sprintf('layer_%d', obj.K));
            z = last_grid(:);

            K_out = obj.kernel(z, x_query(:)') * obj.dz;   % N x Q
            f_hat = K_out' * h + obj.additive(x_query);    % Q x 1
        end

    end % methods
end % classdef
