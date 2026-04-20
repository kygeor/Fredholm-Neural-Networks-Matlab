function gd = make_grid_dictionary(grid, n_iterations)
% MAKE_GRID_DICTIONARY  Build the grid_dictionary struct for FredholmNN.
%
%   gd = make_grid_dictionary(grid, n_iterations)
%
%   Creates a struct with fields 'layer_0', 'layer_1', ..., 'layer_K'
%   where K = n_iterations.  Every layer uses the same grid by default.
%
%   Parameters
%   ----------
%   grid         : column vector — the integration grid.
%   n_iterations : integer K — number of FNN iterations (hidden layers).
%
%   Returns
%   -------
%   gd : struct with fields layer_0 ... layer_{n_iterations}.
%
%   Example
%   -------
%   [z, dz] = fredholm_nn.utils.make_uniform_grid(0, 1, 300);
%   gd      = fredholm_nn.utils.make_grid_dictionary(z, 15);

gd = struct();
for i = 0:n_iterations
    gd.(sprintf('layer_%d', i)) = grid;
end
end
