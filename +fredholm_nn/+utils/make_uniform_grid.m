function [grid, dz] = make_uniform_grid(a, b, n_points)
% MAKE_UNIFORM_GRID  Create a uniform grid on [a, b].
%
%   [grid, dz] = make_uniform_grid(a, b, n_points)
%
%   Returns a column vector of n_points uniformly spaced values from a to b
%   (inclusive) and the step size dz = (b-a)/(n_points-1).
%
%   Parameters
%   ----------
%   a, b      : scalar — domain endpoints.
%   n_points  : integer — number of grid points (>= 2).
%
%   Returns
%   -------
%   grid : (n_points x 1) column vector.
%   dz   : scalar step size.
%
%   Example
%   -------
%   [z, dz] = fredholm_nn.utils.make_uniform_grid(0, 1, 501);

if n_points < 2
    error('fredholm_nn:make_uniform_grid', 'n_points must be >= 2.');
end

grid = linspace(a, b, n_points)';   % column vector
dz   = (b - a) / (n_points - 1);
end
