# Fredholm Neural Networks (MATLAB) 

A MATLAB package for solving **Fredholm integral equations of the second kind** using the Fredholm Neural Network (Fredholm NN/ FNN) framework. For background and theory see also https://github.com/kygeor/Fredholm-Neural-Networks-Py

The FNN encodes the method of successive approximations (Picard / Krasnoselskii-Mann iterations) directly into the weights and biases of a deep network with linear activations. No training is required — the network is constructed analytically from the kernel and free term. 

The MATLAB toolbox also contains the functionalities necessary to run Fredholm Neural Networks for inverse problems (theory detailed below and in the corresponding papers). 

**Supported problem types:**

| Problem | Solver |
|---|---|
| Linear FIE (contractive operator) | `fredholm_nn.solvers.solve_linear_fie` |
| Linear FIE (non-expansive operator, KM) | `fredholm_nn.solvers.solve_linear_fie(..., 'KMConstant', κ)` |
| Second-order BVP ODE | `fredholm_nn.solvers.solve_bvp_ode` |
| Inverse kernel problem | `fredholm_nn.solvers.solve_inverse_kernel` |

---

## Citation

If you use this package please cite:

```bibtex
@article{georgiou2025fredholm,
  title   = {Fredholm neural networks},
  author  = {Georgiou, Kyriakos and Siettos, Constantinos and Yannacopoulos, Athanasios N},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {47}, number = {4}, pages = {C1006--C1031},
  year    = {2025}, publisher = {SIAM}
}
```
and/or

```bibtex
@article{georgiou2025fredholm_pde,
  title   = {Fredholm Neural Networks for forward and inverse problems in elliptic PDEs},
  author  = {Georgiou, Kyriakos and Siettos, Constantinos and Yannacopoulos, Athanasios N},
  journal = {arXiv preprint arXiv:2507.06038},
  year    = {2025}
}
```

## Package structure

```
fredholm_nn_matlab/
├── +fredholm_nn/
│   ├── +models/
│   │   ├── FredholmNN.m          core FNN for linear FIEs
│   │   └── FredholmNN_KM.m       Krasnoselskii-Mann variant
│   ├── +solvers/
│   │   ├── solve_linear_fie.m
│   │   ├── solve_bvp_ode.m
│   │   └── solve_inverse_kernel.m
│   └── +utils/
│       ├── make_uniform_grid.m
│       └── make_grid_dictionary.m
└── examples/
    ├── run_examples.m            runs all examples, saves figures
    ├── example_linear_fie.m
    ├── example_linear_fie_km.m
    ├── example_bvp_ode.m
    └── example_inverse_kernel.m
```

---

## Installation

1. Download or clone this repository.
2. Add the top-level `fredholm_nn_matlab/` folder to your MATLAB path:

```matlab
addpath('/path/to/fredholm_nn_matlab')
```

Or place the call in your `startup.m` to make it permanent. No toolboxes are required beyond the **Optimization Toolbox** (used by `solve_inverse_kernel` for `lsqnonlin`).

---

## Quick start

```matlab
kernel   = @(x, z) sin(z) .* cos(x);
additive = @(x) sin(x);

sol = fredholm_nn.solvers.solve_linear_fie(kernel, additive, [0, pi/2], ...
    'NGrid', 300, 'NIterations', 10);

plot(sol.x, sol.f);
```

---

## Writing `kernel` and `additive`

| Parameter | Contract |
|---|---|
| `kernel(x, z)` | Accepts a **column** vector `x` (N×1) and a **row** vector `z` (1×M); returns an N×M matrix via implicit expansion. Use `.* ` (element-wise) operators. |
| `additive(x)` | Accepts a column vector and returns a column vector of the same size. |

---

## Solvers

### 1. Linear FIE — `solve_linear_fie`

Solves  f(x) = g(x) + ∫_a^b K(x,z) f(z) dz

```matlab
sol = fredholm_nn.solvers.solve_linear_fie(kernel, additive, domain, ...
    'NGrid',       500, ...   % quadrature points
    'NIterations',  50, ...   % Picard iterations (hidden layers)
    'PredictAt',   x_query);  % optional query points (default: grid)
```

For **non-expansive operators** (Krasnoselskii-Mann), add `'KMConstant'`:

```matlab
sol = fredholm_nn.solvers.solve_linear_fie(kernel, additive, domain, ...
    'KMConstant', 0.5);
```

**Return value — struct with fields:**

| Field | Description |
|---|---|
| `sol.x` | Query points |
| `sol.f` | Predicted solution f̂(x) |
| `sol.model` | Underlying `FredholmNN` or `FredholmNN_KM` object |
| `sol.error(exact_fn)` | Function handle: pointwise absolute error |
| `sol.mse(exact_fn)` | Function handle: mean-squared error |

---

### 2. BVP ODE — `solve_bvp_ode`

Solves  y''(x) + p(x) y(x) = q(x),  y(a) = α,  y(b) = β

```matlab
p = 3.2;
sol = fredholm_nn.solvers.solve_bvp_ode( ...
    @(x) 3*p ./ (p + x.^2).^2, ...   % p_func
    @(x) zeros(size(x)),        ...   % q_func
    0.0,                        ...   % alpha = y(a)
    1.0/sqrt(p+1),              ...   % beta  = y(b)
    [0, 1],                     ...   % domain
    'NGrid', 1000, 'NIterations', 10);
```

**Return value — struct with fields:** `sol.x`, `sol.y`, `sol.u` (= y''), `sol.error(fn)`, `sol.mse(fn)`.

---

### 3. Inverse kernel problem — `solve_inverse_kernel`

Given data `f_target` on a grid and the free term `g`, learns a kernel network K_θ minimising

  L(θ) = ||f̂(·;K_θ) − f̃||² + λ ||residual||²

```matlab
sol = fredholm_nn.solvers.solve_inverse_kernel( ...
    f_target, additive, [0, pi/2], ...
    'NGrid',       100, ...
    'NIterations',  15, ...
    'NNeurons',     20, ...
    'MaxIter',     300, ...
    'Lambda',      1e-6);
```

**Return value — struct with fields:** `sol.x`, `sol.f_pred`, `sol.net_kernel`, `sol.mse`.

---


# Fredholm Neural Networks for the inverse problem (for FIEs)

The problem consists of taking as data two functions $f, g : \Omega \to \mathbb{R}$ and modeling an unknown kernel $K : \Omega \times \Omega \to \mathbb{R}$ (e.g., a neural network) so that $f$ satisfies a target integral equation. Hence, the inverse problem is: given $\tilde{f}$ and $g$, find $K$ such that the induced integral operator admits a solution $f$ that matches $\tilde{f}$ on the chosen collocation points. 

Our strategy uses the structure/convergence of the Fredholm NN: select parameters $\theta$ so that, when constructing the estimated kernel $K_\theta$ and feeding it into the Fredholm NN with M hidden layers, the network output $\hat{f}(x;\hat K_\theta)$ is close to the data $\tilde{f}$ under an appropriate loss.


The complete loss is

$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \Big(f(x_i) - \hat{f}(x_i;\hat{K}_{\theta})\Big)^2 + \lambda_{reg}\,{R}(\theta).
$$

Here, $\hat{f}(x;\hat{K}_{\theta})$ denotes the output of the Fredholm NN and $R(\theta)$ is a Tikhonov regularization term. Training is done using the Levenberg–Marquardt algorithm.


<img width="613" height="234" alt="Screenshot 2025-10-08 at 2 26 57 PM" src="https://github.com/user-attachments/assets/5bd73f8c-0b5a-4500-bafc-7535dfb46edc" />

*Figure 3: Algorithm to solve the inverse problem using the Fredholm NN framework.*


