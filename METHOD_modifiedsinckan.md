# ModifiedSincKAN Method Draft

## Method

We build on the input-conditioned mixing idea of modifiedMLP and propose a stronger architecture, named **ModifiedSincKAN**, for scientific function approximation and PDE solving. The original modifiedMLP generates two global templates from the input and reuses them across layers. While effective, its two template branches are shallow linear maps and the hidden feature itself is directly reused as the mixing weight. This couples representation learning and gating, which limits flexibility on singular, oscillatory, and multi-scale targets.

To address this issue, we replace the shallow template generators with three Sinc-based branches and introduce an explicit gate branch. Let the normalized input be denoted by `x`. We first construct three input-conditioned branches:

```math
u(x) = \sigma_u(\mathrm{Sinc}_u(x)), \qquad
v(x) = \sigma_v(\mathrm{Sinc}_v(x)), \qquad
g(x) = \mathrm{Sinc}_g(x),
```

where `u(x)` and `v(x)` are two content templates, `g(x)` is a gate prior, and `\sigma_u, \sigma_v` denote the activation used after the Sinc branches.

Starting from the first hidden state

```math
y_0 = W_0 x + b_0,
```

the `l`-th hidden block computes

```math
h_l = \phi(W_{l-1} y_{l-1} + b_{l-1}),
```

```math
\mathrm{gate}_l = \mathrm{sigmoid}(g(x) + h_l W_g + b_g),
```

```math
y_l = h_l + \mathrm{gate}_l \odot u(x) + (1 - \mathrm{gate}_l) \odot v(x),
```

where `\phi` is the backbone activation function and `\odot` denotes element-wise multiplication. The final prediction is produced by the last linear layer after the final mixed hidden feature.

This design has three advantages. First, the Sinc branches provide a richer inductive bias than shallow linear `u/v` branches, which is especially useful for singular and oscillatory structures. Second, the explicit gate branch decouples **content generation** from **mixing control**, making the hidden state responsible for representation while the gate determines how much information should be injected from each input-conditioned content branch. Third, the additive update `y_l = h_l + (...)` preserves the layerwise refinement behavior of modifiedMLP while giving the network a direct way to inject input-dependent corrections at every hidden block.

Compared with the original modifiedMLP, the proposed ModifiedSincKAN therefore replaces linear global templates with Sinc-based content branches and upgrades hidden-state-driven implicit mixing to explicit gated input-conditioned augmentation. This makes the architecture better suited for scientific machine learning problems where the target function exhibits local singularities, high-frequency oscillations, or multi-scale variation.

## Comparison To ModifiedMLP

The original modifiedMLP can be written as

```math
u(x) = \phi(W_u x + b_u), \qquad
v(x) = \phi(W_v x + b_v),
```

```math
h_l = \phi(W_{l-1} y_{l-1} + b_{l-1}),
```

```math
y_l = h_l \odot u(x) + (1 - h_l) \odot v(x).
```

Our model instead uses

```math
y_l = h_l + \mathrm{gate}_l \odot u(x) + (1 - \mathrm{gate}_l) \odot v(x),
```

with a separate gate

```math
\mathrm{gate}_l = \mathrm{sigmoid}(g(x) + h_l W_g + b_g).
```

This modification turns the hidden state from an implicit mixing coefficient into a genuine feature state, while the new gate branch explicitly controls the interpolation between the two input-conditioned content templates.
