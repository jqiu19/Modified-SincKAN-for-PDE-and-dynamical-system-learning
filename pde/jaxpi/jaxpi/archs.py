from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
import numpy as np

from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


def _stable_sinc(x, eps=1e-3):
    """Compute sin(x) / x with a Taylor branch around zero."""
    x2 = x ** 2
    taylor = (
        1.0
        - x2 / 6.0
        + (x2 ** 2) / 120.0
        - (x2 ** 3) / 5040.0
        + (x2 ** 4) / 362880.0
        - (x2 ** 5) / 39916800.0
    )
    safe_x = jnp.where(jnp.abs(x) < eps, 1.0, x)
    sinc = jnp.sin(safe_x) / safe_x
    return jnp.where(jnp.abs(x) < eps, taylor, sinc)


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []

        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Embedding(nn.Module):
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        return x


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel) # PINN里有时会构造一个满足某些物理结构边界条件初始条件的输出层初始化，让网络一开始就更接近合理解，减少训练早期的偏移

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class Bottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)

        x = (
            x + identity
        )  # Please note that the skip connection is added before the activation function, which is the same as the original ResNet

        x = self.activation_fn(x)

        return x


class PIBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        """
        Physics-informed bottleneck block: Add the skip connection after the activation function,
        which is different from the original ResNet, making it an identity mapping at initialization
        """
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        # alpha = jnp.exp(-alpha)

        x = alpha * x + (1 - alpha) * identity

        return x


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity

        return x


class ModifiedSincBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    gate_mode: str
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v, g):
        h = Dense(features=self.output_dim, reparam=self.reparam)(x)
        h = self.activation_fn(h)

        gate = Dense(features=self.output_dim, reparam=self.reparam)(h)
        gate = nn.sigmoid(gate + g)
        candidate = gate * u + (1.0 - gate) * v

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        if self.gate_mode == "separate_residual":
            x = h + alpha * candidate
        elif self.gate_mode == "separate_sigmoid":
            x = (1.0 - alpha) * h + alpha * candidate
        else:
            raise NotImplementedError(f"Gate mode {self.gate_mode} not supported yet!")

        return x


class ResNet(nn.Module):
    arch_name: Optional[str] = "ResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Bottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                reparam=self.reparam,
            )(x)

        y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class PIResNet(nn.Module):
    arch_name: Optional[str] = "PIResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = PIBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class PirateNet(nn.Module):
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        embs = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        x = embs

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        u = self.activation_fn(u)

        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x, u, v)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y

class SincLayers(nn.Module):
    output_dim: int
    degree: int
    init_h: float
    len_h: int = 2
    decay: str = 'inverse'
    skip: bool = True
    initialization: str = 'None'
    skip_mode: int = 1
    sinc_mode: str = 'stable'
    reparam: Union[None, Dict] = None

    def setup(self):
        # k for (1, 1, D)
        d = int(self.degree)
        k = np.arange(-d // 2, d // 2 + 1)
        k = np.expand_dims(k, axis=(0, 1))
        self.k = jnp.array(k)
        if self.decay == 'inverse':
            h = 1 / (self.init_h * (1 + np.arange(self.len_h)))
        elif self.decay == 'exp':
            h = 1 / (self.init_h ** (1 + np.arange(self.len_h)))
        else:
            raise ValueError(...)
        h = np.expand_dims(h, axis=(0, 2))
        self.h = jnp.array(h)

    @nn.compact
    def __call__(self, x): #此call已经被后面的call覆盖了
        # x shape: (input_dim,) or (batch, input_dim)
        D = self.degree + 1
        input_dim = x.shape[-1]

        shape_coeffs = (input_dim, self.output_dim, self.len_h, D)
        # --- Parameter Initialization: optionally modifiable via reparam dict ---
        if self.reparam is not None and "coeffs_init" in self.reparam:
            coeffs_init = self.reparam["coeffs_init"]
        elif self.initialization == 'Xavier':
            coeffs_init = lambda key, shape: random.normal(key, shape) / jnp.sqrt(input_dim * D)
        elif self.initialization in ('zero', 'zeros'):
            coeffs_init = lambda key, shape: jnp.zeros(shape)
        else:
            coeffs_init = lambda key, shape: random.normal(key, shape)
        coeffs = self.param('coeffs', coeffs_init, shape_coeffs)

        if self.skip:
            weight1 = self.param('weight1', lambda key, shape: jnp.zeros(shape), (1,))
            weight2 = self.param('weight2', lambda key, shape: jnp.ones(shape), (1,))
            y_eqt = Dense(features=self.output_dim, reparam=self.reparam)(x)  # (output_dim,) or (batch, output_dim)

        # --- Forward computation ---
        # (input_dim, len_h, D)
        x_expanded = jnp.tile(jnp.expand_dims(x, axis=(1, 2)), (1, 1, self.degree + 1))
        xx = x_expanded / self.h - self.k  # self.h, self.k already broadcastable

        if self.sinc_mode == 'jax':
            x_interp = jnp.sinc(xx / jnp.pi)
        else:
            x_interp = _stable_sinc(xx)
        # einsum (input_dim, len_h, D), (input_dim, output_dim, len_h, D) -> (output_dim)
        y = jnp.einsum('ild,iold->o', x_interp, coeffs)

        if self.skip:
            if self.skip_mode == 1:
                y = y_eqt + y
            elif self.skip_mode == 2:
                y = weight2 * y_eqt + weight1 * y
            elif self.skip_mode == 3:
                y = (1.0 - weight1) * y + weight1 * y_eqt

        return y

    @nn.compact
    def __call__(self, x,embs):
        # x shape: (input_dim,) or (batch, input_dim)
        D = self.degree + 1
        input_dim = x.shape[-1]

        shape_coeffs = (input_dim, self.output_dim, self.len_h, D)
        # --- Parameter Initialization: optionally modifiable via reparam dict ---
        if self.reparam is not None and "coeffs_init" in self.reparam:
            coeffs_init = self.reparam["coeffs_init"]
        elif self.initialization == 'Xavier':
            coeffs_init = lambda key, shape: random.normal(key, shape) / jnp.sqrt(input_dim * D)
        elif self.initialization in ('zero', 'zeros'):
            coeffs_init = lambda key, shape: jnp.zeros(shape)
        else:
            coeffs_init = lambda key, shape: random.normal(key, shape)
        coeffs = self.param('coeffs', coeffs_init, shape_coeffs)

        if self.skip:
            weight1 = self.param('weight1', lambda key, shape: jnp.zeros(shape), (1,))
            weight2 = self.param('weight2', lambda key, shape: jnp.ones(shape), (1,))
            y_eqt = Dense(features=self.output_dim, reparam=self.reparam)(embs)  # (output_dim,) or (batch, output_dim)

        # --- Forward computation ---
        # (input_dim, len_h, D)
        x_expanded = jnp.tile(jnp.expand_dims(x, axis=(1, 2)), (1, 1, self.degree + 1))
        xx = 2*(x_expanded-0.5) / self.h - self.k  # self.h, self.k already broadcastable

        if self.sinc_mode == 'jax':
            x_interp = jnp.sinc(xx / jnp.pi)
        else:
            x_interp = _stable_sinc(xx)
        # einsum (input_dim, len_h, D), (input_dim, output_dim, len_h, D) -> (output_dim)
        y = jnp.einsum('ild,iold->o', x_interp, coeffs)

        if self.skip:
            if self.skip_mode == 1:
                y = y_eqt + y
            elif self.skip_mode == 2:
                y = weight2 * y_eqt + weight1 * y
            elif self.skip_mode == 3:
                y = (1.0 - weight1) * y + weight1 * y_eqt

        return y
class SincKAN(nn.Module):
    arch_name: Optional[str] = "SincKAN"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    degree: int = 16
    init_h: float =2.0
    len_h: int = 2
    decay: str = 'inverse'
    skip: bool = True
    initialization: str = 'None'
    skip_mode: int = 1
    sinc_mode: str = 'stable'
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        embs = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        u = SincLayers(output_dim=self.hidden_dim, degree=self.degree, init_h=self.init_h, len_h=self.len_h,
                       decay=self.decay, skip=self.skip, initialization=self.initialization, skip_mode=self.skip_mode,
                       sinc_mode=self.sinc_mode, reparam=self.reparam)(x,embs)
        u = self.activation_fn(u)

        v = SincLayers(output_dim=self.hidden_dim, degree=self.degree, init_h=self.init_h, len_h=self.len_h,
                       decay=self.decay, skip=self.skip, initialization=self.initialization, skip_mode=self.skip_mode,
                       sinc_mode=self.sinc_mode, reparam=self.reparam)(x,embs)
        v = self.activation_fn(v)

        x = embs

        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x, u, v)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class ModifiedSincKAN(nn.Module):
    arch_name: Optional[str] = "ModifiedSincKAN"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.1
    gate_mode: str = "separate_sigmoid"
    degree: int = 16
    init_h: float = 2.0
    len_h: int = 2
    u_degree: Optional[int] = None
    v_degree: Optional[int] = None
    g_degree: Optional[int] = None
    u_len_h: Optional[int] = None
    v_len_h: Optional[int] = None
    g_len_h: Optional[int] = None
    decay: str = 'inverse'
    skip: bool = True
    initialization: str = 'None'
    skip_mode: int = 1
    sinc_mode: str = 'stable'
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        embs = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        u_degree = self.degree if self.u_degree is None else self.u_degree
        v_degree = self.degree if self.v_degree is None else self.v_degree
        g_degree = self.degree if self.g_degree is None else self.g_degree
        u_len_h = self.len_h if self.u_len_h is None else self.u_len_h
        v_len_h = self.len_h if self.v_len_h is None else self.v_len_h
        g_len_h = self.len_h if self.g_len_h is None else self.g_len_h

        u = SincLayers(output_dim=self.hidden_dim, degree=u_degree, init_h=self.init_h, len_h=u_len_h,
                       decay=self.decay, skip=self.skip, initialization=self.initialization, skip_mode=self.skip_mode,
                       sinc_mode=self.sinc_mode, reparam=self.reparam)(x, embs)
        u = self.activation_fn(u)

        v = SincLayers(output_dim=self.hidden_dim, degree=v_degree, init_h=self.init_h, len_h=v_len_h,
                       decay=self.decay, skip=self.skip, initialization=self.initialization, skip_mode=self.skip_mode,
                       sinc_mode=self.sinc_mode, reparam=self.reparam)(x, embs)
        v = self.activation_fn(v)

        g = SincLayers(output_dim=self.hidden_dim, degree=g_degree, init_h=self.init_h, len_h=g_len_h,
                       decay=self.decay, skip=self.skip, initialization=self.initialization, skip_mode=self.skip_mode,
                       sinc_mode=self.sinc_mode, reparam=self.reparam)(x, embs)

        x = embs

        for _ in range(self.num_layers):
            x = ModifiedSincBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                gate_mode=self.gate_mode,
                reparam=self.reparam,
            )(x, u, v, g)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class OriginalSincLayers(nn.Module):
    output_dim: int
    degree: int
    init_h: float
    len_h: int = 2
    decay: str = 'inverse'
    skip: bool = True
    initialization: str = 'Xavier'
    skip_mode: int = 1
    sinc_mode: str = 'stable'
    activation: str = 'tanh'
    apply_activation: bool = True
    activation_scale: float = 1.0
    reparam: Union[None, Dict] = None

    def setup(self):
        d = int(self.degree)
        k = np.arange(-np.floor(d / 2), np.ceil(d / 2) + 1)
        self.k = jnp.array(np.expand_dims(k, axis=(0, 1)))
        if self.decay == 'inverse':
            h = 1 / (self.init_h * (1 + np.arange(self.len_h)))
        elif self.decay == 'exp':
            h = 1 / (self.init_h ** (1 + np.arange(self.len_h)))
        else:
            raise ValueError(...)
        self.h = jnp.array(np.expand_dims(h, axis=(0, 2)))
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        D = self.degree + 1
        input_dim = x.shape[-1]
        shape_coeffs = (input_dim, self.output_dim, self.len_h, D)

        if self.reparam is not None and "coeffs_init" in self.reparam:
            coeffs_init = self.reparam["coeffs_init"]
        elif self.initialization == 'Xavier':
            coeffs_init = lambda key, shape: random.normal(key, shape) / jnp.sqrt(input_dim * D)
        elif self.initialization in ('zero', 'zeros'):
            coeffs_init = lambda key, shape: jnp.zeros(shape)
        else:
            coeffs_init = lambda key, shape: random.normal(key, shape)
        coeffs = self.param('coeffs', coeffs_init, shape_coeffs)

        if self.skip:
            weight1 = self.param('weight1', lambda key, shape: jnp.zeros(shape), (1,))
            weight2 = self.param('weight2', lambda key, shape: jnp.ones(shape), (1,))
            y_eqt = Dense(features=self.output_dim, reparam=self.reparam)(x)

        if self.apply_activation and self.activation == "tanh":
            beta = self.activation_scale
            x_basis = beta * jnp.tanh(x / beta)
        elif self.apply_activation:
            x_basis = self.activation_fn(x)
        else:
            x_basis = x
        x_basis = jnp.tile(jnp.expand_dims(x_basis, axis=(1, 2)), (1, 1, D))
        xx = x_basis / self.h + self.k

        if self.sinc_mode == 'jax':
            x_interp = jnp.sinc(xx)
        else:
            x_interp = _stable_sinc(jnp.pi * xx)

        y = jnp.einsum('ild,iold->o', x_interp, coeffs)

        if self.skip:
            if self.skip_mode == 1:
                y = y_eqt + y
            elif self.skip_mode == 2:
                y = weight2 * y_eqt + weight1 * y
            elif self.skip_mode == 3:
                y = (1.0 - weight1) * y_eqt + weight1 * y

        return y


class OriginalDenseBasisLayers(nn.Module):
    output_dim: int
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        return Dense(features=self.output_dim, reparam=self.reparam)(x)


class OriginalChebyLayers(nn.Module):
    output_dim: int
    degree: int
    initialization: str = 'Xavier'
    skip: bool = True
    activation: str = 'tanh'
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)
        self.k = jnp.arange(0, self.degree + 1, dtype=jnp.float32)

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        D = self.degree + 1
        if self.initialization == 'Xavier':
            coeffs_init = lambda key, shape: random.normal(key, shape) / jnp.sqrt(input_dim * D)
        elif self.initialization in ('zero', 'zeros'):
            coeffs_init = lambda key, shape: jnp.zeros(shape)
        else:
            coeffs_init = lambda key, shape: random.normal(key, shape)
        coeffs = self.param('coeffs', coeffs_init, (input_dim, self.output_dim, D))

        x_basis = jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = jnp.expand_dims(jnp.arccos(x_basis), axis=-1)
        basis = jnp.cos(theta * self.k)
        y = jnp.einsum('id,iod->o', basis, coeffs)
        if self.skip:
            y = y + Dense(features=self.output_dim, reparam=self.reparam)(x)
        return y


class OriginalJacobiLayers(nn.Module):
    output_dim: int
    degree: int
    alpha: float = 0.0
    beta: float = 0.0
    frac_power: float = 0.75
    initialization: str = 'Xavier'
    skip: bool = True
    reparam: Union[None, Dict] = None

    def _jacobi_basis(self, x):
        x = jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
        # Fractional Jacobi-style warp: preserves sign while changing spectral density.
        x = jnp.sign(x) * (jnp.abs(x) + 1e-6) ** self.frac_power
        x = jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)

        basis = [jnp.ones_like(x)]
        if self.degree >= 1:
            a = self.alpha
            b = self.beta
            basis.append(0.5 * ((a - b) + (a + b + 2.0) * x))
            for n in range(1, self.degree):
                nf = float(n)
                a1 = 2.0 * (nf + 1.0) * (nf + a + b + 1.0) * (2.0 * nf + a + b)
                a2 = (2.0 * nf + a + b + 1.0) * (a * a - b * b)
                a3 = (
                    (2.0 * nf + a + b)
                    * (2.0 * nf + a + b + 1.0)
                    * (2.0 * nf + a + b + 2.0)
                )
                a4 = 2.0 * (nf + a) * (nf + b) * (2.0 * nf + a + b + 2.0)
                denom = jnp.where(jnp.abs(a1) < 1e-8, 1e-8, a1)
                p_next = ((a2 + a3 * x) * basis[-1] - a4 * basis[-2]) / denom
                basis.append(p_next)
        return jnp.stack(basis, axis=-1)

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        D = self.degree + 1
        if self.initialization == 'Xavier':
            coeffs_init = lambda key, shape: random.normal(key, shape) / jnp.sqrt(input_dim * D)
        elif self.initialization in ('zero', 'zeros'):
            coeffs_init = lambda key, shape: jnp.zeros(shape)
        else:
            coeffs_init = lambda key, shape: random.normal(key, shape)
        coeffs = self.param('coeffs', coeffs_init, (input_dim, self.output_dim, D))
        basis = self._jacobi_basis(x)
        y = jnp.einsum('id,iod->o', basis, coeffs)
        if self.skip:
            y = y + Dense(features=self.output_dim, reparam=self.reparam)(x)
        return y


class OriginalSplineLayers(nn.Module):
    output_dim: int
    degree: int
    interval: Tuple[float, float] = (-1.0, 1.0)
    initialization: str = 'Xavier'
    skip: bool = True
    activation: str = 'tanh'
    reparam: Union[None, Dict] = None

    def setup(self):
        self.k = 3
        self.activation_fn = _get_activation(self.activation)

    def _grid(self, input_dim):
        num_basis = max(int(self.degree), self.k + 1)
        G = num_basis - self.k
        dx = (self.interval[1] - self.interval[0]) / G
        grid = jnp.arange(-self.k, G + self.k + 1, dtype=jnp.float32) * dx + self.interval[0]
        return jnp.tile(jnp.expand_dims(grid, axis=0), (input_dim, 1)), num_basis

    def _basis(self, x, grid):
        x = jnp.expand_dims(jnp.clip(x, self.interval[0] + 1e-6, self.interval[1] - 1e-6), axis=1)
        basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(jnp.float32)
        for k in range(1, self.k + 1):
            left_denom = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_denom = grid[:, k + 1:] - grid[:, 1:(-k)]
            left_denom = jnp.where(jnp.abs(left_denom) < 1e-8, 1.0, left_denom)
            right_denom = jnp.where(jnp.abs(right_denom) < 1e-8, 1.0, right_denom)
            left = (x - grid[:, :-(k + 1)]) / left_denom
            right = (grid[:, k + 1:] - x) / right_denom
            basis = left * basis[:, :-1] + right * basis[:, 1:]
        return basis

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        grid, num_basis = self._grid(input_dim)
        if self.initialization == 'Xavier':
            coeffs_init = lambda key, shape: random.normal(key, shape) / jnp.sqrt(input_dim * num_basis)
        elif self.initialization in ('zero', 'zeros'):
            coeffs_init = lambda key, shape: jnp.zeros(shape)
        else:
            coeffs_init = lambda key, shape: random.normal(key, shape)
        coeffs = self.param('coeffs', coeffs_init, (input_dim, self.output_dim, num_basis))
        scales = self.param('scales', lambda key, shape: jnp.ones(shape), (input_dim, self.output_dim))
        basis = self._basis(x, grid)
        spl = jnp.einsum('id,iod->io', basis, coeffs)
        y = jnp.mean(spl * scales, axis=0)
        if self.skip:
            y = y + Dense(features=self.output_dim, reparam=self.reparam)(self.activation_fn(x))
        return y


def _build_original_basis_layer(
    basis_type,
    output_dim,
    degree,
    init_h,
    len_h,
    decay,
    skip,
    initialization,
    skip_mode,
    sinc_mode,
    activation,
    reparam,
    jacobi_alpha,
    jacobi_beta,
    jacobi_frac_power,
    sinc_apply_activation=True,
    sinc_activation_scale=1.0,
):
    if basis_type in ("sinc", "sinckan"):
        return OriginalSincLayers(
            output_dim=output_dim, degree=degree, init_h=init_h,
            len_h=len_h, decay=decay, skip=skip,
            initialization=initialization, skip_mode=skip_mode,
            sinc_mode=sinc_mode, activation=activation,
            apply_activation=sinc_apply_activation,
            activation_scale=sinc_activation_scale, reparam=reparam,
        )
    if basis_type in ("linear", "dense", "dense_linear"):
        return OriginalDenseBasisLayers(output_dim=output_dim, reparam=reparam)
    if basis_type in ("cheby", "chebyshev", "chebykan"):
        return OriginalChebyLayers(
            output_dim=output_dim, degree=degree, initialization=initialization,
            skip=skip, activation=activation, reparam=reparam,
        )
    if basis_type in ("spline", "cubic_spline", "kan"):
        return OriginalSplineLayers(
            output_dim=output_dim, degree=degree, initialization=initialization,
            skip=skip, activation=activation, reparam=reparam,
        )
    if basis_type in ("jacobi", "fractional_jacobi", "fkan"):
        return OriginalJacobiLayers(
            output_dim=output_dim, degree=degree, alpha=jacobi_alpha,
            beta=jacobi_beta, frac_power=jacobi_frac_power,
            initialization=initialization, skip=skip, reparam=reparam,
        )
    raise ValueError(f"Unsupported original basis type: {basis_type}")


class ModifiedSincKANOriginal(nn.Module):
    arch_name: Optional[str] = "ModifiedSincKANOriginal"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 3
    activation: str = "tanh"
    nonlinearity: float = 0.1
    gate_mode: str = "separate_sigmoid"
    degree: int = 16
    init_h: float = 2.0
    len_h: int = 3
    u_degree: Optional[int] = None
    v_degree: Optional[int] = None
    g_degree: Optional[int] = None
    u_len_h: Optional[int] = None
    v_len_h: Optional[int] = None
    g_len_h: Optional[int] = None
    u_basis: str = "sinc"
    v_basis: str = "sinc"
    g_basis: str = "sinc"
    jacobi_alpha: float = 0.0
    jacobi_beta: float = 0.0
    jacobi_frac_power: float = 0.75
    decay: str = 'inverse'
    skip: bool = True
    initialization: str = 'Xavier'
    skip_mode: int = 1
    sinc_mode: str = 'stable'
    sinc_apply_activation: bool = True
    sinc_activation_scale: float = 1.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        # Match networks.py: normalize [0, 1]^2 coordinates to [-1, 1]^2,
        # then use plain Dense layers instead of Fourier features.
        x_in = 2.0 * (x - 0.5)

        u_degree = self.degree if self.u_degree is None else self.u_degree
        v_degree = self.degree if self.v_degree is None else self.v_degree
        g_degree = self.degree if self.g_degree is None else self.g_degree
        u_len_h = self.len_h if self.u_len_h is None else self.u_len_h
        v_len_h = self.len_h if self.v_len_h is None else self.v_len_h
        g_len_h = self.len_h if self.g_len_h is None else self.g_len_h

        u = _build_original_basis_layer(
            self.u_basis, self.hidden_dim, u_degree, self.init_h, u_len_h,
            self.decay, self.skip, self.initialization, self.skip_mode,
            self.sinc_mode, self.activation, self.reparam, self.jacobi_alpha,
            self.jacobi_beta, self.jacobi_frac_power, self.sinc_apply_activation,
            self.sinc_activation_scale,
        )(x_in)
        u = self.activation_fn(u)

        v = _build_original_basis_layer(
            self.v_basis, self.hidden_dim, v_degree, self.init_h, v_len_h,
            self.decay, self.skip, self.initialization, self.skip_mode,
            self.sinc_mode, self.activation, self.reparam, self.jacobi_alpha,
            self.jacobi_beta, self.jacobi_frac_power, self.sinc_apply_activation,
            self.sinc_activation_scale,
        )(x_in)
        v = self.activation_fn(v)

        g = _build_original_basis_layer(
            self.g_basis, self.hidden_dim, g_degree, self.init_h, g_len_h,
            self.decay, self.skip, self.initialization, self.skip_mode,
            self.sinc_mode, self.activation, self.reparam, self.jacobi_alpha,
            self.jacobi_beta, self.jacobi_frac_power, self.sinc_apply_activation,
            self.sinc_activation_scale,
        )(x_in)

        z = Dense(features=self.hidden_dim, reparam=self.reparam)(x_in)
        for layer_idx in range(self.num_layers - 1):
            h = self.activation_fn(z)
            gate = Dense(features=self.hidden_dim, reparam=self.reparam)(h)
            gate = nn.sigmoid(gate + g)
            candidate = gate * u + (1.0 - gate) * v

            alpha = self.param(f"alpha_{layer_idx}", constant(self.nonlinearity), (1,))
            if self.gate_mode == "separate_residual":
                mixed = h + alpha * candidate
            elif self.gate_mode == "separate_sigmoid":
                mixed = (1.0 - alpha) * h + alpha * candidate
            else:
                mixed = h * u + (1.0 - h) * v
            z = Dense(features=self.hidden_dim, reparam=self.reparam)(mixed)

        y = Dense(features=self.out_dim, reparam=self.reparam)(z)
        return z, y


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class KAN(nn.Module):
    arch_name: Optional[str] = "KAN"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    degree: int = 16
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        if self.periodicity is None and self.fourier_emb is None:
            x = 2.0 * (x - 0.5)

        for _ in range(self.num_layers):
            x = OriginalSplineLayers(
                output_dim=self.hidden_dim,
                degree=self.degree,
                initialization="Xavier",
                skip=True,
                activation=self.activation,
                reparam=self.reparam,
            )(x)

        y = OriginalSplineLayers(
            output_dim=self.out_dim,
            degree=self.degree,
            initialization="Xavier",
            skip=True,
            activation=self.activation,
            reparam=self.reparam,
        )(x)
        return x, y


class ChebyKAN(nn.Module):
    arch_name: Optional[str] = "ChebyKAN"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    degree: int = 16
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        if self.periodicity is None and self.fourier_emb is None:
            x = 2.0 * (x - 0.5)

        for _ in range(self.num_layers):
            x = self.activation_fn(x)
            x = OriginalChebyLayers(
                output_dim=self.hidden_dim,
                degree=self.degree,
                initialization="Xavier",
                skip=False,
                activation=self.activation,
                reparam=self.reparam,
            )(x)

        y = OriginalChebyLayers(
            output_dim=self.out_dim,
            degree=self.degree,
            initialization="Xavier",
            skip=False,
            activation=self.activation,
            reparam=self.reparam,
        )(self.activation_fn(x))
        return x, y


#################################################################################################
#################################### neural operators ###########################################
#################################################################################################

class MlpBlock(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str
    reparam: Union[None, Dict]
    final_activation: bool

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        if self.final_activation:
            x = self.activation_fn(x)

        return x


class DeepONet(nn.Module):
    arch_name: Optional[str] = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            final_activation=False,
            reparam=self.reparam,
        )(u)

        x = Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y = u * x
        y = self.activation_fn(y)
        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y
