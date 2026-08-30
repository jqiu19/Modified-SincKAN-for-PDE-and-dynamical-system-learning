import time
import os

from absl import logging

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax import vmap, jacrev
from jax.tree_util import tree_map

from flax import jax_utils

import ml_collections
import matplotlib.pyplot as plt

import swanlab
import shutil
from flax import core
from jaxpi.archs import Embedding
from jaxpi.models import _create_train_state
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset, sample_points_on_square_boundary


def _as_list(value):
    if value is None:
        return None
    return list(value)


def _set_adaptive_arch(config, idx):
    """Update ModifiedSincKAN basis size for the current curriculum stage."""
    adaptive = getattr(config.training, "adaptive_basis", None)
    if adaptive is None:
        return False

    degrees = _as_list(getattr(adaptive, "degree", None))
    len_h = _as_list(getattr(adaptive, "len_h", None))

    with config.unlocked():
        if degrees is not None:
            degree = int(degrees[idx])
            config.arch.degree = degree
            config.arch.u_degree = degree
            config.arch.v_degree = degree
            config.arch.g_degree = degree
        if len_h is not None:
            h = int(len_h[idx])
            config.arch.len_h = h
            config.arch.u_len_h = h
            config.arch.v_len_h = h
            config.arch.g_len_h = h

    return True


def _set_stage_lr(config, idx):
    lrs = _as_list(getattr(config.training, "lr_by_Re", None))
    if lrs is None:
        return False

    with config.unlocked():
        config.optim.learning_rate = float(lrs[idx])

    return True


def _rebuild_optimizer(config, model):
    """Reset the optimizer for a new curriculum stage while keeping params."""
    old_state = jax.device_get(tree_map(lambda x: x[0], model.state))
    new_state = jax.device_get(
        tree_map(
            lambda x: x[0],
            _create_train_state(
                config,
                params=old_state.params,
                weights=old_state.weights,
            ),
        )
    )
    new_state = new_state.replace(step=old_state.step)
    model.state = jax_utils.replicate(new_state)
    return model


def _expand_sinc_coeffs(old, new):
    """Embed old sinc coefficients into a larger degree/len_h basis.

    Exact old basis positions are copied. Newly introduced h-levels/frequencies
    are initialized from the nearest trained pattern with a small amplitude so
    the old solution is preserved while the new basis can start learning.
    """
    old = np.asarray(old)
    new = np.asarray(new).copy()
    old_in, old_out, old_len_h, old_d = old.shape
    new_in, new_out, new_len_h, new_d = new.shape
    if old_in != new_in or old_out != new_out:
        return new

    old_degree = old_d - 1
    new_degree = new_d - 1
    old_k = np.arange(-old_degree // 2, old_degree // 2 + 1)
    new_k = np.arange(-new_degree // 2, new_degree // 2 + 1)
    old_k_to_idx = {int(k): i for i, k in enumerate(old_k)}

    for h_idx in range(new_len_h):
        src_h = min(h_idx, old_len_h - 1)
        is_old_h = h_idx < old_len_h
        for new_k_idx, k in enumerate(new_k):
            if int(k) in old_k_to_idx:
                src_k = old_k_to_idx[int(k)]
                scale = 1.0 if is_old_h else 1e-3
            else:
                src_k = int(np.argmin(np.abs(old_k - k)))
                scale = 1e-3
            new[:, :, h_idx, new_k_idx] = scale * old[:, :, src_h, src_k]

    return new


def _merge_params(old, new, path=()):
    if isinstance(old, core.FrozenDict):
        old = core.unfreeze(old)
    if isinstance(new, core.FrozenDict):
        new = core.unfreeze(new)

    if isinstance(old, dict) and isinstance(new, dict):
        merged = {}
        for key, new_value in new.items():
            if key in old:
                merged[key] = _merge_params(old[key], new_value, path + (key,))
            else:
                merged[key] = new_value
        return core.freeze(merged) if isinstance(new, core.FrozenDict) else merged

    old_shape = getattr(old, "shape", None)
    new_shape = getattr(new, "shape", None)
    if old_shape == new_shape:
        return old

    if path[-1:] == ("coeffs",) and old_shape is not None and new_shape is not None:
        if len(old_shape) == 4 and len(new_shape) == 4:
            return jnp.asarray(_expand_sinc_coeffs(old, new), dtype=getattr(new, "dtype", None))

    return new


def _rebuild_model_with_transfer(config, old_model):
    old_state = jax.device_get(tree_map(lambda x: x[0], old_model.state))
    new_model = models.NavierStokes2D(config)
    new_state = jax.device_get(tree_map(lambda x: x[0], new_model.state))

    merged_params = _merge_params(old_state.params, new_state.params)
    new_state = new_state.replace(
        step=old_state.step,
        params=merged_params,
        weights=old_state.weights,
    )
    new_model.state = jax_utils.replicate(new_state)
    return new_model


def train_curriculum(config, workdir, model, step_offset, max_steps, Re):
    # Get dataset
    u_ref, v_ref, x_star, y_star, nu = get_dataset(Re)
    U_ref = jnp.sqrt(u_ref**2 + v_ref**2)

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Define domain
    dom = jnp.array([[x0, x1], [y0, y1]])

    # Initialize  residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size))

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    # Update  viscosity
    nu = 1 / Re

    # jit warm up
    print("Waiting for JIT...")
    for step in range(max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        model.state = model.step(model.state, batch, nu)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch, nu)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, x_star, y_star, U_ref, nu)
                swanlab.log(log_dict, step + step_offset)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.swanlab.name, "ckpt",  "Re{}".format(Re))
                if os.path.exists(ckpt_path):
                    if os.path.isfile(ckpt_path):
                        os.remove(ckpt_path)
                    else:
                        shutil.rmtree(ckpt_path)
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    # Get step offset
    step_offset = step + step_offset

    return model, step_offset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize SwanLab
    swanlab_config = config.swanlab
    swanlab_kwargs = {}
    if hasattr(swanlab_config, "mode") and swanlab_config.mode is not None:
        swanlab_kwargs["mode"] = swanlab_config.mode
    swanlab.init(project=swanlab_config.project, name=swanlab_config.name, **swanlab_kwargs)

    # Initialize model
    model = models.NavierStokes2D(config)

    # Curriculum training
    step_offset = 0

    assert len(config.training.max_steps) == len(config.training.Re)
    num_Re = len(config.training.Re)

    for idx in range(num_Re):
        # Set Re and maximum number of training steps
        changed_basis = _set_adaptive_arch(config, idx)
        changed_lr = _set_stage_lr(config, idx)
        if changed_basis:
            print(
                "Adaptive basis for Re = {}: degree = {}, len_h = {}".format(
                    config.training.Re[idx], config.arch.degree, config.arch.len_h
                )
            )
            if idx == 0:
                model = models.NavierStokes2D(config)
            else:
                model = _rebuild_model_with_transfer(config, model)
        elif changed_lr:
            model = _rebuild_optimizer(config, model)

        if changed_lr:
            print(
                "Learning rate for Re = {}: {}".format(
                    config.training.Re[idx], config.optim.learning_rate
                )
            )

        Re = config.training.Re[idx]
        max_steps = config.training.max_steps[idx]
        print("Training for Re = {}".format(Re))
        model, step_offset = train_curriculum(
            config, workdir, model, step_offset, max_steps, Re
        )

    return model
