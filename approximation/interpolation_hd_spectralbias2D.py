import sys

sys.path.append('../')
import argparse
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import jit, random, vmap
from jax.lax import scan
from matplotlib.colors import LogNorm
from scipy.stats import qmc

from data import get_data
from experiment_utils import append_result_row, build_results_path, finish_wandb, init_wandb_run, log_wandb
from networks import get_network
from utils import normalization_hd


parser = argparse.ArgumentParser(description="SincKAN spectral_bias_2D benchmark")
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--datatype", type=str, default="spectral_bias_2D")
parser.add_argument("--npoints", type=int, default=1000, help="mini-batch size")
parser.add_argument("--ntrain", type=int, default=100000, help="number of LHS training points")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--epochs", type=int, default=2000, help="legacy argument; total_steps controls training")
parser.add_argument("--ite", type=int, default=20, help="legacy argument; total_steps controls training")
parser.add_argument("--total_steps", type=int, default=200000, help="total number of optimizer steps")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--interval", type=str, default="-2.0,2.0")
parser.add_argument("--noise", type=int, default=0)
parser.add_argument("--normalization", type=int, default=0)
parser.add_argument("--network", type=str, default="mlp")
parser.add_argument("--kanshape", type=str, default="16")
parser.add_argument("--degree", type=int, default=64)
parser.add_argument("--features", type=int, default=100)
parser.add_argument("--layers", type=int, default=10)
parser.add_argument("--len_h", type=int, default=2)
parser.add_argument("--init_h", type=float, default=2.0)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--decay", type=str, default="inverse")
parser.add_argument("--skip", type=int, default=0)
parser.add_argument("--embed_feature", type=int, default=10)
parser.add_argument("--initialization", type=str, default="Xavier")
parser.add_argument("--uv_activation", type=str, default="none")
parser.add_argument("--gate_mode", type=str, default="separate_residual")
parser.add_argument("--u_degree", type=int, default=None)
parser.add_argument("--v_degree", type=int, default=None)
parser.add_argument("--g_degree", type=int, default=None)
parser.add_argument("--u_len_h", type=int, default=None)
parser.add_argument("--v_len_h", type=int, default=None)
parser.add_argument("--g_len_h", type=int, default=None)
parser.add_argument("--wandb", type=int, default=1)
parser.add_argument("--wandb_project", type=str, default="modifiedsinckan")
parser.add_argument("--wandb_group", type=str, default="spectral_bias2d")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)


def latin_hypercube_points(dim, interval, num, seed):
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    points = sampler.random(n=num)
    lower = np.full((dim,), interval[0], dtype=np.float32)
    upper = np.full((dim,), interval[1], dtype=np.float32)
    return qmc.scale(points, lower, upper).astype(np.float32)


def uniform_test_grid(interval, resolution=256):
    x = np.linspace(interval[0], interval[1], resolution, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)


def net(model, frozen_para, *x):
    return model(jnp.stack(*x), frozen_para)[0]


def compute_loss(model, ob_x, frozen_para, dim):
    output = vmap(net, (None, None, 0))(model, frozen_para, ob_x[:, :dim])
    return jnp.mean((output - ob_x[:, dim]) ** 2)


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def make_step(model, ob_x, frozen_para, optim, opt_state, dim):
    loss, grads = compute_loss_and_grads(model, ob_x, frozen_para, dim)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def make_eval_fn(model, frozen_para):
    @jit
    def eval_batch(carry, xb):
        yb = vmap(net, (None, None, 0))(model, frozen_para, xb)
        return carry, yb

    return eval_batch


def batched_predict(model, frozen_para, x, batch_size=4096):
    eval_batch = make_eval_fn(model, frozen_para)
    n = x.shape[0]
    n_batch = n // batch_size
    if n_batch > 0:
        x_batched = x[: n_batch * batch_size].reshape(n_batch, batch_size, -1)
        _, y_batch = scan(eval_batch, None, x_batched)
        y_pred = y_batch.reshape(-1, 1)
    else:
        y_pred = jnp.zeros((0, 1))

    rem = x[n_batch * batch_size :]
    if rem.size:
        y_rem = vmap(net, (None, None, 0))(model, frozen_para, rem)
        y_pred = jnp.concatenate([y_pred, y_rem], axis=0)
    return y_pred


def plot_three_pdfs(npz_path, interval):
    data = np.load(npz_path)
    pred = data["y_pred"]
    ref = data["y_test"]

    res = 256
    x = np.linspace(interval[0], interval[1], res)
    xx, yy = np.meshgrid(x, x, indexing="ij")

    reference = ref.reshape(xx.shape)
    prediction = pred.reshape(xx.shape)
    error = np.abs(prediction - reference)

    kw_im = dict(cmap="jet", extent=[interval[0], interval[1], interval[0], interval[1]], origin="lower", aspect="auto")

    fig0, ax0 = plt.subplots(figsize=(4.2, 3.6))
    im0 = ax0.imshow(prediction, **kw_im)
    ax0.set_xlabel("$x$", fontsize=12)
    ax0.set_ylabel("$y$", fontsize=12)
    ax0.grid(True, alpha=0.3)
    fig0.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig0.savefig("spectral_bias2D_pred.pdf", dpi=300)
    plt.close(fig0)

    fig1, ax1 = plt.subplots(figsize=(4.2, 3.6))
    im1 = ax1.imshow(reference, **kw_im)
    ax1.set_xlabel("$x$", fontsize=12)
    ax1.set_ylabel("$y$", fontsize=12)
    ax1.grid(True, alpha=0.3)
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig1.savefig("spectral_bias2D_ref.pdf", dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    im2 = ax2.imshow(error, norm=LogNorm(vmin=1e-6, vmax=max(1e-6, float(error.max()))), **kw_im)
    ax2.set_xlabel("$x$", fontsize=12)
    ax2.set_ylabel("$y$", fontsize=12)
    ax2.grid(True, alpha=0.3)
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig("spectral_bias2D_err.pdf", dpi=300)
    plt.close(fig2)


def train(key):
    dim = args.dim
    total_steps = args.total_steps
    batch_size = int(args.npoints)

    interval = [float(v) for v in args.interval.split(",")]
    x_train = latin_hypercube_points(dim=dim, interval=interval, num=args.ntrain, seed=args.seed)
    x_test = uniform_test_grid(interval=interval, resolution=256)

    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train)
    if args.noise == 1:
        y_train = y_train + np.random.normal(0, 0.01, y_train.shape)
    y_test = generate_data(x_test)

    normalizer = normalization_hd(interval, dim, args.normalization)

    input_dim = dim
    output_dim = 1
    keys = random.split(key, 3)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()

    wandb_run = init_wandb_run(args, "spectral_bias2d", extra_config={
        "input_dim": input_dim,
        "output_dim": output_dim,
        "train_sampler": "latin_hypercube",
        "test_grid_resolution": 256,
        "total_steps": total_steps,
    })

    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f"total parameters: {param_count}")

    ob_x = jnp.concatenate([x_train, y_train.reshape(-1, 1)], axis=-1)
    batch_size = max(1, min(batch_size, ob_x.shape[0]))

    warmup_steps = max(1, int(0.05 * total_steps))
    warmup = optax.linear_schedule(init_value=0.0, end_value=args.lr, transition_steps=warmup_steps)
    cosine = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=0.1,
    )
    lr_schedule = optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    history = []
    timings = []
    relative_errors = []
    mse_errors = []
    eval_interval = max(1, total_steps // 20)

    keys = random.split(keys[-1], total_steps + 2)
    for step in range(total_steps):
        perm = random.permutation(keys[step], ob_x.shape[0])
        start = (step * batch_size) % ob_x.shape[0]
        idx = perm[start : start + batch_size]
        if idx.shape[0] < batch_size:
            idx = jnp.concatenate([idx, perm[: batch_size - idx.shape[0]]], axis=0)
        input_points = ob_x[idx]

        t1 = time.time()
        loss, model, opt_state = make_step(model, input_points, frozen_para, optim, opt_state, dim)
        t2 = time.time()
        timings.append(t2 - t1)
        history.append(loss.item())

        if step % eval_interval == 0 or step == total_steps - 1:
            y_pred = batched_predict(model, frozen_para, x_test, batch_size=4096)
            mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
            relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
            mse_errors.append(mse_error)
            relative_errors.append(relative_error)
            print(f"step:{step + 1}/{total_steps}, testing mse:{mse_error:.2e}, relative:{relative_error:.2e}")
            log_wandb(wandb_run, {
                "test/mse": float(mse_error),
                "test/rmse": float(jnp.sqrt(mse_error)),
                "test/relative": float(relative_error),
                "train/loss": float(loss),
            }, step=step + 1)

    avg_time = float(np.mean(np.array(timings)))
    print(f"time: {1 / avg_time:.2e}ite/s")

    y_pred = batched_predict(model, frozen_para, x_test, batch_size=4096)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    rmse_error = jnp.sqrt(mse_error)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    print(f"final, testing mse:{mse_error:.2e}, rmse:{rmse_error:.2e}, relative:{relative_error:.2e}")

    path = f"{args.datatype}_{args.network}_{args.seed}_{args.dim}.eqx"
    eqx.tree_serialise_leaves(path, model)
    path = f"{args.datatype}_{args.network}_{args.seed}_{args.dim}.npz"
    np.savez(
        path,
        loss=history,
        avg_time=avg_time,
        y_pred=y_pred,
        y_test=y_test,
        relative_errors=relative_errors,
        mse_errors=mse_errors,
        x_train=x_train,
        x_test=x_test,
    )

    header = [
        "datatype", "network", "seed", "final_loss_mean", "training_time", "total_ite",
        "test_mse", "test_rmse", "test_relative"
    ]
    save_here = build_results_path(args)
    append_result_row(save_here, header, [
        args.datatype, args.network, args.seed, history[-1], float(np.sum(np.array(timings))), total_steps,
        mse_error, rmse_error, relative_error
    ])
    finish_wandb(wandb_run, {
        "test_mse": float(mse_error),
        "test_rmse": float(rmse_error),
        "test_relative": float(relative_error),
        "avg_iteration_time": avg_time,
        "param_count": int(param_count),
    })

    plot_three_pdfs(path, interval)


if __name__ == "__main__":
    seed = args.seed
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    train(key)
