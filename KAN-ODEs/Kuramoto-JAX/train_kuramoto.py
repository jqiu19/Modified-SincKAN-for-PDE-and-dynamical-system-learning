import argparse
import json
import math
import os
import time
from pathlib import Path

# Avoid failing on machines where a CUDA-enabled jaxlib is installed but no GPU
# is visible. Set JAX_PLATFORMS=cuda explicitly when running on a valid GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import numpy as np


def wrap_pi(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def kuramoto_rhs(theta, omega, coupling):
    diff = theta[None, :] - theta[:, None]
    return omega + (coupling / theta.shape[0]) * jnp.sum(jnp.sin(diff), axis=1)


def rk4_step(fn, x, dt):
    k1 = fn(x)
    k2 = fn(x + 0.5 * dt * k1)
    k3 = fn(x + 0.5 * dt * k2)
    k4 = fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(fn, x0, num_steps, dt):
    def body(x, _):
        x_next = rk4_step(fn, x, dt)
        return x_next, x_next

    _, xs = jax.lax.scan(body, x0, None, length=num_steps)
    return jnp.concatenate([x0[None, :], xs], axis=0)


def make_reference(n, coupling, omega_std, t_final, t_train, dt, dt_ref, seed):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    omega = omega_std * jax.random.normal(k1, (n,), dtype=jnp.float32)
    omega = omega - jnp.mean(omega)
    theta0 = jax.random.uniform(k2, (n,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)

    stride = int(round(dt / dt_ref))
    total_ref_steps = int(round(t_final / dt_ref))
    ref = rollout(lambda x: kuramoto_rhs(x, omega, coupling), theta0, total_ref_steps, dt_ref)
    ref = ref[::stride]
    t = jnp.arange(ref.shape[0], dtype=jnp.float32) * dt
    train_len = int(round(t_train / dt)) + 1
    return theta0, omega, t, ref, train_len


def glorot(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return std * jax.random.normal(key, shape, dtype=jnp.float32)


def init_dense(key, in_dim, out_dim, scale=1.0):
    return {
        "w": scale * glorot(key, (in_dim, out_dim)),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def dense(params, x):
    return x @ params["w"] + params["b"]


def init_mlp(key, in_dim, hidden_dim, out_dim, depth):
    keys = jax.random.split(key, depth + 1)
    layers = []
    d = in_dim
    for i in range(depth):
        layers.append(init_dense(keys[i], d, hidden_dim))
        d = hidden_dim
    layers.append(init_dense(keys[-1], d, out_dim, scale=0.1))
    return {"layers": layers}


def apply_mlp(params, x):
    h = x
    for layer in params["layers"][:-1]:
        h = jnp.tanh(dense(layer, h))
    return dense(params["layers"][-1], h)


def init_modified_mlp(key, in_dim, hidden_dim, out_dim, depth):
    keys = jax.random.split(key, depth + 4)
    return {
        "u": init_dense(keys[0], in_dim, hidden_dim),
        "v": init_dense(keys[1], in_dim, hidden_dim),
        "in": init_dense(keys[2], in_dim, hidden_dim),
        "layers": [init_dense(keys[i + 3], hidden_dim, hidden_dim) for i in range(depth)],
        "out": init_dense(keys[-1], hidden_dim, out_dim, scale=0.1),
    }


def apply_modified_mlp(params, x):
    u = jnp.tanh(dense(params["u"], x))
    v = jnp.tanh(dense(params["v"], x))
    z = dense(params["in"], x)
    for layer in params["layers"]:
        h = jax.nn.sigmoid(z)
        z = dense(layer, h * u + (1.0 - h) * v)
    return dense(params["out"], jnp.tanh(z))


def init_kan_layer(key, in_dim, out_dim, degree):
    k1, k2 = jax.random.split(key)
    centers = jnp.linspace(-1.0, 1.0, degree, dtype=jnp.float32)
    h = jnp.asarray(2.0 / max(degree - 1, 1), dtype=jnp.float32)
    coeff = 0.05 * jax.random.normal(k1, (in_dim, out_dim, degree), dtype=jnp.float32)
    skip = init_dense(k2, in_dim, out_dim, scale=0.1)
    return {"coeff": coeff, "skip": skip, "centers": centers, "h": h}


def apply_kan_layer(params, x):
    xb = jnp.tanh(x / jnp.pi)
    basis = jnp.exp(-((xb[:, None] - params["centers"][None, :]) / params["h"]) ** 2)
    return jnp.einsum("id,iod->o", basis, params["coeff"]) + dense(params["skip"], x)


def init_kan(key, in_dim, hidden_dim, out_dim, depth, degree):
    keys = jax.random.split(key, depth + 1)
    layers = []
    d = in_dim
    for i in range(depth):
        layers.append(init_kan_layer(keys[i], d, hidden_dim, degree))
        d = hidden_dim
    layers.append(init_kan_layer(keys[-1], d, out_dim, degree))
    return {"layers": layers}


def apply_kan(params, x):
    h = x
    for layer in params["layers"][:-1]:
        h = jnp.tanh(apply_kan_layer(layer, h))
    return apply_kan_layer(params["layers"][-1], h)


def stable_sinc(x):
    x2 = x * x
    taylor = 1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0
    safe = jnp.where(jnp.abs(x) < 1.0e-3, 1.0, x)
    value = jnp.sin(safe) / safe
    return jnp.where(jnp.abs(x) < 1.0e-3, taylor, value)


def init_sinc_layer(key, in_dim, out_dim, degree, len_h, init_h=2.0):
    k1, k2 = jax.random.split(key)
    ks = jnp.arange(-(degree // 2), degree // 2 + 1, dtype=jnp.float32)
    basis_degree = ks.shape[0]
    hs = 1.0 / (init_h * (1.0 + jnp.arange(len_h, dtype=jnp.float32)))
    coeff = jax.random.normal(k1, (in_dim, out_dim, len_h, basis_degree), dtype=jnp.float32)
    coeff = 0.1 * coeff / jnp.sqrt(jnp.asarray(in_dim * basis_degree, dtype=jnp.float32))
    skip = init_dense(k2, in_dim, out_dim, scale=0.1)
    return {"coeff": coeff, "skip": skip, "ks": ks, "hs": hs}


def apply_sinc_layer(params, x):
    xb = jnp.tanh(x / jnp.pi)
    xx = xb[:, None, None] / params["hs"][None, :, None] - params["ks"][None, None, :]
    basis = stable_sinc(xx)
    return jnp.einsum("ihd,iohd->o", basis, params["coeff"]) + dense(params["skip"], x)


def init_sinckan(key, in_dim, hidden_dim, out_dim, depth, degree, len_h):
    keys = jax.random.split(key, depth + 1)
    layers = []
    d = in_dim
    for i in range(depth):
        layers.append(init_sinc_layer(keys[i], d, hidden_dim, degree, len_h))
        d = hidden_dim
    layers.append(init_sinc_layer(keys[-1], d, out_dim, degree, len_h))
    return {"layers": layers}


def apply_sinckan(params, x):
    h = x
    for layer in params["layers"][:-1]:
        h = jnp.tanh(apply_sinc_layer(layer, h))
    return apply_sinc_layer(params["layers"][-1], h)


def init_modified_sinc_kan(key, in_dim, hidden_dim, out_dim, depth, u_degree, u_len_h, v_degree, v_len_h, alpha):
    keys = jax.random.split(key, depth + 6)
    return {
        "u": init_sinc_layer(keys[0], in_dim, hidden_dim, u_degree, u_len_h),
        "v": init_sinc_layer(keys[1], in_dim, hidden_dim, v_degree, v_len_h),
        "in": init_dense(keys[2], in_dim, hidden_dim),
        "layers": [init_dense(keys[i + 3], hidden_dim, hidden_dim) for i in range(depth)],
        "gate": init_dense(keys[-3], hidden_dim, hidden_dim),
        "g": init_dense(keys[-2], in_dim, hidden_dim),
        "out": init_dense(keys[-1], hidden_dim, out_dim, scale=0.1),
        "alpha": jnp.asarray(alpha, dtype=jnp.float32),
    }


def apply_modified_sinc_kan(params, x):
    u = jnp.tanh(apply_sinc_layer(params["u"], x))
    v = jnp.tanh(apply_sinc_layer(params["v"], x))
    g = dense(params["g"], x)
    z = dense(params["in"], x)
    for layer in params["layers"]:
        h = jnp.tanh(z)
        gate = jax.nn.sigmoid(dense(params["gate"], h) + g)
        candidate = gate * u + (1.0 - gate) * v
        z = dense(layer, (1.0 - params["alpha"]) * h + params["alpha"] * candidate)
    return dense(params["out"], z)


def init_model(args, key):
    if args.model == "mlp":
        return init_mlp(key, args.n, args.width, args.n, args.depth), apply_mlp
    if args.model == "modifiedmlp":
        return init_modified_mlp(key, args.n, args.width, args.n, args.depth), apply_modified_mlp
    if args.model == "kan":
        return init_kan(key, args.n, args.width, args.n, args.depth, args.degree), apply_kan
    if args.model == "sinckan":
        return init_sinckan(key, args.n, args.width, args.n, args.depth, args.degree, args.len_h), apply_sinckan
    if args.model == "modifiedsinckan":
        return (
            init_modified_sinc_kan(
                key, args.n, args.width, args.n, args.depth, args.u_degree, args.u_len_h,
                args.v_degree, args.v_len_h, args.alpha,
            ),
            apply_modified_sinc_kan,
        )
    raise ValueError(f"unknown model: {args.model}")


def count_params(params):
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves if hasattr(x, "size")))


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def adam_update(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2 ** t), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


def cosine_lr(step, total_steps, lr_max, lr_min):
    progress = jnp.minimum(jnp.asarray(step, dtype=jnp.float32) / float(total_steps), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * progress))


def make_loss_fn(apply_fn, ref, train_len, dt):
    train_ref = ref[:train_len]

    def loss_fn(params):
        pred = rollout(lambda x: apply_fn(params, x), train_ref[0], train_len - 1, dt)
        return jnp.mean((pred - train_ref) ** 2)

    return loss_fn


def compute_metrics(params, apply_fn, ref, train_len, dt, horizons):
    full_pred = rollout(lambda x: apply_fn(params, x), ref[0], ref.shape[0] - 1, dt)
    train_mse = jnp.mean((full_pred[:train_len] - ref[:train_len]) ** 2)
    test_mse = jnp.mean((full_pred[train_len - 1:] - ref[train_len - 1:]) ** 2)

    start = train_len - 1
    metrics = {"train_mse": train_mse, "test_mse": test_mse}
    for h in horizons:
        steps = int(round(h / dt))
        short_pred = rollout(lambda x: apply_fn(params, x), ref[start], steps, dt)
        metrics[f"short_mse_{str(h).replace('.', 'p')}"] = jnp.mean(
            (short_pred - ref[start:start + steps + 1]) ** 2
        )

    pred_seg = full_pred[start:]
    true_seg = ref[start:]
    metrics["stat_mean_err_4_10"] = jnp.abs(jnp.mean(pred_seg) - jnp.mean(true_seg)) / (jnp.abs(jnp.mean(true_seg)) + 1e-8)
    metrics["stat_std_err_4_10"] = jnp.abs(jnp.std(pred_seg) - jnp.std(true_seg)) / (jnp.std(true_seg) + 1e-8)
    return metrics, full_pred


def save_plots(out_dir, metrics_history, t, ref, pred, train_len, max_vars):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(str(exc))
        return

    epochs = [m["epoch"] for m in metrics_history]
    plt.figure(figsize=(7, 4))
    for key in ("train_mse", "test_mse", "short_mse_0p5", "short_mse_1p0", "short_mse_2p0"):
        vals = [m[key] for m in metrics_history if key in m]
        if len(vals) == len(epochs):
            plt.semilogy(epochs, vals, label=key)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=200)
    plt.close()

    n_plot = min(max_vars, ref.shape[1])
    fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.0 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, ref[:, i], "k-", lw=1.5, label="truth")
        ax.plot(t, pred[:, i], "C1--", lw=1.2, label="prediction")
        ax.axvline(t[train_len - 1], color="C3", ls=":", lw=1.2, label="train/test split")
        ax.scatter(t[:train_len], ref[:train_len, i], s=8, color="C0", alpha=0.5, label="train points" if i == 0 else None)
        ax.scatter(t[train_len - 1:], ref[train_len - 1:, i], s=8, color="C2", alpha=0.5, label="test points" if i == 0 else None)
        ax.set_ylabel(f"theta_{i}")
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(out_dir / "rollout_first_vars.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "modifiedmlp", "kan", "sinckan", "modifiedsinckan"], required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--coupling", type=float, default=4.0)
    parser.add_argument("--omega_std", type=float, default=1.0)
    parser.add_argument("--t_final", type=float, default=10.0)
    parser.add_argument("--t_train", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--dt_ref", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--lr_max", type=float, default=5e-4)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--len_h", type=int, default=1)
    parser.add_argument("--u_degree", type=int, default=16)
    parser.add_argument("--u_len_h", type=int, default=1)
    parser.add_argument("--v_degree", type=int, default=32)
    parser.add_argument("--v_len_h", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.09)
    parser.add_argument("--max_plot_vars", type=int, default=6)
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or (
        f"{args.model}_n{args.n}_d{args.depth}_w{args.width}_seed{args.seed}"
    )
    out_dir = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    theta0, omega, t, ref, train_len = make_reference(
        args.n, args.coupling, args.omega_std, args.t_final, args.t_train, args.dt, args.dt_ref, args.seed
    )
    model_key = jax.random.PRNGKey(args.seed + 10_000)
    params, apply_fn = init_model(args, model_key)
    param_count = count_params(params)

    loss_fn = make_loss_fn(apply_fn, ref, train_len, args.dt)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def train_step(params, opt_state, step):
        loss, grads = value_and_grad(params)
        lr = cosine_lr(step, args.epochs, args.lr_max, args.lr_min)
        params, opt_state = adam_update(params, grads, opt_state, lr)
        return params, opt_state, loss, lr

    metric_fn = jax.jit(lambda p: compute_metrics(p, apply_fn, ref, train_len, args.dt, (0.5, 1.0, 2.0)))
    opt_state = {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": 0}

    config = vars(args) | {"run_name": run_name, "parameter_size": param_count}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"run_name": run_name, "parameter_size": param_count, "config": config}))

    best = {"test_mse": float("inf"), "epoch": 0}
    history = []
    start_time = time.time()
    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w") as log_f:
        for epoch in range(1, args.epochs + 1):
            params, opt_state, loss, lr = train_step(params, opt_state, epoch)
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                metrics, pred = metric_fn(params)
                row = {
                    "epoch": epoch,
                    "lr": float(lr),
                    "loss_step": float(loss),
                    **{k: float(v) for k, v in metrics.items()},
                }
                if row["test_mse"] < best["test_mse"]:
                    best = row | {"best_epoch": epoch}
                    np.savez(
                        out_dir / "best_rollout_data.npz",
                        t=np.asarray(t),
                        truth=np.asarray(ref),
                        pred=np.asarray(pred),
                        omega=np.asarray(omega),
                        theta0=np.asarray(theta0),
                        train_len=np.asarray(train_len),
                    )
                history.append(row)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                print(json.dumps(row), flush=True)

    elapsed = time.time() - start_time
    metrics, pred = metric_fn(params)
    final_row = {k: float(v) for k, v in metrics.items()}
    summary = {
        "run_name": run_name,
        "model": args.model,
        "seed": args.seed,
        "parameter_size": param_count,
        "elapsed_sec": elapsed,
        "iter_per_sec": args.epochs / elapsed,
        "best": best,
        "final": final_row,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out_dir / "rollout_data.npz",
        t=np.asarray(t),
        truth=np.asarray(ref),
        pred=np.asarray(pred),
        omega=np.asarray(omega),
        theta0=np.asarray(theta0),
        train_len=np.asarray(train_len),
    )
    save_plots(out_dir, history, np.asarray(t), np.asarray(ref), np.asarray(pred), train_len, args.max_plot_vars)
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
