# Kuramoto-JAX Neural ODE Benchmark

This folder trains Neural ODE vector fields on a standard all-to-all Kuramoto oscillator network:

```text
d theta_i / dt = omega_i + K/N * sum_j sin(theta_j - theta_i), i=1,...,N
```

Default benchmark:

- `N = 50`
- `K = 4.0`
- `omega_i ~ N(0, 1)`, centered to zero mean
- reference trajectory: RK4 with `dt_ref = 0.005`
- saved trajectory spacing: `dt = 0.05`
- training interval: `t = 0..4`
- test/statistics interval: `t = 4..10`

Models:

- `mlp`
- `modifiedmlp`
- `kan`
- `sinckan`
- `modifiedsinckan`

Each run writes:

- `metrics.jsonl`: training/test/short-rollout/statistical metrics every `--log_every`
- `summary.json`: final and best metrics, parameter count, speed
- `loss.png`: loss/error curves
- `rollout_data.npz`: final truth/prediction/time data
- `best_rollout_data.npz`: best-test checkpoint rollout data
- `rollout_first_vars.png`: first variables truth vs prediction with train/test split

Run short selection after the current Lotka-Volterra jobs finish:

```bash
cd ~/code/python_code/modifiedsinckan/KAN-ODEs/Kuramoto-JAX
bash scripts/run_short_sweep.sh
```

The training script defaults to CPU if `JAX_PLATFORMS` is unset. To force GPU on
a machine with working JAX CUDA support:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=7 bash scripts/run_short_sweep.sh
```

After selecting best configs, edit `scripts/run_full_3seeds_template.sh` and run:

```bash
bash scripts/run_full_3seeds_template.sh
python summarize_kuramoto.py --results_dir results_table
```
