import ml_collections


def _common(name, project="PINN-LDC-table", short=True, seed=42):
    config = ml_collections.ConfigDict()
    config.mode = "train"
    config.use_pi_init = False

    config.swanlab = swanlab = ml_collections.ConfigDict()
    swanlab.project = project
    swanlab.name = name
    swanlab.tag = None
    swanlab.mode = "disabled"

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 10000
    optim.staircase = False
    optim.warmup_steps = 5000
    optim.grad_accum_steps = 0
    optim.schedule_free = False

    config.training = training = ml_collections.ConfigDict()
    training.Re = [100, 400, 1000, 1600, 3200]
    training.max_steps = [4000, 8000, 20000, 20000, 200000] if short else [10000, 20000, 50000, 50000, 500000]
    training.batch_size = 4096

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"u_bc": 100.0, "v_bc": 100.0, "ru": 1.0, "rv": 1.0, "rc": 10.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000 if not short else None
    saving.num_keep_ckpts = 20

    config.input_dim = 2
    config.seed = seed
    return config


def make_mlp_config(name, arch_name, short=True, seed=42, hidden_dim=256, num_layers=4, lr=1e-3):
    config = _common(name, short=short, seed=seed)
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = arch_name
    arch.num_layers = num_layers
    arch.hidden_dim = hidden_dim
    arch.out_dim = 3
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    arch.pi_init = None
    config.optim.learning_rate = lr
    return config


def make_basis_config(name, arch_name, short=True, seed=42, hidden_dim=256, num_layers=4, degree=16, lr=1e-3):
    config = _common(name, short=short, seed=seed)
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = arch_name
    arch.num_layers = num_layers
    arch.hidden_dim = hidden_dim
    arch.out_dim = 3
    arch.degree = degree
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    config.optim.learning_rate = lr
    return config


def make_sinckan_config(name, short=True, seed=42, hidden_dim=256, num_layers=4, degree=8, len_h=1, lr=1e-3):
    config = _common(name, short=short, seed=seed)
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "SincKAN"
    arch.num_layers = num_layers
    arch.hidden_dim = hidden_dim
    arch.out_dim = 3
    arch.degree = degree
    arch.init_h = 2.0
    arch.len_h = len_h
    arch.decay = "inverse"
    arch.skip = True
    arch.initialization = "zeros"
    arch.skip_mode = 1
    arch.sinc_mode = "stable"
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    arch.nonlinearity = 0.0
    arch.pi_init = None
    config.optim.learning_rate = lr
    return config


def make_ours_config(name, short=True, seed=42):
    config = _common(name, short=short, seed=seed)
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedSincKANOriginal"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "tanh"
    arch.nonlinearity = 0.09
    arch.gate_mode = "separate_sigmoid"
    arch.degree = 16
    arch.init_h = 2.0
    arch.len_h = 3
    arch.u_degree = 16
    arch.u_len_h = 3
    arch.v_degree = 28
    arch.v_len_h = 5
    arch.g_degree = 8
    arch.g_len_h = 1
    arch.u_basis = "sinc"
    arch.v_basis = "sinc"
    arch.g_basis = "linear"
    arch.jacobi_alpha = 0.0
    arch.jacobi_beta = 0.0
    arch.jacobi_frac_power = 0.75
    arch.decay = "inverse"
    arch.skip = True
    arch.initialization = "Xavier"
    arch.skip_mode = 1
    arch.sinc_mode = "stable"
    arch.sinc_apply_activation = True
    arch.sinc_activation_scale = 1.0
    arch.periodicity = None
    arch.fourier_emb = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    arch.pi_init = None
    config.optim.learning_rate = 5e-4
    return config
