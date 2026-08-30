from configs.table_base import make_ours_config


def make_config(
    name,
    seed=42,
    max_steps=20000,
    u_degree=16,
    u_len_h=3,
    v_degree=28,
    v_len_h=5,
    g_basis="linear",
    g_degree=8,
    g_len_h=1,
    alpha=0.09,
    hidden_dim=50,
    num_layers=4,
    lr=1e-3,
):
    config = make_ours_config(name, seed=seed)
    config.training.max_steps = max_steps
    config.arch.hidden_dim = hidden_dim
    config.arch.num_layers = num_layers
    config.arch.nonlinearity = alpha
    config.arch.u_degree = u_degree
    config.arch.u_len_h = u_len_h
    config.arch.v_degree = v_degree
    config.arch.v_len_h = v_len_h
    config.arch.g_basis = g_basis
    config.arch.g_degree = g_degree
    config.arch.g_len_h = g_len_h
    config.optim.learning_rate = lr
    return config
