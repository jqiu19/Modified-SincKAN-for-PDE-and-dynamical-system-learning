from configs.modifiedsinckan_original_d16h3_base import get_config as get_base_config


def make_config(name, v_basis, u_degree=16, u_len_h=3, v_degree=16, v_len_h=3,
                g_degree=8, g_len_h=1, lr=5e-4, jacobi_frac_power=0.75,
                g_basis="sinc", nonlinearity=0.1, gate_mode="separate_sigmoid"):
    """w256/l4 ModifiedSincKANOriginal basis ablation with sinc u and configurable v."""
    config = get_base_config()

    config.swanlab.name = name
    config.swanlab.project = "PINN-LDC-modifiedsinckan-basis-ablation"
    config.swanlab.mode = "disabled"

    config.arch.hidden_dim = 256
    config.arch.num_layers = 4
    config.arch.u_basis = "sinc"
    config.arch.v_basis = v_basis
    config.arch.g_basis = g_basis
    config.arch.u_degree = u_degree
    config.arch.u_len_h = u_len_h
    config.arch.v_degree = v_degree
    config.arch.v_len_h = v_len_h
    config.arch.g_degree = g_degree
    config.arch.g_len_h = g_len_h
    config.arch.jacobi_alpha = 0.0
    config.arch.jacobi_beta = 0.0
    config.arch.jacobi_frac_power = jacobi_frac_power
    config.arch.sinc_mode = "stable"
    config.arch.gate_mode = gate_mode
    config.arch.nonlinearity = nonlinearity

    config.optim.learning_rate = lr
    config.optim.warmup_steps = 5000
    config.training.max_steps = [4000, 8000, 20000, 20000, 200000]

    return config
