from configs.modifiedsinckan import get_config as get_base_config


def get_config():
    """ModifiedSincKAN with the same fixed sinc basis for u/v/g."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_fixed_d32_h5"
    config.swanlab.project = "PINN-LDC-modifiedsinckan-ablation"
    config.swanlab.mode = "disabled"

    config.arch.degree = 32
    config.arch.u_degree = 32
    config.arch.v_degree = 32
    config.arch.g_degree = 32
    config.arch.len_h = 5
    config.arch.u_len_h = 5
    config.arch.v_len_h = 5
    config.arch.g_len_h = 5

    # Short screening run: 40% of the original curriculum budget.
    config.training.max_steps = [4000, 8000, 20000, 20000, 200000]

    return config
