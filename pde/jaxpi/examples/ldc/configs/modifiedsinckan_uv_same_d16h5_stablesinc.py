from configs.modifiedsinckan import get_config as get_base_config


def get_config():
    """ModifiedSincKAN stable sinc with u/v using degree=16, len_h=5."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_uv_same_d16h5_stablesinc"
    config.swanlab.project = "PINN-LDC-modifiedsinckan-ablation"
    config.swanlab.mode = "disabled"

    config.arch.sinc_mode = "stable"
    config.arch.gate_mode = "separate_sigmoid"
    config.arch.degree = 8
    config.arch.len_h = 1
    config.arch.u_degree = 16
    config.arch.u_len_h = 5
    config.arch.v_degree = 16
    config.arch.v_len_h = 5
    config.arch.g_degree = 8
    config.arch.g_len_h = 1

    config.training.max_steps = [4000, 8000, 20000, 20000, 200000]

    return config
