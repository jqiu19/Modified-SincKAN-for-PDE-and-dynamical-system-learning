from configs.modifiedsinckan_uv_same_d16h3_stablesinc import get_config as get_base_config


def get_config():
    """ModifiedSincKAN stable sinc d16/h3 with separate_residual gate."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_uv_same_d16h3_residual_stablesinc"
    config.arch.gate_mode = "separate_residual"

    return config
