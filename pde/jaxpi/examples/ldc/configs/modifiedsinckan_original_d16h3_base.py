from configs.modifiedsinckan_uv_same_d16h3_stablesinc import get_config as get_base_config


def get_config():
    """Original ModifiedSincKAN-style d16h3 without Fourier embedding."""
    config = get_base_config()

    config.arch.arch_name = "ModifiedSincKANOriginal"
    config.arch.fourier_emb = None
    config.arch.periodicity = None
    config.arch.hidden_dim = 256
    config.arch.num_layers = 4
    config.arch.initialization = "Xavier"
    config.arch.gate_mode = "separate_sigmoid"
    config.arch.sinc_mode = "stable"

    return config
