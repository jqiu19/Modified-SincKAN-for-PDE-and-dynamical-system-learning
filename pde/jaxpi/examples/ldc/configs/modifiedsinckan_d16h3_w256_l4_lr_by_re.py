from configs.modifiedsinckan_uv_same_d16h3_stablesinc import get_config as get_base_config


def get_config():
    """d16h3 baseline width/depth with Re-adaptive learning rate."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_d16h3_w256_l4_lr_by_re"
    config.arch.hidden_dim = 256
    config.arch.fourier_emb.embed_dim = 256
    config.arch.num_layers = 4
    config.training.lr_by_Re = [1e-3, 1e-3, 5e-4, 5e-4, 3e-4]

    return config
