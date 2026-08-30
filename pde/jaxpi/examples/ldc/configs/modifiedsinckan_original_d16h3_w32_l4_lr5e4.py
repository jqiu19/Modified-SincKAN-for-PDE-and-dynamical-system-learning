from configs.modifiedsinckan_original_d16h3_base import get_config as get_base_config


def get_config():
    """Original d16h3 with width=32, depth=4, lr=5e-4."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_original_d16h3_w32_l4_lr5e4"
    config.arch.hidden_dim = 32
    config.arch.num_layers = 4
    config.optim.learning_rate = 5e-4
    config.optim.warmup_steps = 5000

    return config
