from configs.modifiedsinckan_original_d16h3_base import get_config as get_base_config


def get_config():
    """Original d16h3 with wider hidden width 512 and baseline lr."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_original_d16h3_w512_l4_lr1e3"
    config.arch.hidden_dim = 512
    config.arch.num_layers = 4
    config.optim.learning_rate = 1e-3

    return config
