from configs.pirate import get_config as _get_config


def get_config():
    config = _get_config()
    config.swanlab.name = "pirate_bs1024"
    config.training.batch_size = 1024
    return config
