from configs.sinc import get_config as _get_config


def get_config():
    config = _get_config()
    config.swanlab.name = "sinc_re3200"
    config.swanlab.mode = "disabled"
    config.training.Re = [3200]
    config.training.max_steps = [500000]
    config.training.batch_size = 1024
    return config
