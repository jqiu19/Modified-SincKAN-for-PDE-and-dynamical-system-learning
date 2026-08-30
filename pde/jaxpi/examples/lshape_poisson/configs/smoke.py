from configs.table_base import make_mlp_config


def get_config():
    config = make_mlp_config("lshape_smoke", "Mlp", short=True, seed=42)
    config.training.max_steps = 3
    config.training.num_residual = 512
    config.training.num_boundary = 120
    config.training.batch_size = 128
    config.training.reference_grid = 41
    config.logging.log_every_steps = 1
    config.weighting.scheme = "none"
    return config
