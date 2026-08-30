def apply_highres(config):
    config.training.num_residual = 100000
    config.training.num_boundary = 10000
    config.training.batch_size = 4096
    config.training.reference_grid = 644
    config.training.max_steps = 100000
    config.swanlab.name = f"{config.swanlab.name}_highres"
    return config

