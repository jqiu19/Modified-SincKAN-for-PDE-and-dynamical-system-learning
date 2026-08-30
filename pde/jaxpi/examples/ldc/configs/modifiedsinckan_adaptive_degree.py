import ml_collections

from configs.modifiedsinckan import get_config as get_base_config


def get_config():
    """ModifiedSincKAN curriculum with adaptive degree and fixed len_h."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_adaptive_degree"
    config.swanlab.project = "PINN-LDC-modifiedsinckan-ablation"
    config.swanlab.mode = "disabled"

    config.training.adaptive_basis = ml_collections.ConfigDict()
    config.training.adaptive_basis.degree = [8, 16, 32, 64, 128]
    config.training.adaptive_basis.len_h = [1, 1, 1, 1, 1]
    config.training.max_steps = [4000, 8000, 20000, 20000, 200000]

    return config
