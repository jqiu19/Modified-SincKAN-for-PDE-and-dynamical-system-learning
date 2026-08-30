from configs.tune_ours_base import make_config


def get_config():
    config = make_config("lshape_tune_ours_gate_residual_w64_l4", hidden_dim=64)
    config.arch.gate_mode = "separate_residual"
    return config
