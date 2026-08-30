from configs.tune_ours_base import make_config


def get_config():
    config = make_config("lshape_tune_ours_gate_residual_w50_l4")
    config.arch.gate_mode = "separate_residual"
    return config
