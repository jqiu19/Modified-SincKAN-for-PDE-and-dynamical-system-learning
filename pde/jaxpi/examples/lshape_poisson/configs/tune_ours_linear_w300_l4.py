from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_linear_w300_l4", hidden_dim=300)
