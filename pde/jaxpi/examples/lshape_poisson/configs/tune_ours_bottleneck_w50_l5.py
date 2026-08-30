from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_bottleneck_w50_l5", hidden_dim=50, num_layers=5)
