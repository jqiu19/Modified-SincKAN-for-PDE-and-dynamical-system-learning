from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_gsinc_w100_l5", g_basis="sinc", g_degree=8, g_len_h=1, hidden_dim=100, num_layers=5)
