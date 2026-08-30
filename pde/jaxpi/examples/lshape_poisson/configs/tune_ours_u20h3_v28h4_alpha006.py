from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_u20h3_v28h4_alpha006", u_degree=20, u_len_h=3, v_degree=28, v_len_h=4, alpha=0.06)
