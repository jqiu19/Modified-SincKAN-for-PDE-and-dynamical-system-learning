from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_u12h2_v20h3_alpha006", u_degree=12, u_len_h=2, v_degree=20, v_len_h=3, alpha=0.06)
