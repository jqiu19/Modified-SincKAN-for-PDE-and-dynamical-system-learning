from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_u16h3_v32h5_alpha006", v_degree=32, v_len_h=5, alpha=0.06)
