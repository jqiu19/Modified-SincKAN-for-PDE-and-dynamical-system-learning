from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_u16h3_v24h4_alpha004", v_degree=24, v_len_h=4, alpha=0.04)
