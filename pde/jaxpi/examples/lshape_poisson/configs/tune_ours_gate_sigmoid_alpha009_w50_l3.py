from configs.tune_ours_base import make_config


def get_config():
    return make_config("lshape_tune_ours_gate_sigmoid_alpha009_w50_l3", num_layers=3)
