from configs.table_base import make_mlp_config


def get_config():
    return make_mlp_config("lshape_table_mlp_seed123", "Mlp", seed=123)
