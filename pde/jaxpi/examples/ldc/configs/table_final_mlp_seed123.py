from configs.table_base import make_mlp_config


def get_config():
    return make_mlp_config("table_final_mlp_seed123", "Mlp", short=False, seed=123)
