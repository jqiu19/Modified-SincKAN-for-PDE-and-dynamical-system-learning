from configs.table_base import make_mlp_config


def get_config():
    return make_mlp_config("table_final_modifiedmlp_seed123", "ModifiedMlp", short=False, seed=123)
