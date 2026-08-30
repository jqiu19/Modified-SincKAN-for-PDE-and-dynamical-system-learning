from configs.table_base import make_mlp_config


def get_config():
    return make_mlp_config("table_modifiedmlp_w256_l4_short", "ModifiedMlp", short=True, hidden_dim=256, num_layers=4)
