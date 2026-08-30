from configs.table_base import make_sinckan_config


def get_config():
    return make_sinckan_config("table_final_sinckan_seed123", short=False, seed=123, hidden_dim=256, num_layers=4, degree=16, len_h=3)
