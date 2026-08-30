from configs.table_base import make_sinckan_config


def get_config():
    return make_sinckan_config("table_sinckan_d16h3_w256_l4_short", short=True, hidden_dim=256, num_layers=4, degree=16, len_h=3)
