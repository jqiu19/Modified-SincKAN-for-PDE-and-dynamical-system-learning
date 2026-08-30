from configs.table_base import make_basis_config


def get_config():
    return make_basis_config("table_kan_d16_w128_l4_short", "KAN", short=True, hidden_dim=128, num_layers=4, degree=16)
