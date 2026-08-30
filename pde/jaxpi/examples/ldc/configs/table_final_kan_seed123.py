from configs.table_base import make_basis_config


def get_config():
    return make_basis_config("table_final_kan_seed123", "KAN", short=False, seed=123, hidden_dim=128, num_layers=4, degree=16)
