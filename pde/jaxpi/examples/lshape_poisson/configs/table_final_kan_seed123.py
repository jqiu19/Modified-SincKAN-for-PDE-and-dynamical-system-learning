from configs.table_base import make_basis_config


def get_config():
    return make_basis_config("lshape_table_kan_seed123", "KAN", seed=123, degree=16)
