from configs.table_base import make_basis_config


def get_config():
    return make_basis_config("lshape_table_kan_seed42", "KAN", seed=42, degree=16)
