from configs.table_base import make_basis_config
from configs.table_highres_base import apply_highres


def get_config():
    return apply_highres(make_basis_config("lshape_table_kan_seed2026", "KAN", seed=2026, degree=16))

