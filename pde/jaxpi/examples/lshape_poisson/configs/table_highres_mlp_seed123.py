from configs.table_base import make_mlp_config
from configs.table_highres_base import apply_highres


def get_config():
    return apply_highres(make_mlp_config("lshape_table_mlp_seed123", "Mlp", seed=123))

