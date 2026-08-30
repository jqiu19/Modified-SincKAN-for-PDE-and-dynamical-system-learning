from configs.table_base import make_ours_config
from configs.table_highres_base import apply_highres


def get_config():
    return apply_highres(make_ours_config("lshape_table_ours_seed42", seed=42))

