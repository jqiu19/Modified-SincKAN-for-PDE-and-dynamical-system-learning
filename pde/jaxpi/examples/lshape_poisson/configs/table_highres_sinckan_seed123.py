from configs.table_base import make_sinckan_config
from configs.table_highres_base import apply_highres


def get_config():
    return apply_highres(make_sinckan_config("lshape_table_sinckan_seed123", seed=123, degree=8, len_h=1))

