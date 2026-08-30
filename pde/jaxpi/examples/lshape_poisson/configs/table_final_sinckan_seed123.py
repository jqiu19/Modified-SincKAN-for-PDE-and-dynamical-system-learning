from configs.table_base import make_sinckan_config


def get_config():
    return make_sinckan_config("lshape_table_sinckan_seed123", seed=123, degree=8, len_h=1)
