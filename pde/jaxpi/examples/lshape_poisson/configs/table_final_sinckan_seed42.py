from configs.table_base import make_sinckan_config


def get_config():
    return make_sinckan_config("lshape_table_sinckan_seed42", seed=42, degree=8, len_h=1)
