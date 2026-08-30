from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_cheby_d24",
        v_basis="chebyshev",
        v_degree=24,
    )
