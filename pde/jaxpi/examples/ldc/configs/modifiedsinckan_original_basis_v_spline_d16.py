from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_spline_d16",
        v_basis="cubic_spline",
        v_degree=16,
    )
