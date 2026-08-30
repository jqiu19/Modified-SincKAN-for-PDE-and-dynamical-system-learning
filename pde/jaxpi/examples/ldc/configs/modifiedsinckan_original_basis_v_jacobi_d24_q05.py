from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_jacobi_d24_q05",
        v_basis="fractional_jacobi",
        v_degree=24,
        jacobi_frac_power=0.5,
    )
