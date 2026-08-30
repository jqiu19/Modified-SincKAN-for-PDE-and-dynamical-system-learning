from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_jacobi_d16_q075",
        v_basis="fractional_jacobi",
        v_degree=16,
        jacobi_frac_power=0.75,
    )
