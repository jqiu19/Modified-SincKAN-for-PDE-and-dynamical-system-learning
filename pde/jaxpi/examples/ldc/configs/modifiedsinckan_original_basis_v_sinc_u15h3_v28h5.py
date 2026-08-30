from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_sinc_u15h3_v28h5",
        v_basis="sinc",
        u_degree=15,
        u_len_h=3,
        v_degree=28,
        v_len_h=5,
    )
