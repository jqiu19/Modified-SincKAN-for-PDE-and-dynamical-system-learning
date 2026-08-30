from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_sinc_u12h2_v24h4",
        v_basis="sinc",
        u_degree=12,
        u_len_h=2,
        v_degree=24,
        v_len_h=4,
    )
