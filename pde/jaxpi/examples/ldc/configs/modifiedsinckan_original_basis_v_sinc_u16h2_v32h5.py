from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_sinc_u16h2_v32h5",
        v_basis="sinc",
        u_degree=16,
        u_len_h=2,
        v_degree=32,
        v_len_h=5,
    )
