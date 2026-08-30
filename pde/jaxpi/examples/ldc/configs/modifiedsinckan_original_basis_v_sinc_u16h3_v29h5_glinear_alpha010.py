from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_sinc_u16h3_v29h5_glinear_alpha010",
        v_basis="sinc",
        u_degree=16,
        u_len_h=3,
        v_degree=29,
        v_len_h=5,
        g_basis="linear",
        nonlinearity=0.10,
    )
