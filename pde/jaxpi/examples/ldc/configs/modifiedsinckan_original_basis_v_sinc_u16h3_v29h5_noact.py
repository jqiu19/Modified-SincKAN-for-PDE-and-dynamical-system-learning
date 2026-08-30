from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    config = make_config(
        "modifiedsinckan_original_basis_v_sinc_u16h3_v29h5_noact",
        v_basis="sinc",
        u_degree=16,
        u_len_h=3,
        v_degree=29,
        v_len_h=5,
    )
    config.arch.sinc_apply_activation = False
    return config
