from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_linear_u12h2_lr5e4",
        v_basis="dense_linear",
        u_degree=12,
        u_len_h=2,
        lr=5e-4,
    )
