from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_linear_u20h3_lr5e4",
        v_basis="dense_linear",
        u_degree=20,
        u_len_h=3,
        lr=5e-4,
    )
