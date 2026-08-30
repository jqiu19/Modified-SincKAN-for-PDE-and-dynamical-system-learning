from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_linear_u16h3_lr1e3",
        v_basis="dense_linear",
        u_degree=16,
        u_len_h=3,
        lr=1e-3,
    )
