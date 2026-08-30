from configs.modifiedsinckan_original_basis_ablation_base import make_config


def get_config():
    return make_config(
        "modifiedsinckan_original_basis_v_linear_alpha02",
        v_basis="dense_linear",
        nonlinearity=0.2,
    )
