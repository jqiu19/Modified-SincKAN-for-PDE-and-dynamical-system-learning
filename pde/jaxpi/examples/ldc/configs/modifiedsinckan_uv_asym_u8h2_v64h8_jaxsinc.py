from configs.modifiedsinckan_uv_asym_u8h2_v64h8_stablesinc import get_config as get_base_config


def get_config():
    """Asymmetric u/v basis ablation using jnp.sinc(xx / pi)."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_uv_asym_u8h2_v64h8_jaxsinc"
    config.arch.sinc_mode = "jax"

    return config
