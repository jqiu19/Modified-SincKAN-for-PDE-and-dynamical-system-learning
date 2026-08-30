from configs.modifiedsinckan_uv_same_d32h5_stablesinc import get_config as get_base_config


def get_config():
    """Same u/v basis ablation using jnp.sinc(xx / pi) for the sinc basis."""
    config = get_base_config()

    config.swanlab.name = "modifiedsinckan_uv_same_d32h5_jaxsinc"
    config.arch.sinc_mode = "jax"

    return config
