import os
import time

import jax
from jax.tree_util import tree_map

import ml_collections
import swanlab

from jaxpi.logging import Logger
from jaxpi.samplers import SpaceSampler
from jaxpi.utils import save_checkpoint

import models
from utils import finite_difference_reference, sample_lshape_interior


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    os.chdir(workdir)
    use_swanlab = config.swanlab.mode != "disabled"
    if use_swanlab:
        swanlab.init(
            project=config.swanlab.project,
            name=config.swanlab.name,
            mode=config.swanlab.mode,
            logdir=os.path.join(os.getcwd(), "swanlog"),
        )

    x_res = sample_lshape_interior(config.training.num_residual, seed=config.seed)
    x_ref, u_ref = finite_difference_reference(config.training.reference_grid)

    res_sampler = iter(SpaceSampler(x_res, config.training.batch_size))
    model = models.LShapePoisson2D(config)
    evaluator = models.LShapePoissonEvaluator(config, model)
    logger = Logger()

    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if jax.process_index() == 0 and step % config.logging.log_every_steps == 0:
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            host_batch = jax.device_get(tree_map(lambda x: x[0], batch))
            log_dict = evaluator(state, host_batch, x_ref, u_ref)
            if use_swanlab:
                swanlab.log(log_dict, step)
            logger.log_iter(step, start_time, time.time(), log_dict)

        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.swanlab.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model
