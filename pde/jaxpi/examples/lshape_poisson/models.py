from functools import partial

import jax.numpy as jnp
from jax import grad, hessian, jit, lax, pmap, vmap
from jax.tree_util import tree_map

from jaxpi.evaluator import BaseEvaluator
from jaxpi.models import ForwardBVP
from jaxpi.utils import ntk_fn

from utils import sample_lshape_boundary


class LShapePoisson2D(ForwardBVP):
    def __init__(self, config):
        super().__init__(config)
        self.x_bc = sample_lshape_boundary(config.training.num_boundary)
        self.u_bc = jnp.zeros((self.x_bc.shape[0], 1))
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))

    def neural_net(self, params, x, y):
        # The PDE is posed on [-1, 1]^2 \ [0, 1]^2, while the sinc-based
        # architectures were tuned for [0, 1]^2 coordinates.
        z = (jnp.stack([x, y]) + 1.0) / 2.0
        _, out = self.state.apply_fn(params, z)
        return out[0]

    def u_net(self, params, x, y):
        return self.neural_net(params, x, y)

    def r_net(self, params, x, y):
        hess = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        u_xx = hess[0][0]
        u_yy = hess[1][1]
        return -u_xx - u_yy - 1.0

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        u_bc_pred = self.u_pred_fn(params, self.x_bc[:, 0], self.x_bc[:, 1])[:, None]
        bc_loss = jnp.mean((u_bc_pred - self.u_bc) ** 2)
        r_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
        r_loss = jnp.mean(r_pred**2)
        return {"bc": bc_loss, "res": r_loss}

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        bc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.x_bc[:, 0], self.x_bc[:, 1]
        )
        res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.r_net, params, batch[:, 0], batch[:, 1]
        )
        return {"bc": bc_ntk, "res": res_ntk}

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch):
        weights = self.compute_weights(state.params, batch)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch):
        grads = grad(self.loss)(state.params, state.weights, batch)
        grads = lax.pmean(grads, "batch")
        return state.apply_gradients(grads=grads)

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_ref, u_ref):
        u_pred = self.u_pred_fn(params, x_ref[:, 0], x_ref[:, 1])[:, None]
        return jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)


class LShapePoissonEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, x_ref, u_ref):
        self.log_dict["l2_error"] = self.model.compute_l2_error(params, x_ref, u_ref)

    def __call__(self, state, batch, x_ref, u_ref):
        self.log_dict = super().__call__(state, batch)
        if self.config.logging.log_errors:
            self.log_errors(state.params, x_ref, u_ref)
        return self.log_dict
