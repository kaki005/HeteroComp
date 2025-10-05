import logging
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import optax

T = TypeVar("T")


class Trainer(Generic[T]):
    def __init__(self, optimizer, model: T, is_update_param: T | None = None):
        self.optimizer = optimizer
        self.is_update_param: T | None = is_update_param
        self.model: T = model
        """current model params"""
        self.opt_state: optax.OptState = optimizer.init(model)
        self.logger = logging.getLogger()

    def update(self, grad: jnp.ndarray) -> T:
        updates, self.opt_state = self.optimizer.update(grad, self.opt_state, self.model)
        if self.is_update_param is None:
            self.model = eqx.apply_updates(self.model, updates)
        else:  # if fix some parameters
            diff_model, static_model = eqx.partition(self.model, self.is_update_param)  # split
            diff_updates, _ = eqx.partition(updates, self.is_update_param)
            diff_model = eqx.apply_updates(diff_model, diff_updates)
            self.model = eqx.combine(diff_model, static_model)  # combine
        return self.model
