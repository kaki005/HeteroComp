# HeteroComp




## Setup
- `Jaxopt` package contains a bug.
  - Please replace line 248 in `.venv/lib/python3.12/site-packages/jaxopt/_src/tree_util.py`
  ``` python
    if isinstance(
        # p, (bool, int, float, complex, onp.ndarray, jnp.ndarray) OLD
        p, (bool, int, float, complex, onp.ndarray, jax.Array) # NEW
    )
  ```


## Demo
```
    uv sync
    uv run main.py --config-name=edge model.name=heterocomp
    uv run plot_anomaly.py --config-name=edge
    uv run calc_anomalyscore.py --config-name=edge model.name=heterocomp
```
