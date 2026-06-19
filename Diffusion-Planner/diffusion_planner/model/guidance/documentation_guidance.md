# Classifer Guidance Tutorial

## Create your own guidance function

1. Create ``diffusion_planner/model/guidance/<my_guidance>.py``

```python
def my_guidance_fn(x, t, cond, inputs) -> torch.Tensor:
    ...

    return reward
```

2. Add ``<my_guidance_fn>`` in ``diffusion_planner/model/guidance/guidance_wrapper.py``

```python
# diffusion_planner/model/guidance/guidance_wrapper.py

...

class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [
            <my_guidance_1>,
            <my_guidance_2>,
            ...
            <my_guidance_N>
        ]

    def __call__(...):
        ...

...
```

3. Run ``sim_guidance_demo.sh``
4. Enjoy.