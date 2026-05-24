"""Kernel facades.

Only `NUTS` is shipped initially. All other NumPyro kernels (`HMC`,
`SA`, `BarkerMH`, `HMCGibbs`, `DiscreteHMCGibbs`, `MixedHMC`) and the
BlackJAX-backed kernels (`SGLD`, `MALA`, ...) are deferred —
`from effectful_mcmc import HMC` raises `ImportError` from the package
`__init__.__getattr__` with a message that names the kernel.

The facade is intentionally thin: it holds the user's effectful model
and the kernel-specific kwargs, deferring construction of the actual
`numpyro.infer.NUTS` until `MCMC.run` provides the model args needed
for compilation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpyro.infer

from effectful_mcmc.compile import compile_to_numpyro


class NUTS:
    """Facade over `numpyro.infer.NUTS`.

    `**kernel_kwargs` are forwarded verbatim to `numpyro.infer.NUTS`
    when the kernel is constructed inside `MCMC.run(...)`. The kernel
    can't be built earlier because compilation needs runtime model args.
    """

    def __init__(self, model: Callable[..., Any], **kernel_kwargs: Any):
        self._model = model
        self._kwargs = kernel_kwargs

    def _compile(self, *args: Any, **kwargs: Any):
        """Run the compile pass and return the constructed NumPyro kernel
        plus the recorded site list and model return value.

        Called from `MCMC.run`.
        """
        compiled_model, sites, model_return_value = compile_to_numpyro(
            self._model, *args, **kwargs,
        )
        return (
            numpyro.infer.NUTS(compiled_model, **self._kwargs),
            sites,
            model_return_value,
        )
