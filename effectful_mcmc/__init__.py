"""Bridge between effectful and numpyro.infer.

Public API:
    sample, factor, param, deterministic   - defops + _NumPyroBackend
    NUTS, MCMC                             - kernel facade + driver
    Intervene                              - intervention helper
"""

from effectful.ops.semantics import fwd as _fwd
from effectful.ops.syntax import ObjectInterpretation, implements

from effectful_mcmc.primitives import (
    sample,
    factor,
    param,
    deterministic,
    _NumPyroBackend,
)
from effectful_mcmc.kernels import NUTS
from effectful_mcmc.mcmc import MCMC


class Intervene(ObjectInterpretation):
    """Effectful handler that overrides a named `sample` site.

    Intervened sites are *removed* from the posterior — the user already
    has the value (they passed it in). What changes is the posterior of
    the remaining (non-intervened) sites, which now conditions on the
    intervention as if `value` were an observation.

    Usage:
        with handler(Intervene("sigma", jnp.array(1.0))):
            model(data)

    Equivalent to numpyro.handlers.substitute({"sigma": value}) under
    a pure NumPyro program, but composable with any other effectful
    handler via `coproduct(...)`. Users who need richer intervention
    semantics (conditional, scheduled, etc.) subclass directly.
    """

    def __init__(self, target_name: str, value):
        super().__init__()
        self.target_name = target_name
        self.value = value

    @implements(sample)
    def sample(self, d, obs=None, *, name=None, infer=None):
        if name == self.target_name and obs is None:
            return self.value
        return _fwd()


__all__ = [
    "sample",
    "factor",
    "param",
    "deterministic",
    "Intervene",
    "NUTS",
    "MCMC",
    "_NumPyroBackend",
]


# Deferred kernels — split by backend so the deferral message is
# accurate about what would have to land to promote each kernel.
#
# Per PEP 562, module-level __getattr__ should raise `AttributeError`;
# Python's `from X import Y` machinery converts AttributeError into
# ImportError automatically. Raising ImportError directly works but
# surprises tools that catch AttributeError (e.g. hasattr).

# MCMC kernels that exist in `numpyro.infer.*` but aren't exposed yet.
# `Predictive`, `SVI`, etc. are not MCMC and aren't on this list.
# Promoting any is a one-facade addition in `kernels.py`.
_DEFERRED_NUMPYRO_KERNELS = (
    "HMC", "SA", "BarkerMH",
    "HMCGibbs", "DiscreteHMCGibbs", "MixedHMC",
    "AIES", "ESS",          # affine-invariant ensemble / ensemble slice
    "HMCECS",               # HMC with energy-conserving subsampling
)
# Kernels that live in `blackjax.*`, not numpyro.infer. Promoting any
# requires a parallel BlackJAX backend (a meaningful surface-area
# expansion) or treating each one as a one-off port. `NUTS_blackjax`
# and `HMC_blackjax` are listed explicitly so a user reaching for the
# BlackJAX variant of an already-shipped kernel gets a clear pointer
# instead of a same-name shadow of the NumPyro one.
_DEFERRED_BLACKJAX_KERNELS = (
    "NUTS_blackjax", "HMC_blackjax",  # BlackJAX variants of NUTS/HMC
    "SGLD", "MALA",
    "AdjustedMCLMCDynamic",
)


def __getattr__(name: str):
    if name in _DEFERRED_NUMPYRO_KERNELS:
        raise AttributeError(
            f"effectful_mcmc: NumPyro kernel {name!r} is not exposed yet. "
            f"Only NUTS is currently shipped; open an issue if you need this "
            f"kernel and we'll prioritise."
        )
    if name in _DEFERRED_BLACKJAX_KERNELS:
        raise AttributeError(
            f"effectful_mcmc: kernel {name!r} requires the BlackJAX backend, "
            f"which isn't wired into the bridge yet."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
