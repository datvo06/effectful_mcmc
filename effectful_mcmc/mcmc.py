"""MCMC driver: thin wrapper around numpyro.infer.MCMC.

Wraps rather than subclasses because the underlying NumPyro kernel
can't be built until the user model is compiled, and compilation
requires runtime model args. A subclass with a deferred
`super().__init__` would be `isinstance(m, numpyro.infer.MCMC)` true
but `m.num_warmup` raise — an LSP violation. Honest wrapping with
`__getattr__` / `__setattr__` forwarding for post-run attribute
access is cleaner.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpyro.infer

from effectful_mcmc.primitives import _site_name


# Attributes the wrapper owns (vs. forwarded to the wrapped NumPyro
# MCMC). Listed at module scope so __setattr__ can decide without
# triggering its own recursion on `self.__dict__`. Anything not in this
# set, when assigned post-run, forwards to `_np_mcmc` — this is what
# makes the warm-restart workflow `mcmc.post_warmup_state = prior.last_state`
# actually take effect.
_WRAPPER_OWN_ATTRS = frozenset({
    "_bridge_kernel", "_mcmc_kwargs", "_np_mcmc",
    "_sites", "model_return_value", "_pending_np_attrs",
})

# Names that look like real `numpyro.infer.MCMC` attributes — used by
# __getattr__ to distinguish "you called it too early" (the name exists
# on NumPyro but our wrapped instance doesn't yet) from "you misspelled
# it" (the name isn't on NumPyro either). Computed once at import time
# so a typo lookup isn't quadratic in calls.
_NP_MCMC_KNOWN_NAMES = (
    frozenset(dir(numpyro.infer.MCMC))
    | frozenset(inspect.signature(numpyro.infer.MCMC.__init__).parameters)
)


class MCMC:
    """Wraps `numpyro.infer.MCMC`. Compilation of the effectful user
    model happens lazily inside `.run(*args, **kwargs)` because the
    compile-time trace needs those args.

    Post-run attribute access (`print_summary`, `get_extra_fields`,
    `last_state`, `num_warmup`, `num_samples`, `num_chains`, etc.) is
    forwarded to the underlying NumPyro `MCMC` via `__getattr__`.
    Pre-run attribute access raises a clear `AttributeError`.
    """

    def __init__(self, kernel: Any, **mcmc_kwargs: Any):
        self._bridge_kernel = kernel
        self._mcmc_kwargs = mcmc_kwargs
        # Populated by .run() from the single compile-time trace:
        self._np_mcmc: numpyro.infer.MCMC | None = None
        self._sites: list | None = None
        self.model_return_value: Any = None

    def run(
        self, rng_key, *args: Any,
        extra_fields: tuple[str, ...] = (),
        init_params=None,
        **kwargs: Any,
    ):
        """Compile the user model with `(*args, **kwargs)`, build the
        underlying NumPyro MCMC, and delegate to it.

        `extra_fields` and `init_params` are the only two named kwargs
        on `numpyro.infer.MCMC.run` (verified via `inspect.signature`).

        `**kwargs` here is for the **user model**, not for NumPyro's
        `.run` — anything you pass via kwargs reaches the user model
        via the compile pass at compile time, then the user-model args
        are baked into the compiled closure (see `compile.py`).

        Warm-restart workflow: NumPyro doesn't take an `init_state`
        kwarg on `.run`; instead, set `mcmc.post_warmup_state =
        prior_mcmc.last_state` *before* calling `.run()`. To make this
        work for an MCMC instance whose `_np_mcmc` hasn't been built
        yet, `__setattr__` stashes such pre-run writes in a private
        `_pending_np_attrs` dict; this method flushes that dict into
        the newly-built `_np_mcmc` before invoking `.run()`. The result
        is that the documented workflow

            mcmc2 = MCMC(NUTS(model), num_warmup=…, num_samples=…)
            mcmc2.post_warmup_state = mcmc1.last_state
            mcmc2.run(key2)

        actually skips warmup, as users expect.
        """
        np_kernel, self._sites, self.model_return_value = (
            self._bridge_kernel._compile(*args, **kwargs))
        self._np_mcmc = numpyro.infer.MCMC(np_kernel, **self._mcmc_kwargs)
        # Flush any pre-run-set attributes (e.g. `post_warmup_state`)
        # into the freshly-built NumPyro MCMC.
        pending = self.__dict__.pop("_pending_np_attrs", {})
        for k, v in pending.items():
            setattr(self._np_mcmc, k, v)
        # Single source of truth for args: the compile pass captured them
        # into the recorded sites. We pass `()`/`{}` to NumPyro's `.run`
        # so the compiled closure (which intentionally ignores its own
        # `*_args, **_kwargs`) doesn't see the args twice.
        return self._np_mcmc.run(
            rng_key,
            extra_fields=extra_fields, init_params=init_params,
        )

    def get_samples(self, group_by_chain: bool = False) -> dict:
        """Posterior samples, keyed by `Operation` (the handle returned
        from `sample` in the user model). Use `get_samples_by_name()`
        for the NumPyro-shape string-keyed dict.

        Pre-run access raises `AttributeError`, matching the error
        model used by `__getattr__` for forwarded attributes — the
        wrapped MCMC object simply doesn't exist yet.
        """
        # Local import: `effectful_mcmc.compile` doesn't actually import
        # from `effectful_mcmc.mcmc`, but we put the import here anyway
        # to keep the driver decoupled from the compile pass's site
        # dataclasses at module import time (only callers of get_samples
        # need it).
        from effectful_mcmc.compile import SampleSite
        if self._np_mcmc is None:
            raise AttributeError(
                "MCMC.get_samples is only available after .run() is called"
            )
        np_samples = self._np_mcmc.get_samples(group_by_chain=group_by_chain)
        return {
            s.var: np_samples[_site_name(s.public_name, s.var)]
            for s in self._sites if isinstance(s, SampleSite) and s.obs is None
        }

    def get_samples_by_name(self, group_by_chain: bool = False) -> dict:
        """Posterior samples in the NumPyro string-keyed shape."""
        if self._np_mcmc is None:
            raise AttributeError(
                "MCMC.get_samples_by_name is only available after .run() is called"
            )
        return self._np_mcmc.get_samples(group_by_chain=group_by_chain)

    def __getattr__(self, name: str) -> Any:
        """Forward post-run attribute access to the underlying NumPyro MCMC.

        Three cases:
          1. `_np_mcmc` is set → forward via getattr.
          2. `_np_mcmc` is None and `name` is a known NumPyro `MCMC`
             attribute or constructor kwarg → "only available after
             .run()" message.
          3. `_np_mcmc` is None and `name` is unrecognised → typo-style
             "no attribute" message, so users see "you misspelled it"
             instead of "you called it too early".

        Python only calls __getattr__ for attributes not found normally,
        so this never shadows the wrapper's own methods.
        """
        np_mcmc = self.__dict__.get("_np_mcmc")
        if np_mcmc is not None:
            return getattr(np_mcmc, name)
        # Pre-run heuristic: distinguish "called too early" from "typo"
        # using the module-level cached name set.
        if name in _NP_MCMC_KNOWN_NAMES:
            raise AttributeError(
                f"MCMC.{name} is only available after .run() is called"
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignment to the underlying NumPyro MCMC,
        with stash-then-flush semantics for pre-run writes.

        Three cases:
          1. `name` is a wrapper-own attr (`_bridge_kernel`, `_np_mcmc`,
             `_sites`, `model_return_value`, `_mcmc_kwargs`,
             `_pending_np_attrs`) → store on `self.__dict__`.
          2. `_np_mcmc` exists → forward to NumPyro instance directly.
          3. `_np_mcmc` is None → stash in `_pending_np_attrs`; `run()`
             flushes the stash into the freshly-built NumPyro instance
             before invoking it. This makes the documented warm-restart
             workflow (set `post_warmup_state`, then call `.run`) work
             even on a freshly-constructed `MCMC` whose `_np_mcmc`
             hasn't been built yet.
        """
        if name in _WRAPPER_OWN_ATTRS:
            object.__setattr__(self, name, value)
            return
        np_mcmc = self.__dict__.get("_np_mcmc")
        if np_mcmc is None:
            pending = self.__dict__.setdefault("_pending_np_attrs", {})
            pending[name] = value
            return
        setattr(np_mcmc, name, value)
