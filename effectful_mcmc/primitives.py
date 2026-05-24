"""Defops + default NumPyro-backed handler + shared name helper.

Four primitives:
    - sample(d, obs=None, *, name=None, infer=None)
    - factor(log_factor, *, name=None)
    - param(init_value=None, *, name=None, constraint=None, event_dim=None)
    - deterministic(value, *, name=None)

Default handler `_NumPyroBackend` forwards each to its `numpyro.*`
analogue. Site names are resolved through a single helper `_site_name`
that's also used by the compile pass's `_CompilerBackend`, so the two
backends can't drift on how anonymous sites get named.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import numpyro

from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Expr, NotHandled, Operation, Term


# Process-global monotonic counter for anonymous-site names when no
# externally-held `var` is available (i.e. _NumPyroBackend.sample
# under the default eager path). Necessary because id() of a transient
# Operation collides when garbage-collected and re-allocated at the
# same address. `_CompilerBackend` doesn't need this — it holds the
# Operation alive in SampleSite, so id() is stable; we use id() there
# for cross-run stability of dict keys within a SampleSite list.
_ANON_COUNTER = itertools.count()


@defop
def sample(
    d, obs=None, *, name: str | None = None, infer=None,
) -> Expr[jax.Array]:
    """Sample from `d`.

    Returns an `Expr[jax.Array]` — either a concrete value (under
    `_NumPyroBackend`) or a free-variable term backed by an `Operation`
    (under the compile pass's `_CompilerBackend`). Either arm is usable
    in downstream distribution constructors via the term registrations
    in `effectful.handlers.numpyro`.

    Pass `name=` to expose the site for string-keyed access via
    `mcmc.get_samples_by_name()` or for intervention via
    `Intervene(name, ...)`. When `name` is omitted, the bridge
    synthesises one from a fresh `Operation`'s identity.
    """
    raise NotHandled


@defop
def factor(log_factor, *, name: str | None = None) -> None:
    """Add `log_factor` to the joint log-density."""
    raise NotHandled


@defop
def param(
    init_value=None, *, name: str | None = None,
    constraint=None, event_dim: int | None = None,
) -> Expr[jax.Array]:
    """Register an optimisable parameter."""
    raise NotHandled


@defop
def deterministic(value, *, name: str | None = None) -> Expr[jax.Array]:
    """Record `value` as a deterministic site so it appears in posterior
    samples without being a free latent."""
    raise NotHandled


def _site_name(public_name: str | None, var: Operation | None = None) -> str:
    """Single source of truth for NumPyro site-name synthesis.

    Called by both `_NumPyroBackend` and the compile pass's
    `_CompilerBackend` so the two backends can't drift.

    Three cases:
      1. `public_name` supplied → use it verbatim.
      2. `var` supplied → use `id(var)` (caller guarantees `var` outlives
         the resulting name's use; `_CompilerBackend` does this by
         storing `var` in `SampleSite.var`).
      3. Neither supplied → use a process-global monotonic counter
         (`_ANON_COUNTER`). `id()` would be unsafe here because no caller
         is holding the Operation alive; Python GC would free it and the
         next anonymous call could land at the same address, producing
         duplicate names that NumPyro rejects.

    The Operation's `__name__` is *not* usable — it defaults to the
    wrapped type's name (e.g. `'Array'`), which collides across sites.
    """
    if public_name is not None:
        return public_name
    if var is None:
        return f"_site_{next(_ANON_COUNTER)}"
    return f"_site_{id(var):x}"


class _NumPyroBackend(ObjectInterpretation):
    """Default handler: forwards effectful defops to `numpyro.*`
    primitives. Stateless — anonymous-name uniqueness is guaranteed by
    `Operation.define`, not by an instance counter."""

    @implements(sample)
    def sample(self, d, obs=None, *, name=None, infer=None):
        kwargs: dict[str, Any] = {}
        if infer is not None:
            kwargs["infer"] = infer
        return numpyro.sample(_site_name(name), d, obs=obs, **kwargs)

    @implements(factor)
    def factor(self, log_factor, *, name=None):
        return numpyro.factor(_site_name(name), log_factor)

    @implements(param)
    def param(self, init_value=None, *, name=None, constraint=None, event_dim=None):
        kwargs: dict[str, Any] = {}
        if constraint is not None:
            kwargs["constraint"] = constraint
        if event_dim is not None:
            kwargs["event_dim"] = event_dim
        return numpyro.param(_site_name(name), init_value, **kwargs)

    @implements(deterministic)
    def deterministic(self, value, *, name=None):
        return numpyro.deterministic(_site_name(name), value)
