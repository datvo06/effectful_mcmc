"""compile_to_numpyro: effectful model → NumPyro model closure.

Single-trace semantics: the user's model runs exactly once during
compilation. The trace records every sample / factor / param /
deterministic site via `_CompilerBackend`. The returned closure
re-emits those sites as `numpyro.{sample, factor, param, deterministic}`
calls, in topological order over free-variable dependencies.

The compile pass also works around an effectful issue —
`Independent`'s `support` returns an unevaluated Term — by recursively
materialising `_DistributionTerm`s whose support is non-concrete. See
`_materialize_distribution` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Any, Callable

import jax
import numpyro
import numpyro.distributions as dist

from effectful.handlers.jax import bind_dims, sizesof
from effectful.handlers.numpyro import _DistributionTerm
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled, Operation, Term

from effectful_mcmc.primitives import (
    _site_name,
    deterministic,
    factor,
    param,
    sample,
)


# ---------------------------------------------------------------------------
# Site records
# ---------------------------------------------------------------------------

@dataclass
class SampleSite:
    var: Operation             # free variable returned to user code
    dist: Any                  # distribution (may be a Term)
    obs: Any | None
    public_name: str | None
    infer: dict | None = None


@dataclass
class FactorSite:
    log_factor: Any
    public_name: str | None
    idx: int                   # source order, used for auto-naming


@dataclass
class ParamSite:
    var: Operation
    init_value: Any
    public_name: str | None
    constraint: Any | None
    event_dim: int | None


@dataclass
class DeterministicSite:
    value: Any
    public_name: str | None
    idx: int


# ---------------------------------------------------------------------------
# Compile-time handler
# ---------------------------------------------------------------------------

class _CompilerBackend(ObjectInterpretation):
    """Records every primitive call during one compile-time trace.

    Returns free-variable terms (`Op()`) for `sample` and `param` so
    downstream user code can use the result symbolically in distribution
    constructors and arithmetic. `factor` returns `None`; `deterministic`
    returns its value (passthrough), so the user can use it downstream.
    """

    def __init__(self):
        super().__init__()
        self.samples: list[SampleSite] = []
        self.factors: list[FactorSite] = []
        self.params: list[ParamSite] = []
        self.deterministics: list[DeterministicSite] = []

    @implements(sample)
    def sample(self, d, obs=None, *, name=None, infer=None):
        var = Operation.define(jax.Array)
        self.samples.append(SampleSite(
            var=var, dist=d, obs=obs, public_name=name, infer=infer,
        ))
        return var()                          # free-variable term

    @implements(factor)
    def factor(self, log_factor, *, name=None):
        self.factors.append(FactorSite(
            log_factor=log_factor, public_name=name, idx=len(self.factors),
        ))
        return None

    @implements(param)
    def param(self, init_value=None, *, name=None, constraint=None, event_dim=None):
        var = Operation.define(jax.Array)
        self.params.append(ParamSite(
            var=var, init_value=init_value, public_name=name,
            constraint=constraint, event_dim=event_dim,
        ))
        return var()                          # free-variable term

    @implements(deterministic)
    def deterministic(self, value, *, name=None):
        self.deterministics.append(DeterministicSite(
            value=value, public_name=name, idx=len(self.deterministics),
        ))
        return value                          # passthrough — value may be a Term


# ---------------------------------------------------------------------------
# Distribution + obs materialisation
#
# Two failure modes to handle at the NumPyro boundary:
#
#   1. `_DistributionTerm._is_eager` returning False when any arg is a
#      Term, even concrete ones (e.g. `Independent(Normal(zeros, ones), 1)`).
#      `d.support` then returns an unevaluated `_CallableTerm` and
#      `biject_to` raises.
#
#   2. `jax_getitem(t, (sym(),))` producing a term whose `.shape`
#      contains symbolic dims. Passed to a NumPyro distribution
#      constructor (or as `obs`), the constructor's internal
#      `jax.lax.broadcast_shapes` iterates that shape tuple and hits
#      `_CallableTerm.__iter__` which raises.
#
# `_materialize_distribution` handles (1) by reconstructing via
# `d._constr(*args, **kwargs)`, recursively materialising nested
# term-wrapped distribution args. `_materialize_arg` handles (2) by
# calling `bind_dims(a, *symbols)` to promote symbolic dims to leading
# positional dims, producing a concrete-shape array.
#
# Both fixes apply uniformly to every site's distribution AND to its
# `obs` (when set), via `_materialize_arg`.
# ---------------------------------------------------------------------------

def _materialize_arg(a: Any) -> Any:
    """Materialise a possibly-term value for passing to NumPyro.

    Three cases:
      1. `_DistributionTerm` → recurse via `_materialize_distribution`.
      2. Term with non-empty `sizesof` (free named dims, e.g. from
         `jax_getitem(t, (sym(),))`) → `bind_dims(a, *symbols)` to
         produce a concrete-shape array. Symbols are bound in
         name-sorted order for determinism.
      3. Anything else (concrete array, scalar, already-eager Term
         resolved through the surrounding `handler(bound)` context):
         return as-is.
    """
    # Order matters: a `_DistributionTerm` IS a `Term`, so check
    # distribution-ness first.
    if isinstance(a, dist.Distribution) and isinstance(a, _DistributionTerm):
        return _materialize_distribution(a)
    if isinstance(a, Term):
        named_dims = sizesof(a)
        if named_dims:
            syms = sorted(named_dims.keys(),
                          key=lambda s: getattr(s, "__name__", id(s)))
            return bind_dims(a, *syms)
    return a


def _materialize_distribution(d: Any) -> dist.Distribution:
    """Reconstruct a `_DistributionTerm` into a concrete NumPyro
    distribution. Only reconstructs when needed (see module docstring
    for the two failure modes); otherwise returns `d` as-is so that
    NumPyro continues to see the effectful term wrapper, which it
    handles correctly for the common cases.

    Reconstruction triggers on either:
      1. Support resolves to a Term / fails to resolve (Independent
         and friends — `_is_eager` is too conservative).
      2. Any constructor arg has free named dims (symbolic-shape
         arrays from `jax_getitem(t, (sym(),))`).

    The "leave it alone" path matters for cases like discrete
    distributions under NUTS, where the bare `_DistributionTerm`
    routes through NumPyro's discrete-site error path correctly; an
    unconditional reconstruction would land it in NumPyro's silent
    enumeration path instead, masking the intended error.
    """
    if not isinstance(d, dist.Distribution):
        return d
    if not isinstance(d, _DistributionTerm):
        return d

    # Trigger 1: support is a Term or doesn't resolve.
    try:
        support = d.support
        support_ok = isinstance(support, dist.constraints.Constraint)
    except (NotImplementedError, AttributeError):
        support_ok = False

    # Trigger 2: any arg has symbolic free dims.
    def _has_symbolic_shape(a: Any) -> bool:
        return isinstance(a, Term) and bool(sizesof(a))
    args_symbolic = (
        any(_has_symbolic_shape(a) for a in d._args)
        or any(_has_symbolic_shape(v) for v in d._kwargs.values())
    )

    if support_ok and not args_symbolic:
        return d                              # neither trigger fires; pass through

    new_args = tuple(_materialize_arg(a) for a in d._args)
    new_kwargs = {
        k: _materialize_arg(v)
        for k, v in d._kwargs.items()
        if v is not None                      # numpyro is strict about None
    }
    return d._constr(*new_args, **new_kwargs)


# ---------------------------------------------------------------------------
# Topological scheduling
# ---------------------------------------------------------------------------

def _schedule(
    samples: list[SampleSite], factors: list[FactorSite],
) -> list[SampleSite | FactorSite]:
    """Order samples and factors so dependencies are bound before use.

    Uses `fvsof` (free variables of a term) on each site's payload to
    discover dependencies on prior sample sites. Factors that depend on
    sample sites run after those samples; independent factors run early.
    """
    sample_vars = {s.var for s in samples}
    sites = list(samples) + list(factors)
    site_by_var = {s.var: s for s in samples}

    # Build a dependency graph keyed by site (using id as a hashable key).
    deps: dict[Any, set[Any]] = {}
    for s in sites:
        payload = s.dist if isinstance(s, SampleSite) else s.log_factor
        free_vars = fvsof(payload) & sample_vars
        # SampleSite shouldn't list itself as a dep (avoid self-loops).
        free_vars = free_vars - ({s.var} if isinstance(s, SampleSite) else set())
        deps[id(s)] = {id(site_by_var[v]) for v in free_vars}

    topo = TopologicalSorter(deps)
    by_id = {id(s): s for s in sites}
    return [by_id[i] for i in topo.static_order()]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_to_numpyro(
    model: Callable[..., Any], *args: Any, **kwargs: Any,
) -> tuple[Callable[[], None], list, Any]:
    """Run `model(*args, **kwargs)` once under `_CompilerBackend`,
    capture all primitive sites, and return a zero-argument NumPyro
    model closure that re-emits them in dependency order.

    Returns:
        (closure, sites, model_return_value) where
            - `closure()` runs the compiled NumPyro program once;
            - `sites` is a list of `SampleSite | FactorSite | ParamSite
                                    | DeterministicSite` records;
            - `model_return_value` is whatever the user's model returned
              from the compile-time trace.
    """
    backend = _CompilerBackend()
    with handler(backend):
        model_return_value = model(*args, **kwargs)

    samples = backend.samples
    factors = backend.factors
    params = backend.params
    deterministics = backend.deterministics
    sample_factor_order = _schedule(samples, factors)

    # All records, in canonical NumPyro emit order:
    #   params -> samples+factors (topological) -> deterministics
    sites: list = list(params) + list(sample_factor_order) + list(deterministics)

    def numpyro_model(*_args: Any, **_kwargs: Any) -> None:
        """Compiled closure. `*_args, **_kwargs` are ignored — the user
        model's args were captured at compile time and baked into the
        recorded sites. NumPyro's `MCMC.run` forwards its own args
        through to this closure on every iteration; we accept them so
        the signature lines up and ignore them so the compile-time
        capture is the single source of truth (the single-trace
        invariant)."""
        # Plain dict carrier — `handler(dict)` is accepted directly by
        # effectful.
        bound: dict[Operation, Callable[[], Any]] = {}

        for site in sites:
            if isinstance(site, ParamSite):
                site_kwargs: dict[str, Any] = {}
                if site.constraint is not None:
                    site_kwargs["constraint"] = site.constraint
                if site.event_dim is not None:
                    site_kwargs["event_dim"] = site.event_dim
                v = numpyro.param(
                    _site_name(site.public_name, site.var),
                    site.init_value,
                    **site_kwargs,
                )
                bound[site.var] = (lambda v=v: v)

            elif isinstance(site, SampleSite):
                with handler(bound):
                    d_evaluated = evaluate(site.dist)
                d_concrete = _materialize_distribution(d_evaluated)
                site_kwargs = {}
                if site.obs is not None:
                    # obs may itself be a `jax_getitem`-produced term with
                    # symbolic shape; materialise via `_materialize_arg` so
                    # NumPyro sees a concrete-shape array.
                    with handler(bound):
                        obs_evaluated = evaluate(site.obs)
                    site_kwargs["obs"] = _materialize_arg(obs_evaluated)
                if site.infer is not None:
                    site_kwargs["infer"] = site.infer
                v = numpyro.sample(
                    _site_name(site.public_name, site.var),
                    d_concrete,
                    **site_kwargs,
                )
                bound[site.var] = (lambda v=v: v)

            elif isinstance(site, FactorSite):
                with handler(bound):
                    log_factor_val = evaluate(site.log_factor)
                np_name = site.public_name or f"_factor_{site.idx}"
                numpyro.factor(np_name, log_factor_val)

            elif isinstance(site, DeterministicSite):
                with handler(bound):
                    value_val = evaluate(site.value)
                numpyro.deterministic(
                    _site_name(site.public_name) if site.public_name
                    else f"_deterministic_{site.idx}",
                    value_val,
                )

            else:
                raise TypeError(f"Unknown site type: {type(site).__name__}")

    return numpyro_model, sites, model_return_value
