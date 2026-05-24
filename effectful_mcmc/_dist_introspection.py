"""Distribution introspection: enumerate every distribution effectful
wraps in `effectful.handlers.numpyro`.

Single source of truth used by both `scripts/biject_to_spike.py` (the
constraint-support compatibility spike) and `tests/test_distributions.py`
(the full-distribution parametric smoke). They share it so the
parametric smoke can't silently drift if upstream effectful adds a
distribution.
"""

import inspect

import numpyro.distributions as dist

import effectful.handlers.numpyro  # noqa: F401  — register the wrappers
from effectful.ops.syntax import defdata


# Constraint *class names* whose support has no real-line bijection.
# Imported by `scripts/biject_to_spike.py` AND by
# `tests/test_distributions.py` so the two paths agree on what
# counts as "discrete" without depending on each other across the
# scripts/ boundary.
DISCRETE_CONSTRAINT_NAMES: frozenset[str] = frozenset({
    "_Boolean", "_IntegerInterval", "_IntegerGreaterThan",
    "_IntegerNonnegative",        # NumPyro spelling (lowercase n)
    "_IntegerPositive", "_Multinomial",
})


def list_registered_distributions() -> list[type[dist.Distribution]]:
    """Return every numpyro distribution class that has an effectful
    wrapper registered via @defdata.register(...).

    Uses `defdata._registry.registry` — singledispatch's public
    `registry` attribute on the underlying dispatch function exposed
    by effectful's `_CustomSingleDispatchCallable`. The earlier
    closure-cell access (`__closure__[2].cell_contents`) worked but
    was brittle against CPython cell-order changes.
    """
    try:
        registry = defdata._registry.registry
    except AttributeError as e:  # pragma: no cover
        raise RuntimeError(
            "effectful.ops.syntax.defdata layout changed; "
            "_dist_introspection.list_registered_distributions needs updating"
        ) from e

    classes = sorted(
        (t for t in registry
         if isinstance(t, type) and issubclass(t, dist.Distribution)),
        key=lambda t: t.__name__,
    )
    _sanity_check(classes)
    return classes


def _sanity_check(classes: list[type[dist.Distribution]]) -> None:
    """Assertions that catch silent enumeration breakage AND
    factory-dict drift. Each assertion is a hard raise: the point of
    `_sanity_check` is to be loud, not log a warning that CI swallows.
    """
    if len(classes) < 30:
        raise RuntimeError(
            f"_dist_introspection found only {len(classes)} registered "
            f"distributions; expected >=30. effectful internals likely "
            f"changed shape — see effectful_mcmc/_dist_introspection.py:30."
        )
    # Load-bearing distributions — every named one must be registered.
    required = {"Normal", "HalfNormal", "Beta", "Dirichlet",
                "LKJCholesky", "Independent"}
    names = {c.__name__ for c in classes}
    missing_required = required - names
    if missing_required:
        raise RuntimeError(
            f"_dist_introspection missing load-bearing distributions: {missing_required}"
        )
    # Factory coverage: every registered distribution must have a factory.
    # Without this check, a newly-added effectful distribution silently
    # gets `KeyError` in _FACTORIES and the parametric smoke either
    # crashes mid-run or skips it.
    missing_factories = names - set(_FACTORIES.keys())
    if missing_factories:
        raise RuntimeError(
            f"_dist_introspection missing factories for: {sorted(missing_factories)}. "
            f"Add entries to _FACTORIES in effectful_mcmc/_dist_introspection.py."
        )


# Factory dict: distribution class -> callable returning a minimal valid instance.
# Used by the spike and by the parametric smoke. Each factory uses
# the *effectful* wrapper if it exists (so we exercise the wrapper's __init__),
# falling back to the bare NumPyro class.
def _enp(name: str):
    """Look up a wrapper from effectful.handlers.numpyro by name."""
    import effectful.handlers.numpyro as enp
    return getattr(enp, name, getattr(dist, name))


def default_factory(d_cls: type[dist.Distribution]):
    """Return a callable that constructs a minimal valid instance of `d_cls`
    via the effectful wrapper. Raises KeyError if no factory is registered."""
    if d_cls.__name__ not in _FACTORIES:
        raise KeyError(
            f"No factory for {d_cls.__name__}; add one to "
            f"effectful_mcmc._dist_introspection._FACTORIES")
    return _FACTORIES[d_cls.__name__]


# Use effectful's wrapped `jax.numpy` for parity with effectful's own
# distribution wrappers. The wrapper auto-lifts every callable through
# `_register_jax_op`, adding named-tensor awareness. For the eager-array
# construction inside the factory lambdas here, the behaviour is
# identical to plain `jax.numpy`; the consistency win is that any code
# path that ever reaches these constructed values via term machinery
# handles them transparently.
def _f():
    import effectful.handlers.jax.numpy as jnp
    return {
        # Two-parameter location-scale, all defaults.
        "Cauchy":       lambda: _enp("Cauchy")(),
        "Gumbel":       lambda: _enp("Gumbel")(),
        "Laplace":      lambda: _enp("Laplace")(),
        "LogNormal":    lambda: _enp("LogNormal")(),
        "Logistic":     lambda: _enp("Logistic")(),
        "Normal":       lambda: _enp("Normal")(),
        "StudentT":     lambda: _enp("StudentT")(df=3.0),
        # Discrete probs/logits, scalar.
        "BernoulliProbs":   lambda: _enp("BernoulliProbs")(probs=0.5),
        "BernoulliLogits":  lambda: _enp("BernoulliLogits")(logits=0.0),
        "GeometricProbs":   lambda: _enp("GeometricProbs")(probs=0.5),
        "GeometricLogits":  lambda: _enp("GeometricLogits")(logits=0.0),
        # Discrete probs/logits, vector.
        "CategoricalProbs":   lambda: _enp("CategoricalProbs")(probs=jnp.array([0.2, 0.3, 0.5])),
        "CategoricalLogits":  lambda: _enp("CategoricalLogits")(logits=jnp.zeros(3)),
        # Beta-family.
        "Beta":         lambda: _enp("Beta")(2.0, 2.0),
        "Kumaraswamy":  lambda: _enp("Kumaraswamy")(2.0, 2.0),
        # Binomial / Multinomial.
        "BinomialProbs":         lambda: _enp("BinomialProbs")(probs=0.5, total_count=5),
        "BinomialLogits":        lambda: _enp("BinomialLogits")(logits=0.0, total_count=5),
        "NegativeBinomialProbs": lambda: _enp("NegativeBinomialProbs")(5, 0.5),
        "NegativeBinomialLogits":lambda: _enp("NegativeBinomialLogits")(5, 0.0),
        "MultinomialProbs":      lambda: _enp("MultinomialProbs")(probs=jnp.array([0.2, 0.3, 0.5]), total_count=5),
        "MultinomialLogits":     lambda: _enp("MultinomialLogits")(logits=jnp.zeros(3), total_count=5),
        # df-only / rate-only / concentration-only.
        "Chi2":         lambda: _enp("Chi2")(df=3.0),
        "Exponential":  lambda: _enp("Exponential")(),
        "Poisson":      lambda: _enp("Poisson")(rate=1.0),
        "Dirichlet":    lambda: _enp("Dirichlet")(jnp.ones(3)),
        "DirichletMultinomial": lambda: _enp("DirichletMultinomial")(jnp.ones(3), total_count=5),
        "Gamma":        lambda: _enp("Gamma")(2.0),
        # Half-real-line.
        "HalfCauchy":   lambda: _enp("HalfCauchy")(),
        "HalfNormal":   lambda: _enp("HalfNormal")(),
        # Multivariate.
        "LKJCholesky":  lambda: _enp("LKJCholesky")(dim=3, concentration=1.0),
        "MultivariateNormal": lambda: _enp("MultivariateNormal")(
            loc=jnp.zeros(3), covariance_matrix=jnp.eye(3)),
        "LowRankMultivariateNormal": lambda: _enp("LowRankMultivariateNormal")(
            loc=jnp.zeros(3), cov_factor=jnp.ones((3, 1)), cov_diag=jnp.ones(3)),
        "Wishart":      lambda: _enp("Wishart")(df=4.0, scale_tril=jnp.eye(3)),
        # Other parametric.
        "Pareto":       lambda: _enp("Pareto")(scale=1.0, alpha=2.0),
        "Uniform":      lambda: _enp("Uniform")(),
        "VonMises":     lambda: _enp("VonMises")(loc=0.0, concentration=1.0),
        "Weibull":      lambda: _enp("Weibull")(scale=1.0, concentration=1.0),
        # Special.
        "Delta":        lambda: _enp("Delta")(),
        "RelaxedBernoulliLogits": lambda: _enp("RelaxedBernoulliLogits")(
            temperature=0.5, logits=0.0),
        "Independent":  lambda: _enp("Independent")(
            _enp("Normal")(jnp.zeros(2), jnp.ones(2)), reinterpreted_batch_ndims=1),
    }


_FACTORIES = _f()
del _f
