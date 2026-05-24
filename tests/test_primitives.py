"""Tests for primitives.

Algebraic-law tests only (no literal-name assertions, no gold values).
These laws pin the contract that the compile pass, kernel facade, and
intervention helper all depend on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as nph
import pytest

from effectful.handlers.numpyro import Normal, HalfNormal
from effectful.ops.semantics import coproduct, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements

from effectful_mcmc import sample, factor, param, deterministic, _NumPyroBackend
from effectful_mcmc.primitives import _site_name


# ---------------------------------------------------------------------------
# Trace helpers — extract sites by *type* (sample/factor/param/deterministic),
# not by literal name, so tests don't assert on synthesised name strings.
# ---------------------------------------------------------------------------

def _traced(model, *args, **kwargs):
    """Run `model(*args, **kwargs)` under `_NumPyroBackend` + numpyro.trace,
    return the trace dict."""
    with handler(_NumPyroBackend()):
        with nph.seed(rng_seed=0):
            with nph.trace() as tr:
                model(*args, **kwargs)
    return tr


def _sites_of_type(tr, type_):
    return [s for s in tr.values() if s["type"] == type_]


# ---------------------------------------------------------------------------
# Equational laws
# ---------------------------------------------------------------------------

def test_obs_respect_law():
    """For any handler stack, sample(d, obs=v) returns v."""
    v = jnp.array(2.5)

    def model():
        return sample(Normal(0.0, 1.0), obs=v, name="x")

    with handler(_NumPyroBackend()):
        with nph.seed(rng_seed=0):
            result = model()
    assert jnp.array_equal(result, v)


def test_name_honoured_sample():
    """When name='a' is supplied, the NumPyro trace contains a site keyed 'a'."""
    def model():
        sample(Normal(0.0, 1.0), name="alpha")

    tr = _traced(model)
    assert "alpha" in tr
    assert tr["alpha"]["type"] == "sample"


def test_name_honoured_factor():
    def model():
        factor(jnp.array(1.5), name="bonus")

    tr = _traced(model)
    assert "bonus" in tr
    # NumPyro represents factor as a sample site with a Unit distribution
    # carrying log_factor; either "sample" or "deterministic" is acceptable
    # — just assert the site exists by the user-supplied name.


def test_name_honoured_param():
    def model():
        param(jnp.array(0.7), name="theta")

    tr = _traced(model)
    assert "theta" in tr
    assert tr["theta"]["type"] == "param"


def test_name_honoured_deterministic():
    def model():
        deterministic(jnp.array(42.0), name="answer")

    tr = _traced(model)
    assert "answer" in tr
    assert tr["answer"]["type"] == "deterministic"


def test_name_injectivity_anonymous_sample():
    """N anonymous sample calls produce N distinct site keys.

    Does NOT assert literal names (`_site_0` etc.) — only that the
    synthesised names are unique. Uniqueness is guaranteed by
    Operation.define producing globally-unique Operations.
    """
    N = 5

    def model():
        for _ in range(N):
            sample(Normal(0.0, 1.0))

    tr = _traced(model)
    sample_sites = _sites_of_type(tr, "sample")
    assert len(sample_sites) == N
    names = [s["name"] for s in sample_sites]
    assert len(set(names)) == N, f"duplicate names: {names}"


def test_name_injectivity_across_models():
    """Distinct sites across two independent traces don't collide.

    Operation.define produces globally-unique names by construction,
    so re-running the same model twice yields disjoint anonymous-site
    name sets. Property-style assertion rather than literal-name pin
    so the test doesn't break when the name format is refactored.
    """
    def model():
        sample(Normal(0.0, 1.0))
        sample(HalfNormal(1.0))

    tr1 = _traced(model)
    tr2 = _traced(model)
    keys1 = {s["name"] for s in _sites_of_type(tr1, "sample")}
    keys2 = {s["name"] for s in _sites_of_type(tr2, "sample")}
    # Disjoint — no collision (Operation identity is globally unique).
    assert keys1.isdisjoint(keys2), f"collision: {keys1 & keys2}"


def test_factor_identity_law():
    """factor(0.0) leaves the joint log-density unchanged."""
    from numpyro.infer.util import log_density

    def model_with_factor():
        x = sample(Normal(0.0, 1.0), name="x")
        factor(jnp.array(0.0), name="noop")

    def model_without_factor():
        x = sample(Normal(0.0, 1.0), name="x")

    with handler(_NumPyroBackend()):
        ld_with, _    = log_density(model_with_factor,    (), {}, {"x": jnp.array(0.5)})
        ld_without, _ = log_density(model_without_factor, (), {}, {"x": jnp.array(0.5)})
    assert float(ld_with) == pytest.approx(float(ld_without), abs=1e-6)


def test_deterministic_passthrough_law():
    """deterministic(value) returns value unchanged and inserts a
    deterministic-type site in the trace."""
    v = jnp.array([1.0, 2.0, 3.0])

    def model():
        return deterministic(v, name="d")

    with handler(_NumPyroBackend()):
        with nph.seed(rng_seed=0):
            with nph.trace() as tr:
                result = model()
    assert jnp.array_equal(result, v)
    assert tr["d"]["type"] == "deterministic"
    assert jnp.array_equal(tr["d"]["value"], v)


def test_param_default_law():
    """Under _NumPyroBackend, param(init_value=v) returns v on first call."""
    v = jnp.array(0.3)

    def model():
        return param(v, name="p")

    with handler(_NumPyroBackend()):
        with nph.seed(rng_seed=0):
            with nph.trace() as tr:
                result = model()
    assert jnp.array_equal(result, v)


def test_composition_identity_law_sample():
    """Stacking a no-op handler via coproduct doesn't change semantics."""

    class _NoOp(ObjectInterpretation):
        @implements(sample)
        def sample(self, d, obs=None, *, name=None, infer=None):
            return fwd()

    def model():
        return sample(Normal(0.0, 1.0), name="x")

    # Same RNG seed, same stack except for the no-op layer.
    with handler(_NumPyroBackend()):
        with nph.seed(rng_seed=42):
            v_baseline = model()

    with handler(_NumPyroBackend()):
        with handler(_NoOp()):
            with nph.seed(rng_seed=42):
                v_with_noop = model()

    assert jnp.allclose(v_baseline, v_with_noop)


# ---------------------------------------------------------------------------
# Shared helper: _site_name should match between _NumPyroBackend
# and _CompilerBackend callers.
# ---------------------------------------------------------------------------

def test_site_name_passthrough():
    assert _site_name("explicit") == "explicit"
    assert _site_name("explicit", var=None) == "explicit"


def test_site_name_anonymous_uses_operation():
    """When public_name is None, _site_name returns a unique Operation-derived name."""
    n1 = _site_name(None)
    n2 = _site_name(None)
    assert n1 != n2     # globally-unique Operations
    assert isinstance(n1, str) and isinstance(n2, str)


def test_site_name_anonymous_with_var():
    """When var is supplied, the same name is returned across repeated calls.

    Operations of type `jax.Array` all share `__name__ == 'Array'` (the
    type name) — not unique. So `_site_name` derives uniqueness from
    `id(var)` instead. The contract: same `var` → same name (idempotent);
    different `var`s → different names.
    """
    from effectful.ops.types import Operation
    op1 = Operation.define(jax.Array)
    op2 = Operation.define(jax.Array)
    assert _site_name(None, op1) == _site_name(None, op1)   # idempotent
    assert _site_name(None, op1) != _site_name(None, op2)   # injective
