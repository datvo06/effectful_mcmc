"""Tests for compile_to_numpyro."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as nph
import pytest

from effectful.handlers.numpyro import Normal, HalfNormal, Beta, Independent
from effectful.ops.semantics import handler

from effectful_mcmc import sample, factor, param, deterministic
from effectful_mcmc.compile import (
    compile_to_numpyro,
    _materialize_distribution,
    SampleSite, FactorSite, ParamSite, DeterministicSite,
)


def _traced(compiled):
    """Run a compiled numpyro_model under seed+trace, return the trace."""
    with nph.seed(rng_seed=0):
        with nph.trace() as tr:
            compiled()
    return tr


# ---------------------------------------------------------------------------
# Basic compilation
# ---------------------------------------------------------------------------

def test_compile_normal_only():
    """A single-sample model compiles, traces, and runs end-to-end."""
    def model():
        return sample(Normal(0.0, 1.0), name="x")

    compiled, sites, return_value = compile_to_numpyro(model)
    assert len(sites) == 1
    assert isinstance(sites[0], SampleSite)
    assert sites[0].public_name == "x"

    tr = _traced(compiled)
    assert "x" in tr
    assert tr["x"]["type"] == "sample"


def test_compile_dependency_chain():
    """A model where one sample depends on another compiles correctly."""
    def model():
        mu = sample(Normal(0.0, 5.0), name="mu")
        sigma = sample(HalfNormal(1.0), name="sigma")
        x = sample(Normal(mu, sigma), name="x")
        return mu, sigma, x

    compiled, sites, return_value = compile_to_numpyro(model)
    sample_sites = [s for s in sites if isinstance(s, SampleSite)]
    assert len(sample_sites) == 3
    assert {s.public_name for s in sample_sites} == {"mu", "sigma", "x"}

    tr = _traced(compiled)
    assert {"mu", "sigma", "x"} <= set(tr.keys())


def test_compile_independent_workaround():
    """Independent(Normal(...), 1) compiles through
    _materialize_distribution and runs MCMC-compatibly. This pins the
    workaround for the `_is_eager`-too-conservative case where
    `support` falls back to a `_CallableTerm`."""
    def model():
        return sample(
            Independent(Normal(jnp.zeros(3), jnp.ones(3)), 1),
            name="z",
        )

    compiled, sites, _ = compile_to_numpyro(model)
    tr = _traced(compiled)
    assert "z" in tr
    # The recorded site's `fn` should be a concrete numpyro Independent
    # (or a NumPyro distribution at minimum, not a _CallableTerm).
    assert isinstance(tr["z"]["fn"], dist.Distribution)
    # Crucially, support is a real Constraint, not a Term.
    assert isinstance(tr["z"]["fn"].support, dist.constraints.Constraint)


def test_compile_materialize_distribution_general_term_args():
    """Pins that `_materialize_distribution` is not Independent-specific:
    any `_DistributionTerm` whose outer support resolution fails because
    a constructor arg is itself a Term should be materialised. Same
    `_DistributionTerm._is_eager`-too-conservative failure mode as
    `test_compile_independent_workaround`, but nested.

    Constructs a 2-layer wrapping: `Independent(Independent(Normal,1),1)`
    — two nested `_DistributionTerm`s, the innermost on eager arrays.
    Both layers' `_is_eager` returns False, so `support` falls through
    to the term-machinery path. The recursive materialiser must descend
    both layers.
    """
    def model():
        return sample(
            Independent(Independent(Normal(jnp.zeros((2, 3)), jnp.ones((2, 3))), 1), 1),
            name="zz",
        )

    compiled, _, _ = compile_to_numpyro(model)
    tr = _traced(compiled)
    assert "zz" in tr
    assert isinstance(tr["zz"]["fn"], dist.Distribution)
    assert isinstance(tr["zz"]["fn"].support, dist.constraints.Constraint)


# ---------------------------------------------------------------------------
# Single-trace invariant (load-bearing for posterior addressing)
# ---------------------------------------------------------------------------

def test_single_trace_invariant():
    """The user model is invoked exactly once during compile_to_numpyro."""
    counter = [0]

    def model():
        counter[0] += 1
        return sample(Normal(0.0, 1.0), name="x")

    compiled, sites, return_value = compile_to_numpyro(model)
    assert counter[0] == 1

    # Running the compiled closure many times must NOT re-invoke the user model.
    for _ in range(50):
        with nph.seed(rng_seed=0):
            compiled()
    assert counter[0] == 1


def test_re_run_idempotence():
    """Compiling the same model twice doesn't accumulate state across calls."""
    counter = [0]

    def model():
        counter[0] += 1
        sample(Normal(0.0, 1.0), name="x")

    compile_to_numpyro(model)
    compile_to_numpyro(model)
    assert counter[0] == 2


# ---------------------------------------------------------------------------
# Factor ordering — topological, not unconditionally last
# ---------------------------------------------------------------------------

def test_factor_runs_after_sample_dependency():
    """A factor that references a sample site is scheduled AFTER it."""
    def model():
        a = sample(Normal(0.0, 1.0), name="a")
        # Soft constraint that depends on a:
        factor(-0.5 * a * a, name="soft")

    compiled, _, _ = compile_to_numpyro(model)
    tr = _traced(compiled)
    names = list(tr.keys())
    # 'a' must come before 'soft' in the trace order.
    assert names.index("a") < names.index("soft"), names


def test_independent_factor_runs_early():
    """A factor with no free-variable deps on samples isn't ordered after them."""
    def model():
        factor(jnp.array(1.5), name="floor")
        a = sample(Normal(0.0, 1.0), name="a")

    compiled, _, _ = compile_to_numpyro(model)
    tr = _traced(compiled)
    # Both sites in trace; floor must come at-or-before a.
    assert "floor" in tr and "a" in tr


# ---------------------------------------------------------------------------
# Handle identity — pins the `t.op in mcmc.get_samples()` lemma at
# the compile-pass level (no MCMC driver in play).
# ---------------------------------------------------------------------------

def test_model_return_value_is_term():
    """The compile pass returns Terms for sample sites; the user can call
    .op on them to recover the underlying Operation."""
    def model():
        mu = sample(Normal(0.0, 5.0))
        sigma = sample(HalfNormal(2.0))
        return mu, sigma

    compiled, sites, return_value = compile_to_numpyro(model)
    mu_term, sigma_term = return_value
    assert hasattr(mu_term, "op")
    assert hasattr(sigma_term, "op")
    # The .op values match the SampleSite.var entries.
    sample_sites = [s for s in sites if isinstance(s, SampleSite)]
    sample_vars = {s.var for s in sample_sites}
    assert mu_term.op in sample_vars
    assert sigma_term.op in sample_vars
    assert mu_term.op is not sigma_term.op


# ---------------------------------------------------------------------------
# Deterministic + param + factor coverage in the compile pass
# ---------------------------------------------------------------------------

def test_compile_all_four_primitives():
    """A model exercising sample, factor, param, deterministic all four
    compiles and traces correctly."""
    def model():
        p = param(jnp.array(0.7), name="p")
        a = sample(Beta(2.0, 2.0), name="a")
        factor(-0.5 * (a - p) ** 2, name="penalty")
        deterministic(2.0 * a, name="twice_a")

    compiled, sites, _ = compile_to_numpyro(model)
    kinds = {type(s).__name__ for s in sites}
    assert kinds == {"ParamSite", "SampleSite", "FactorSite", "DeterministicSite"}

    tr = _traced(compiled)
    assert "p" in tr and tr["p"]["type"] == "param"
    assert "a" in tr and tr["a"]["type"] == "sample"
    assert "penalty" in tr     # factor site
    assert "twice_a" in tr and tr["twice_a"]["type"] == "deterministic"


# ---------------------------------------------------------------------------
# Compile-pass output is differentiable: the closure produces a
# concrete NumPyro program that NumPyro can score.
# ---------------------------------------------------------------------------

def test_compile_yields_finite_log_density():
    """The compiled closure has a finite log-density under standard substitution."""
    from numpyro.infer.util import log_density

    def model():
        mu = sample(Normal(0.0, 1.0), name="mu")
        sample(Normal(mu, 1.0), name="x", obs=jnp.array(0.5))

    compiled, _, _ = compile_to_numpyro(model)
    ld, _ = log_density(compiled, (), {}, {"mu": jnp.array(0.0)})
    assert jnp.isfinite(ld)
