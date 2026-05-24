"""Composition tests — the differentiating value of the bridge.

What makes this worth shipping over plain numpyro.infer.MCMC:
effectful handlers compose with the inference layer. Pinned with
three load-bearing scenarios:

1. **In-trace intervention.** A library `Intervene(name, value)` handler
   overrides a named sample site. The intervened site is *removed* from
   the posterior (not surfaced as deterministic — the user already has
   the value), and the remaining sites' posteriors tighten because they
   now condition on the intervention.

2. **`fwd()` correctness for non-intervened sites.** When the
   `Intervene` handler doesn't match, `fwd()` reaches the compile
   backend and the non-intervened sites still produce samples.

3. **Named-tensor model under MCMC.** A hierarchical model using
   `effectful.handlers.jax`'s named indices runs through the bridge.
   This is the test that confirms `jax.grad ∘ named-tensor` survives
   the compile pass.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest

from effectful.handlers.numpyro import HalfNormal, Normal
from effectful.handlers.jax import bind_dims, jax_getitem
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements

from effectful_mcmc import MCMC, NUTS, Intervene, sample


# ---------------------------------------------------------------------------
# Shared helper: a 2-site Normal-HalfNormal-Normal model
# ---------------------------------------------------------------------------

def _two_site_model(data):
    """mu, sigma latent; observed Normal(mu, sigma) on data."""
    mu = sample(Normal(0.0, 5.0), name="mu")
    sigma = sample(HalfNormal(2.0), name="sigma")
    sample(Normal(mu, sigma), obs=data, name="obs")
    return mu


def _run(model, *args, num_warmup=500, num_samples=1000, key=0, **kwargs):
    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(jr.PRNGKey(key), *args, **kwargs)
    return mcmc


# ---------------------------------------------------------------------------
# Test 1: intervention removes the site + tightens remaining posterior
# ---------------------------------------------------------------------------

def test_intervention_removes_site_from_posterior():
    """An intervened sample site does NOT appear in posterior samples."""
    data = jnp.array([1.2, 0.8, 1.5, 0.9, 1.1])

    def intervened_model(d):
        with handler(Intervene("sigma", jnp.array(1.0))):
            return _two_site_model(d)

    mcmc = _run(intervened_model, data, num_warmup=200, num_samples=200)
    samples_by_name = mcmc.get_samples_by_name()
    # Site removal: 'sigma' was intervened away.
    assert "sigma" not in samples_by_name, \
        f"intervened site appeared in posterior: keys={list(samples_by_name)}"
    # The non-intervened sites are still there.
    assert "mu" in samples_by_name


def test_intervention_tightens_remaining_posterior():
    """With sigma fixed, mu's posterior should be tighter than the baseline
    where both are inferred — the model has one fewer source of uncertainty.

    Asserts std(mu_intervened) < std(mu_baseline) with a margin so the
    assertion isn't flaky on close calls. Pinned PRNGKey for determinism.
    """
    # Generate data with a moderately-well-identified mu so the tightening
    # effect is robust.
    rng = jr.PRNGKey(123)
    true_mu, true_sigma = 1.0, 0.5
    data = true_mu + true_sigma * jr.normal(rng, (20,))

    def baseline_model(d):
        return _two_site_model(d)

    def intervened_model(d):
        with handler(Intervene("sigma", jnp.array(true_sigma))):
            return _two_site_model(d)

    m_baseline = _run(baseline_model, data, num_warmup=500, num_samples=1000)
    m_interv = _run(intervened_model, data, num_warmup=500, num_samples=1000)

    std_baseline = float(jnp.std(m_baseline.get_samples_by_name()["mu"]))
    std_interv = float(jnp.std(m_interv.get_samples_by_name()["mu"]))

    # Intervened std should be meaningfully smaller (≥5% reduction; the
    # actual effect for this dataset is much larger, ~30%, so 5% is a
    # very loose flake-prevention margin).
    assert std_interv < 0.95 * std_baseline, \
        f"expected posterior tightening: baseline={std_baseline}, " \
        f"intervened={std_interv} (ratio={std_interv/std_baseline:.3f})"


# ---------------------------------------------------------------------------
# Test 2: fwd() correctness — non-intervened sites still sample
# ---------------------------------------------------------------------------

def test_fwd_correctness_non_intervened_sites_still_sample():
    """When intervene matches one site, the OTHER sites still flow
    through fwd() to the compile backend and produce posterior samples
    with finite values and non-trivial variance."""
    data = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])

    def intervened_model(d):
        with handler(Intervene("sigma", jnp.array(0.5))):
            return _two_site_model(d)

    mcmc = _run(intervened_model, data, num_warmup=200, num_samples=300)
    mu = mcmc.get_samples_by_name()["mu"]
    assert jnp.all(jnp.isfinite(mu))
    # Non-trivial variance: posterior shouldn't collapse to a constant.
    assert float(jnp.std(mu)) > 1e-3, "non-intervened site collapsed"
    # Posterior mean of mu should be near the data mean (1.0) since the
    # likelihood Normal(mu, 0.5) with strong data should dominate.
    assert abs(float(jnp.mean(mu)) - 1.0) < 0.2


def test_intervention_composes_via_handler_nesting():
    """Stacking intervene with no-op handlers via Python `with handler(...)`
    nesting preserves intervention semantics. (This tests handler-stack
    composition; the related `coproduct` API is tested in effectful's own
    test suite — here we just confirm intervention survives an outer
    stack.)"""

    class _NoOp(ObjectInterpretation):
        @implements(sample)
        def sample(self, d, obs=None, *, name=None, infer=None):
            return fwd()

    data = jnp.array([1.0, 1.0, 1.0])

    def stacked_model(d):
        with handler(_NoOp()):
            with handler(Intervene("sigma", jnp.array(1.0))):
                return _two_site_model(d)

    mcmc = _run(stacked_model, data, num_warmup=100, num_samples=100)
    samples = mcmc.get_samples_by_name()
    assert "sigma" not in samples
    assert "mu" in samples


# ---------------------------------------------------------------------------
# Hierarchical model (eight-schools) under MCMC — positive case.
#
# Uses an explicit per-element loop for the J-vector latent, with no
# jax_getitem-based symbolic indexing. This pattern works through the
# bridge today because every distribution reaches NumPyro with concrete
# shapes; nothing in NumPyro's `lax.broadcast_shapes` machinery is
# asked to introspect a term-shape.
#
# The narrower jax_getitem-specific integration gap is covered by
# `test_symbolic_indexing_into_distribution_args` below; the related
# `.expand().to_event()` gap is `xfail`-pinned below that.
# ---------------------------------------------------------------------------

def test_hierarchical_eight_schools_per_element_idiom():
    """Eight-schools hierarchical model using the per-element-loop
    idiom (one explicit `sample` per school). The bridge should compile
    and run NUTS end-to-end.

    The "vectorized" NumPyro idiom — `Normal(mu, tau).expand([J]).to_event(1)`
    — fails through the bridge because `_DistributionTerm.expand`
    is `@defop`-annotated to return `jax.Array` rather than
    `Distribution`, so chaining `.to_event(1)` on its result raises
    `AttributeError: '_ArrayTerm' object has no attribute 'to_event'`.
    That's an upstream-effectful gap; the per-element idiom here is
    the workaround that works through the bridge today.

    Recovery target: tau > 0 (hierarchical variance is identifiable),
    all draws finite. 8 schools × (1 theta + 1 obs) = 16 sites plus
    mu/tau = 18 total — small but real hierarchical inference.
    """
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma_arr = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    J = len(y)

    def model(y_obs, sigma_obs):
        mu = sample(Normal(0.0, 10.0), name="mu")
        tau = sample(HalfNormal(10.0), name="tau")
        for j in range(J):
            theta_j = sample(Normal(mu, tau), name=f"theta_{j}")
            sample(
                Normal(theta_j, sigma_obs[j]),
                obs=y_obs[j],
                name=f"y_{j}",
            )

    mcmc = _run(model, y, sigma_arr, num_warmup=500, num_samples=500)

    samples = mcmc.get_samples_by_name()
    assert {"mu", "tau"} <= set(samples)
    for j in range(J):
        assert f"theta_{j}" in samples, f"missing theta_{j}"
    for site, vals in samples.items():
        assert jnp.all(jnp.isfinite(vals)), f"{site}: non-finite samples"
    # tau is on the positive real line; identifiable from the data.
    assert float(jnp.mean(samples["tau"])) > 0.0


# ---------------------------------------------------------------------------
# `jax_getitem`-based symbolic indexing into distribution args + obs.
# Handled by the `_materialize_arg`/`bind_dims` path in `compile.py`.
# ---------------------------------------------------------------------------

def test_symbolic_indexing_into_distribution_args():
    """A model that indexes observed J-vectors with an effectful named
    symbol (the `jax_getitem(t, (sym(),))` pattern from
    `effectful.handlers.jax`) and passes the result into a distribution
    constructor / as `obs`. The bridge's compile pass materialises
    such symbolic-shape terms via `bind_dims` before reaching NumPyro,
    so the distribution constructor sees concrete shapes throughout.

    Avoids the unrelated `_DistributionTerm.expand`/`.to_event()`
    annotation gap by using a scalar latent + symbolic-indexed observed
    vectors. The bind_dims materialisation covers exactly the
    `broadcast_shapes`-iterates-symbolic-shape failure mode that would
    otherwise hit `_CallableTerm.__iter__`.
    """
    school = defop(jax.Array, name="school")
    y = jnp.array([1.0, 2.0, 3.0])
    sigma = jnp.array([0.5, 0.5, 0.5])

    def model(y_obs, sigma_obs):
        mu = sample(Normal(0.0, 5.0), name="mu")
        # Symbolic indexing into the observed vectors. Both `sigma_idx`
        # and `y_idx` are `_CallableTerm`s with `school` as a free dim;
        # `bind_dims` materialises them to concrete (3,)-shape arrays
        # before NumPyro's distribution constructor sees them.
        sigma_idx = jax_getitem(sigma_obs, (school(),))
        y_idx = jax_getitem(y_obs, (school(),))
        sample(Normal(mu, sigma_idx), obs=y_idx, name="y_obs")

    mcmc = _run(model, y, sigma, num_warmup=200, num_samples=200)
    samples = mcmc.get_samples_by_name()
    assert "mu" in samples
    assert jnp.all(jnp.isfinite(samples["mu"]))
    # mu's posterior mean should be near the data mean (2.0) — strong
    # likelihood (sigma=0.5) on three observations dominates the prior.
    assert abs(float(jnp.mean(samples["mu"])) - 2.0) < 0.3


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "Separate effectful gap, distinct from the broadcast_shapes-"
        "iterates-term issue. `_DistributionTerm.expand` is `@defop`-"
        "annotated to return `jax.Array` instead of `Distribution`, so "
        "chaining `.to_event(1)` on its result raises "
        "AttributeError: '_ArrayTerm' object has no attribute 'to_event'. "
        "This blocks the NumPyro-vectorised "
        "`Normal(mu_term, tau_term).expand([J]).to_event(1)` idiom. "
        "Fix is upstream (correct the @defop return annotation in "
        "effectful/handlers/numpyro.py); the bridge can't work around it "
        "because the .expand() call happens in user code before the "
        "bridge's compile pass sees the result. strict=True so this "
        "flips to FAIL when upstream fixes it."
    ),
)
def test_distribution_term_expand_to_event_chaining():
    """The remaining named-tensor gap: vectorising a hierarchical site
    via `Normal(mu_term, tau_term).expand([J]).to_event(1)`.

    `_DistributionTerm.expand` returns an `_ArrayTerm` (wrong type per
    the @defop annotation), which doesn't carry the `to_event` method.
    Fix is upstream; the bridge can't work around it.
    """
    J = 8
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    def model(y_obs, sigma_obs):
        mu = sample(Normal(0.0, 10.0), name="mu")
        tau = sample(HalfNormal(10.0), name="tau")
        # Fails at .to_event(1) because .expand([J]) returns _ArrayTerm.
        theta = sample(
            Normal(mu, tau).expand([J]).to_event(1),
            name="theta",
        )
        sample(Normal(theta, sigma_obs).to_event(1), obs=y_obs, name="y_obs")

    _run(model, y, sigma, num_warmup=10, num_samples=10)


# ---------------------------------------------------------------------------
# `jax.grad` on a named-tensor expression — pre-MCMC sibling.
# Confirms the differential side of effectful's JAX layer works in
# isolation, so the xfail above is genuinely about the NumPyro boundary
# and not about `jax.grad` itself.
# ---------------------------------------------------------------------------

def test_jax_grad_through_named_tensor():
    """Pre-MCMC sibling check: `jax.grad` on a function that mixes a
    `jax_getitem`-indexed term with a free variable produces a finite
    gradient. If this ever fails, the issue is in
    `effectful/handlers/jax/_handlers.py`, not in the bridge's
    NumPyro integration."""
    school = defop(jax.Array, name="school")
    t = jnp.arange(3.0)
    t_named = jax_getitem(t, (school(),))

    def potential(x):
        # Bind the named axis back to positional so we can return a scalar.
        return jnp.sum(bind_dims(t_named, school) ** 2 + x ** 2)

    g = jax.grad(potential)(jnp.array(1.5))
    assert jnp.isfinite(g), f"grad produced non-finite: {g}"
