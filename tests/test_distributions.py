"""Distribution coverage tests.

Two layers:

1. **Support-family tests** (9 tests, one per support family) with
   explicit posterior-mean assertions bounded by ±4·MCSE, where MCSE
   is computed as `std/sqrt(ESS)` via `numpyro.diagnostics.summary`.
   Use a pinned `jr.PRNGKey(0)` so reruns are deterministic.

2. **Full-distribution parametric smoke** over every distribution
   registered in `effectful.handlers.numpyro` (~40 entries). For each
   non-discrete distribution, run 50 warmup + 50 samples and assert
   the chain produces finite values. Discrete distributions are
   skipped explicitly: discrete RVs have no real-line bijection and
   can't be HMC-targeted without enumeration.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.diagnostics import summary

from effectful.handlers.numpyro import (
    Beta,
    CategoricalProbs,
    Dirichlet,
    HalfNormal,
    Independent,
    LKJCholesky,
    Normal,
)

from effectful_mcmc import sample, MCMC, NUTS
from effectful_mcmc._dist_introspection import (
    DISCRETE_CONSTRAINT_NAMES,
    default_factory,
    list_registered_distributions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcse(samples: jnp.ndarray) -> float:
    """Monte-Carlo standard error of the posterior mean.

    MCSE = std / sqrt(ESS), using NumPyro's autocorrelation-corrected
    ESS. The naive `std / sqrt(N)` is ~3× too tight on autocorrelated
    chains and would produce spurious failures.
    """
    # numpyro.diagnostics.summary expects (num_chains, num_samples, ...)
    s = jnp.asarray(samples)
    if s.ndim == 1:
        s = s[None, :]
    stats = summary({"v": s}, prob=0.9)
    std   = float(jnp.atleast_1d(jnp.asarray(stats["v"]["std"])).mean())
    n_eff = float(jnp.atleast_1d(jnp.asarray(stats["v"]["n_eff"])).mean())
    return std / math.sqrt(max(n_eff, 1.0))


def _run(model, num_warmup: int = 200, num_samples: int = 500, key=0):
    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(jr.PRNGKey(key))
    return mcmc


# ---------------------------------------------------------------------------
# Support-family tests (one per family)
# ---------------------------------------------------------------------------

def test_real_support_normal():
    """real-line: Normal(0, 1) posterior mean within 4·MCSE of 0."""
    def model():
        sample(Normal(0.0, 1.0), name="x")
    samples = _run(model).get_samples_by_name()["x"]
    mean = float(jnp.mean(samples))
    assert abs(mean - 0.0) < 4 * _mcse(samples), f"mean={mean}"


def test_positive_support_halfnormal():
    """positive: HalfNormal(1) posterior mean within 4·MCSE of √(2/π)."""
    def model():
        sample(HalfNormal(1.0), name="x")
    samples = _run(model).get_samples_by_name()["x"]
    expected = math.sqrt(2.0 / math.pi)
    mean = float(jnp.mean(samples))
    assert abs(mean - expected) < 4 * _mcse(samples), f"mean={mean}, exp={expected}"


def test_unit_interval_beta():
    """(0,1): Beta(2, 2) posterior mean within 4·MCSE of 0.5."""
    def model():
        sample(Beta(2.0, 2.0), name="x")
    samples = _run(model).get_samples_by_name()["x"]
    mean = float(jnp.mean(samples))
    assert abs(mean - 0.5) < 4 * _mcse(samples), f"mean={mean}"


def test_simplex_dirichlet():
    """simplex: Dirichlet(ones(3)) posterior mean within 4·MCSE of 1/3 per component."""
    def model():
        sample(Dirichlet(jnp.ones(3)), name="p")
    samples = _run(model).get_samples_by_name()["p"]   # shape (500, 3)
    mean = jnp.mean(samples, axis=0)
    for i in range(3):
        mcse_i = _mcse(samples[:, i])
        assert abs(float(mean[i]) - 1.0 / 3.0) < 4 * mcse_i, \
            f"component {i}: mean={mean[i]}"


def test_ordered_vector():
    """ordered_vector via TransformedDistribution + OrderedTransform.

    Asserts (a) samples are ordered along the inner axis (transform
    invariant) and (b) the chain runs to completion. The posterior
    mean isn't analytically clean (it's a transform of a Normal), so
    no per-element mean assertion."""
    from numpyro.distributions.transforms import OrderedTransform

    def model():
        sample(
            dist.TransformedDistribution(
                dist.Normal(0.0, 1.0).expand([3]).to_event(1),
                OrderedTransform(),
            ),
            name="o",
        )
    samples = _run(model).get_samples_by_name()["o"]   # shape (500, 3)
    diffs = jnp.diff(samples, axis=-1)
    assert jnp.all(diffs > 0), "OrderedTransform should produce increasing components"


def test_lower_cholesky_lkj():
    """lower_cholesky: LKJCholesky(3, 1) samples are lower-triangular
    Cholesky factors of correlation matrices.

    Invariant: diag(L) > 0 and rows have unit norm (since L @ L.T is a
    correlation matrix with unit diagonal)."""
    def model():
        sample(LKJCholesky(dim=3, concentration=1.0), name="L")
    samples = _run(model, num_samples=200).get_samples_by_name()["L"]
    # shape (200, 3, 3)
    diag = jnp.diagonal(samples, axis1=-2, axis2=-1)
    assert jnp.all(diag > 0), "Cholesky factor diagonal must be positive"
    row_norms = jnp.sqrt(jnp.sum(samples ** 2, axis=-1))
    assert jnp.allclose(row_norms, 1.0, atol=1e-5), "rows must have unit norm"


def test_event_wrapped_independent():
    """event-wrapped: Independent(Normal(zeros(2), ones(2)), 1) exercises
    the `_DistributionTerm._is_eager`-too-conservative path that the
    compile pass's `_materialize_distribution` works around.
    Posterior mean per component within 4·MCSE of 0."""
    def model():
        sample(Independent(Normal(jnp.zeros(2), jnp.ones(2)), 1), name="z")
    samples = _run(model).get_samples_by_name()["z"]   # shape (500, 2)
    mean = jnp.mean(samples, axis=0)
    for i in range(2):
        mcse_i = _mcse(samples[:, i])
        assert abs(float(mean[i]) - 0.0) < 4 * mcse_i, \
            f"component {i}: mean={mean[i]}"


def test_transformed_affine():
    """transformed (non-ordered): TransformedDistribution(Normal(0,1),
    AffineTransform(loc=2, scale=0.5)) — Y = 2 + 0.5·X with X~N(0,1),
    so E[Y]=2, Var(Y)=0.25."""
    from numpyro.distributions.transforms import AffineTransform

    def model():
        sample(
            dist.TransformedDistribution(
                dist.Normal(0.0, 1.0),
                AffineTransform(loc=2.0, scale=0.5),
            ),
            name="y",
        )
    samples = _run(model).get_samples_by_name()["y"]
    mean = float(jnp.mean(samples))
    assert abs(mean - 2.0) < 4 * _mcse(samples), f"mean={mean}"


def test_discrete_categorical_expected_skip():
    """discrete: Categorical lives at `_IntegerInterval` support, which
    has no real-line bijection — NUTS rejects it at the init step.

    This test pins the expected failure mode so future maintainers
    know discrete RVs are intentionally not bridge-supported (use
    NumPyro's enumeration / DiscreteHMCGibbs / MixedHMC instead,
    none of which are exposed by the bridge yet)."""
    def model():
        sample(CategoricalProbs(jnp.array([0.2, 0.3, 0.5])), name="c")
    # NumPyro's NUTS rejects unenumerated discrete sites with a specific
    # RuntimeError message naming "enumerate support". Asserting the
    # message protects against a future change that silently masks the
    # discrete rejection behind a different error path.
    with pytest.raises(RuntimeError, match="enumerate support"):
        _run(model, num_warmup=10, num_samples=10)


# ---------------------------------------------------------------------------
# Full-distribution parametric smoke (over the ~40 registered distributions)
# ---------------------------------------------------------------------------

# Distributions excluded from the parametric smoke at *collection* time
# (in-body `pytest.skip` is an xfail anti-pattern; exclude from the
# parametrize call instead). Each entry needs a one-line rationale.
PARAMETRIC_SMOKE_EXCLUSIONS: dict[str, str] = {
    # Delta(v) is a point mass — zero variance, zero entropy. NUTS can't
    # initialise because the gradient of -log p(theta) is undefined
    # (the density is a Dirac delta, not a smooth function).
    "Delta": "point mass — undefined gradient for NUTS init",
    # VonMises is a circular distribution; NUTS handles it but emits
    # a UserWarning recommending CircularReparam. Without explicit
    # reparameterisation the chain still runs but mixes poorly.
    "VonMises": "circular site — needs CircularReparam to mix well",
}

# Enforced contract: every exclusion has a non-empty rationale so a
# future maintainer can't quietly add a name with no reason.
assert all(rationale.strip() for rationale in PARAMETRIC_SMOKE_EXCLUSIONS.values()), \
    "every PARAMETRIC_SMOKE_EXCLUSIONS entry needs a one-line rationale"


def _is_discrete_distribution(d) -> bool:
    """A distribution is discrete iff its support's class name is in
    DISCRETE_CONSTRAINT_NAMES. Determined post-construction."""
    try:
        support = d.support
    except (NotImplementedError, AttributeError):
        return False
    return type(support).__name__ in DISCRETE_CONSTRAINT_NAMES


def _smoke_candidates() -> list:
    """Distributions to run through the parametric smoke. Excludes
    structural non-starters (see PARAMETRIC_SMOKE_EXCLUSIONS) at
    collection time so they don't appear in test IDs at all."""
    return [c for c in list_registered_distributions()
            if c.__name__ not in PARAMETRIC_SMOKE_EXCLUSIONS]


@pytest.mark.parametrize(
    "d_cls",
    _smoke_candidates(),
    ids=lambda c: c.__name__,
)
def test_full_distribution_smoke(d_cls):
    """For each registered distribution (modulo PARAMETRIC_SMOKE_EXCLUSIONS),
    run a tiny MCMC (50/50) on a 1-site model. Discrete distributions
    skip cleanly at the support check. Assert the chain completes and
    produces finite values.

    Certifies the 'NUTS works on all effectful distributions' claim
    across the entire ~40-distribution surface, not just the 9
    support-family representatives above.
    """
    d = default_factory(d_cls)()
    if _is_discrete_distribution(d):
        pytest.skip(f"{d_cls.__name__}: discrete support, no real-line bijection")

    def model():
        sample(default_factory(d_cls)(), name="x")

    mcmc = _run(model, num_warmup=50, num_samples=50)
    samples = mcmc.get_samples_by_name()["x"]
    assert jnp.all(jnp.isfinite(samples)), \
        f"{d_cls.__name__}: non-finite samples produced"
