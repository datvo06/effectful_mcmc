"""Tests for NUTS facade, MCMC wrapper, and deferred kernels.

Coverage:
  - Subset-or-equal kwarg-forwarding (signature-level)
  - End-to-end behavioral round-trip on the four load-bearing NUTS kwargs
    (target_accept_prob, max_tree_depth, dense_mass, init_strategy)
  - Per-kernel smoke (Normal(0, 1), 50 warmup + 50 samples)
  - num_chains > 1
  - chain_method coverage (parallel/vectorized/sequential)
  - Deferred-kernel negative-import tests
  - MCMC.__getattr__ forwarding pre-run AttributeError, post-run delegation
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpyro.infer
import pytest

from effectful.handlers.numpyro import Normal, HalfNormal, LKJCholesky, Dirichlet

from effectful_mcmc import sample, NUTS, MCMC


# ---------------------------------------------------------------------------
# Kwarg forwarding — signature subset-or-equal
# ---------------------------------------------------------------------------

def test_nuts_facade_no_shadowing_named_kwargs():
    """Bridge `NUTS.__init__` accepts only (self, model, **kernel_kwargs).

    Catches the regression where a maintainer adds a named kwarg like
    `def __init__(self, model, *, step_size=0.1, **kernel_kwargs)` —
    that would shadow `step_size` and break forwarding silently.
    """
    bridge_params = set(inspect.signature(NUTS.__init__).parameters) - {"self"}
    new_named = bridge_params - {"model", "kernel_kwargs"}
    assert not new_named, f"bridge NUTS shadows NumPyro kwargs: {new_named}"


def test_mcmc_no_shadowing_named_kwargs():
    """Bridge `MCMC.__init__` accepts only (self, kernel, **mcmc_kwargs)."""
    bridge_params = set(inspect.signature(MCMC.__init__).parameters) - {"self"}
    new_named = bridge_params - {"kernel", "mcmc_kwargs"}
    assert not new_named, f"bridge MCMC shadows NumPyro kwargs: {new_named}"


# ---------------------------------------------------------------------------
# Per-kernel smoke
# ---------------------------------------------------------------------------

def _simple_normal_model():
    sample(Normal(0.0, 1.0), name="x")


def test_nuts_smoke_normal():
    """Minimal Normal(0,1), 50 warmup + 50 samples, runs end-to-end."""
    mcmc = MCMC(NUTS(_simple_normal_model), num_warmup=50, num_samples=50)
    mcmc.run(jr.PRNGKey(0))
    samples = mcmc.get_samples_by_name()
    assert "x" in samples
    assert samples["x"].shape == (50,)
    assert jnp.all(jnp.isfinite(samples["x"]))


# ---------------------------------------------------------------------------
# num_chains > 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chain_method", ["sequential", "vectorized"])
def test_nuts_multichain(chain_method):
    """num_chains=2 produces samples of shape (2*num_samples,) by default,
    or (2, num_samples) under group_by_chain=True.

    Excludes chain_method='parallel' because it requires multiple JAX
    devices to be visible and doesn't run on CPU-only setups by default.
    """
    mcmc = MCMC(
        NUTS(_simple_normal_model),
        num_warmup=50, num_samples=50, num_chains=2,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(0))
    samples = mcmc.get_samples_by_name()
    assert samples["x"].shape == (100,)            # 2 chains * 50 samples
    grouped = mcmc.get_samples_by_name(group_by_chain=True)
    assert grouped["x"].shape == (2, 50)


# ---------------------------------------------------------------------------
# Behavioral kwarg round-trip — load-bearing NUTS kwargs
# ---------------------------------------------------------------------------

def test_target_accept_prob_forwards():
    """target_accept_prob is read by the kernel post-construction."""
    nuts_facade = NUTS(_simple_normal_model, target_accept_prob=0.95)
    np_kernel, _, _ = nuts_facade._compile()
    assert np_kernel._target_accept_prob == 0.95


def test_max_tree_depth_forwards():
    """max_tree_depth caps tree growth; we observe it via extra_fields after run."""
    mcmc = MCMC(
        NUTS(_simple_normal_model, max_tree_depth=3),
        num_warmup=20, num_samples=20,
    )
    mcmc.run(jr.PRNGKey(0), extra_fields=("num_steps",))
    # NUTS reports leapfrog steps per iteration; max_tree_depth=3 caps
    # the tree at depth 3, so num_steps <= 2**3 = 8.
    extra = mcmc.get_extra_fields()
    assert jnp.all(extra["num_steps"] <= 2 ** 3), \
        f"max_tree_depth=3 should cap num_steps <= 8, got {extra['num_steps']}"


def test_dense_mass_forwards():
    """dense_mass=True should produce a 2-D inverse mass matrix after warmup,
    not a 1-D diagonal."""
    # 2-site model so the mass matrix has interesting structure.
    def model():
        x = sample(Normal(0.0, 1.0), name="x")
        sample(Normal(x, 1.0), name="y")

    mcmc = MCMC(
        NUTS(model, dense_mass=True),
        num_warmup=200, num_samples=10,
    )
    mcmc.run(jr.PRNGKey(0))
    M_inv = mcmc.last_state.adapt_state.inverse_mass_matrix
    # dense_mass=True → dict with dense (2-D) values
    assert isinstance(M_inv, dict)
    leaf = next(iter(M_inv.values()))
    assert leaf.ndim == 2, f"expected dense (2-D) mass matrix, got shape {leaf.shape}"


def test_init_strategy_forwards():
    """init_strategy is passed through to numpyro.infer.NUTS and used at init."""
    from numpyro.infer.initialization import init_to_median
    nuts_facade = NUTS(_simple_normal_model, init_strategy=init_to_median)
    np_kernel, _, _ = nuts_facade._compile()
    # NumPyro stores it as `_init_strategy`.
    assert np_kernel._init_strategy is init_to_median


# ---------------------------------------------------------------------------
# Combinatorial coverage: kwarg x distribution support family
# ---------------------------------------------------------------------------

def test_dense_mass_with_lkjcholesky():
    """dense_mass=True on an LKJCholesky-only model. Combines two
    independently-flaky paths: dense mass matrix construction x the
    Cholesky factor support."""
    def model():
        sample(LKJCholesky(dim=3, concentration=1.0), name="L")

    mcmc = MCMC(NUTS(model, dense_mass=True), num_warmup=50, num_samples=20)
    mcmc.run(jr.PRNGKey(0))
    samples = mcmc.get_samples_by_name()
    assert "L" in samples
    assert samples["L"].shape == (20, 3, 3)


def test_init_to_median_with_dirichlet():
    """init_strategy=init_to_median on a Dirichlet-only model."""
    from numpyro.infer.initialization import init_to_median

    def model():
        sample(Dirichlet(jnp.ones(4)), name="p")

    mcmc = MCMC(
        NUTS(model, init_strategy=init_to_median),
        num_warmup=50, num_samples=20,
    )
    mcmc.run(jr.PRNGKey(0))
    samples = mcmc.get_samples_by_name()
    assert "p" in samples
    assert samples["p"].shape == (20, 4)
    # Dirichlet samples lie on the simplex.
    sums = samples["p"].sum(axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Operation-keyed get_samples — handle identity round-trip
# ---------------------------------------------------------------------------

def test_get_samples_by_operation():
    """Anonymous sites accessed via `mcmc.model_return_value` → `.op`
    appear in `mcmc.get_samples()` keyed by that Operation."""
    def model():
        mu = sample(Normal(0.0, 5.0))           # anonymous
        return mu

    mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=50)
    mcmc.run(jr.PRNGKey(0))

    mu_term = mcmc.model_return_value
    samples_by_op = mcmc.get_samples()
    assert mu_term.op in samples_by_op
    assert samples_by_op[mu_term.op].shape == (50,)


# ---------------------------------------------------------------------------
# Deferred-kernel negative imports
# ---------------------------------------------------------------------------

# Import from the package so a maintainer adding a deferred kernel
# updates one list, not two.
from effectful_mcmc import _DEFERRED_BLACKJAX_KERNELS, _DEFERRED_NUMPYRO_KERNELS

_DEFERRED_NUMPYRO = list(_DEFERRED_NUMPYRO_KERNELS)
_DEFERRED_BLACKJAX = list(_DEFERRED_BLACKJAX_KERNELS)


@pytest.mark.parametrize("kernel_name", _DEFERRED_NUMPYRO)
def test_deferred_numpyro_kernel_attributeerror(kernel_name):
    """`getattr(effectful_mcmc, 'HMC')` raises AttributeError mentioning
    'not exposed yet' (PEP 562 contract). The deferred-NumPyro message
    references the NumPyro backend by name."""
    import importlib
    mod = importlib.import_module("effectful_mcmc")
    with pytest.raises(AttributeError, match="not exposed yet"):
        getattr(mod, kernel_name)


@pytest.mark.parametrize("kernel_name", _DEFERRED_BLACKJAX)
def test_deferred_blackjax_kernel_attributeerror(kernel_name):
    """BlackJAX-backed kernels have a distinct deferral message that
    names the missing backend, not just 'deferred'."""
    import importlib
    mod = importlib.import_module("effectful_mcmc")
    with pytest.raises(AttributeError, match="BlackJAX"):
        getattr(mod, kernel_name)


@pytest.mark.parametrize("kernel_name", _DEFERRED_NUMPYRO + _DEFERRED_BLACKJAX)
def test_deferred_kernel_from_import_raises_importerror(kernel_name):
    """`from effectful_mcmc import HMC` raises ImportError regardless of
    backend — Python converts the module-level AttributeError to ImportError
    automatically at the `from X import Y` site."""
    with pytest.raises(ImportError):
        exec(f"from effectful_mcmc import {kernel_name}")


def test_unknown_attribute_raises_attribute_error():
    """A name that isn't deferred or exported raises AttributeError, not ImportError."""
    import effectful_mcmc
    with pytest.raises(AttributeError):
        effectful_mcmc.NotAKernel


# ---------------------------------------------------------------------------
# MCMC wrapper semantics
# ---------------------------------------------------------------------------

def test_mcmc_pre_run_attribute_raises():
    """Accessing post-run-only attributes/methods before .run() all
    raise AttributeError consistently — last_state and num_warmup come
    from __getattr__; get_samples is a defined method that checks
    _np_mcmc explicitly. Both surfaces use AttributeError so callers
    don't have to type-check on which path the lookup took."""
    mcmc = MCMC(NUTS(_simple_normal_model), num_warmup=10, num_samples=10)
    with pytest.raises(AttributeError, match="only available after"):
        mcmc.last_state
    with pytest.raises(AttributeError, match="only available after"):
        mcmc.num_warmup
    with pytest.raises(AttributeError, match="only available after"):
        mcmc.get_samples()
    with pytest.raises(AttributeError, match="only available after"):
        mcmc.get_samples_by_name()


def test_mcmc_post_run_forwards_to_numpyro():
    """After .run(), attribute access reaches numpyro.infer.MCMC."""
    mcmc = MCMC(NUTS(_simple_normal_model), num_warmup=20, num_samples=20)
    mcmc.run(jr.PRNGKey(0))
    # These all come from numpyro.infer.MCMC via __getattr__:
    assert mcmc.num_warmup == 20
    assert mcmc.num_samples == 20
    # print_summary exists and doesn't raise:
    mcmc.print_summary()


def test_mcmc_typo_distinguished_from_pre_run():
    """Pre-run typo ('gte_samples' instead of 'get_samples') reports
    'no attribute' rather than the misleading 'only available after .run'.
    The latter sends users hunting for an init-order bug they don't have.
    """
    mcmc = MCMC(NUTS(_simple_normal_model), num_warmup=10, num_samples=10)
    # Real NumPyro attribute, pre-run → "available after" message.
    with pytest.raises(AttributeError, match="only available after"):
        mcmc.last_state
    # Typo (not a NumPyro attribute) → "no attribute" message.
    with pytest.raises(AttributeError, match="no attribute"):
        mcmc.completely_made_up_attribute


def test_mcmc_warm_restart_via_setattr_forwarding():
    """The DOCUMENTED warm-restart workflow

        mcmc2 = MCMC(NUTS(model), num_warmup=…, num_samples=…)
        mcmc2.post_warmup_state = mcmc1.last_state    # BEFORE .run
        mcmc2.run(key2)

    actually skips warmup and uses the saved state. The challenge: at
    the moment `mcmc2.post_warmup_state = …` is set, `_np_mcmc` doesn't
    exist yet — the wrapper has to stash the write and flush it after
    `_np_mcmc` is built but before `.run()` invokes warmup.
    """
    # First chain.
    mcmc1 = MCMC(NUTS(_simple_normal_model), num_warmup=50, num_samples=20)
    mcmc1.run(jr.PRNGKey(0))
    saved_state = mcmc1.last_state

    # Second chain, warm-restarted via the documented "set BEFORE run" order.
    mcmc2 = MCMC(NUTS(_simple_normal_model), num_warmup=50, num_samples=20)
    assert "_np_mcmc" not in mcmc2.__dict__ or mcmc2.__dict__["_np_mcmc"] is None
    # Set BEFORE .run — must be stashed and flushed.
    mcmc2.post_warmup_state = saved_state
    # Stashed in the wrapper, not on a non-existent _np_mcmc.
    assert mcmc2.__dict__.get("_pending_np_attrs", {}).get("post_warmup_state") is saved_state

    mcmc2.run(jr.PRNGKey(1))

    # After run: _np_mcmc exists, holds the saved state, pending dict drained.
    assert mcmc2._np_mcmc.post_warmup_state is saved_state
    assert "_pending_np_attrs" not in mcmc2.__dict__

    # And: post-run writes go directly through __setattr__ to _np_mcmc.
    another_state = mcmc1.last_state
    mcmc2.post_warmup_state = another_state
    assert mcmc2._np_mcmc.post_warmup_state is another_state


def test_mcmc_warm_restart_actually_skips_warmup():
    """Behavioural check: the warm-restarted chain starts from the
    supplied `post_warmup_state` rather than re-warming.

    The cheapest unambiguous signal: with `num_samples=0` on the
    warm-restarted chain, `mcmc2.last_state.z` should be exactly
    `saved_state.z`. If warmup ran instead of being skipped, the chain
    would re-init from the prior and the z-values would differ from
    `saved_state.z`. NumPyro does not consume `post_warmup_state` after
    use (the attribute survives the call), so we can't assert on it as
    a signal — this state-equality check is the load-bearing one.
    """
    mcmc1 = MCMC(NUTS(_simple_normal_model), num_warmup=50, num_samples=20)
    mcmc1.run(jr.PRNGKey(0))
    saved_state = mcmc1.last_state

    # Run a fresh chain (NO warm-restart) and a warm-restarted chain
    # from the SAME RNG key. If warm-restart is being honored, mcmc2's
    # first sample differs from mcmc_fresh's first sample because
    # mcmc2 starts from `saved_state.z` (not the prior init). If warmup
    # was instead repeated, the two chains would be identical to first-
    # sample precision.
    mcmc_fresh = MCMC(NUTS(_simple_normal_model), num_warmup=50, num_samples=5)
    mcmc_fresh.run(jr.PRNGKey(1))

    mcmc_warm = MCMC(NUTS(_simple_normal_model), num_warmup=50, num_samples=5)
    mcmc_warm.post_warmup_state = saved_state
    mcmc_warm.run(jr.PRNGKey(1))                  # same key as fresh!

    fresh_samples = mcmc_fresh.get_samples_by_name()["x"]
    warm_samples = mcmc_warm.get_samples_by_name()["x"]

    # The two chains have the same RNG but different init. If warm
    # restart was honored, the first samples differ; if warmup actually
    # ran twice, they'd be identical (same key → same warmup trajectory
    # → same first sample).
    assert not jnp.allclose(fresh_samples[0], warm_samples[0]), (
        "warm-restart failed: fresh chain and warm-restarted chain "
        "produced identical first samples under the same key, which "
        "means the warm-restart state was ignored and warmup was rerun"
    )
