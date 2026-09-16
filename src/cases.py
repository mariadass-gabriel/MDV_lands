"""Tractable PVI test cases with known possibilistic posteriors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .cbo import poss_posterior


@dataclass
class PVICase:
    """Container for a PVI experiment with a known exact posterior."""

    name: str
    theta_grid: Tensor
    ell: Tensor
    log_pi: Tensor
    log_g_star: Tensor


def make_bernoulli_beta(
    n_obs: int = 20,
    p_true: float = 0.7,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    N: int = 500,
    seed: int = 42) -> PVICase:
    """Bernoulli likelihood with Beta prior. Posterior: Beta(alpha+k, beta+(n-k))."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    obs = torch.bernoulli(torch.full((n_obs,), p_true), generator=rng)
    k = int(obs.sum().item())

    EPS = 1e-6
    theta_grid = torch.linspace(EPS, 1.0 - EPS, N)
    ell = -(k * theta_grid.log() + (n_obs - k) * (1 - theta_grid).log())
    log_pi = (alpha_prior - 1) * theta_grid.log() + (beta_prior - 1) * (1 - theta_grid).log()
    return PVICase(
        name="bernoulli_beta",
        theta_grid=theta_grid,
        ell=ell,
        log_pi=log_pi,
        log_g_star=poss_posterior(ell, log_pi))


def make_gaussian_known_var(
    n_obs: int = 30,
    mu_true: float = 1.5,
    sigma_obs: float = 1.0,
    mu_prior: float = 0.0,
    sigma_prior: float = 2.0,
    N: int = 500,
    seed: int = 42) -> PVICase:
    """Gaussian likelihood with known variance and Gaussian prior. Posterior is Gaussian."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    obs = mu_true + sigma_obs * torch.randn(n_obs, generator=rng)
    x_bar = obs.mean().item()

    margin = 5.0 * max(sigma_obs / math.sqrt(n_obs), sigma_prior)
    lo = min(mu_true, mu_prior) - margin
    hi = max(mu_true, mu_prior) + margin
    theta_grid = torch.linspace(lo, hi, N)
    ell = (n_obs / (2 * sigma_obs ** 2)) * (theta_grid - x_bar) ** 2
    log_pi = -0.5 * ((theta_grid - mu_prior) / sigma_prior) ** 2
    return PVICase(
        name="gaussian_known_var",
        theta_grid=theta_grid,
        ell=ell,
        log_pi=log_pi,
        log_g_star=poss_posterior(ell, log_pi))


def make_laplace_likelihood(
    n_obs: int = 20,
    mu_true: float = 1.0,
    b_obs: float = 1.0,
    N: int = 500,
    seed: int = 42) -> PVICase:
    """Laplace likelihood with uniform prior. g* is non-smooth at the sample median."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    u = torch.rand(n_obs, generator=rng) - 0.5
    obs = mu_true - b_obs * u.sign() * torch.log(1 - 2 * u.abs())

    median = obs.median().item()
    spread = b_obs * 6.0
    theta_grid = torch.linspace(median - spread, median + spread, N)
    ell = (1.0 / b_obs) * (theta_grid.unsqueeze(1) - obs.unsqueeze(0)).abs().sum(dim=1)
    log_pi = torch.zeros(N)
    return PVICase(
        name="laplace_likelihood",
        theta_grid=theta_grid,
        ell=ell,
        log_pi=log_pi,
        log_g_star=poss_posterior(ell, log_pi))


def make_bimodal(
    n_obs: int = 10,
    N: int = 500,
    seed: int = 42) -> PVICase:
    """Mixture Gaussian likelihood with bimodal g*. Tests non-uniqueness of CBO optimizer."""
    THETA_STAR: float = 2.0
    rng = torch.Generator()
    rng.manual_seed(seed)
    which = torch.bernoulli(torch.full((n_obs,), 0.5), generator=rng)
    obs = torch.where(
        which.bool(),
        THETA_STAR + torch.randn(n_obs, generator=rng),
        -THETA_STAR + torch.randn(n_obs, generator=rng))

    theta_grid = torch.linspace(-5.0, 5.0, N)
    log_p_plus = -0.5 * (obs.unsqueeze(0) - theta_grid.unsqueeze(1)) ** 2
    log_p_minus = -0.5 * (obs.unsqueeze(0) + theta_grid.unsqueeze(1)) ** 2
    log_mix = torch.stack([log_p_plus, log_p_minus], dim=0).logsumexp(dim=0) - math.log(2)
    ell = -log_mix.sum(dim=1)
    log_pi = torch.zeros(N)
    return PVICase(
        name="bimodal",
        theta_grid=theta_grid,
        ell=ell,
        log_pi=log_pi,
        log_g_star=poss_posterior(ell, log_pi))
