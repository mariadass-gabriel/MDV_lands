"""Possibilistic exponential families for PVI experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


def _normalize_log(log_vals: Tensor) -> Tensor:
    """Shift log values so max = 0."""
    return log_vals - log_vals.max()


def _laplace_sigma(log_g_star: Tensor, theta_grid: Tensor) -> float:
    """Possibilistic scale via curvature at the mode: 1/sqrt(-d2 log_g_star / dtheta2)."""
    idx = max(1, min(int(log_g_star.argmax().item()), len(theta_grid) - 2))
    dtheta = (theta_grid[-1] - theta_grid[0]).item() / (len(theta_grid) - 1)
    curv = (log_g_star[idx + 1] - 2 * log_g_star[idx] + log_g_star[idx - 1]).item() / dtheta ** 2
    return 1.0 / math.sqrt(max(-curv, 1e-6))


@dataclass
class GaussianPoss:
    """Normal possibility: log g = -0.5 * ((theta - mu) / sigma)^2, normalized to max 0."""

    mu: float
    sigma: float

    def log_poss(self, theta_grid: Tensor, params: Dict[str, float] | None = None) -> Tensor:
        mu = params["mu"] if params is not None else self.mu
        sigma = params["sigma"] if params is not None else self.sigma
        return _normalize_log(-0.5 * ((theta_grid - mu) / sigma) ** 2)

    def mode(self, params: Dict[str, float] | None = None) -> float:
        return params["mu"] if params is not None else self.mu

    def fit(self, log_g_star: Tensor, theta_grid: Tensor) -> "GaussianPoss":
        """Possibilistic Laplace approximation: mu = argmax g*, sigma from curvature."""
        mu = theta_grid[log_g_star.argmax().item()].item()
        return GaussianPoss(mu=mu, sigma=_laplace_sigma(log_g_star, theta_grid))


@dataclass
class LaplacePoss:
    """Laplace possibility: log g = -|theta - mu| / b, normalized to max 0."""

    mu: float
    b: float

    def log_poss(self, theta_grid: Tensor, params: Dict[str, float] | None = None) -> Tensor:
        mu = params["mu"] if params is not None else self.mu
        b = params["b"] if params is not None else self.b
        return _normalize_log(-(theta_grid - mu).abs() / b)

    def mode(self, params: Dict[str, float] | None = None) -> float:
        return params["mu"] if params is not None else self.mu

    def fit(self, log_g_star: Tensor, theta_grid: Tensor) -> "LaplacePoss":
        """mu = argmax g*, b from half-width of {log_g_star >= log(0.5)} / (2*log2)."""
        mu = theta_grid[log_g_star.argmax().item()].item()
        above = log_g_star >= math.log(0.5)
        if above.any():
            idxs = torch.where(above)[0]
            width = (theta_grid[idxs[-1]] - theta_grid[idxs[0]]).abs().item()
            b = max(width / (2.0 * math.log(2)), 1e-6)
        else:
            b = 1.0
        return LaplacePoss(mu=mu, b=b)


@dataclass
class BetaPoss:
    """Beta possibility for theta in (0,1): log g = (a-1)log(theta) + (b-1)log(1-theta) + c."""

    alpha: float
    beta: float

    def log_poss(self, theta_grid: Tensor, params: Dict[str, float] | None = None) -> Tensor:
        alpha = params["alpha"] if params is not None else self.alpha
        beta_p = params["beta"] if params is not None else self.beta
        eps = 1e-8
        th = theta_grid.clamp(eps, 1 - eps)
        return _normalize_log((alpha - 1) * th.log() + (beta_p - 1) * (1 - th).log())

    def mode(self, params: Dict[str, float] | None = None) -> float:
        alpha = params["alpha"] if params is not None else self.alpha
        beta_p = params["beta"] if params is not None else self.beta
        if alpha > 1 and beta_p > 1:
            return (alpha - 1) / (alpha + beta_p - 2)
        elif alpha <= 1 and beta_p > 1:
            return 0.0
        elif alpha > 1 and beta_p <= 1:
            return 1.0
        else:
            return 0.5

    def fit(self, log_g_star: Tensor, theta_grid: Tensor) -> "BetaPoss":
        """OLS on log g* = c + (a-1)log(theta) + (b-1)log(1-theta). Exact for conjugate Beta."""
        N = len(theta_grid)
        eps = 1e-8
        th = theta_grid.clamp(eps, 1 - eps)
        X = torch.stack([
            torch.ones(N),
            th.log(),
            (1 - th).log()], dim=1)
        try:
            coeffs = torch.linalg.solve(X.t() @ X, X.t() @ log_g_star)
            alpha = float(coeffs[1].item() + 1)
            beta = float(coeffs[2].item() + 1)
        except Exception:
            idx = int(log_g_star.argmax().item())
            mu = theta_grid[idx].clamp(eps, 1 - eps).item()
            alpha = mu * 10 + 1
            beta = (1 - mu) * 10 + 1
        return BetaPoss(
            alpha=max(1.1, min(50.0, alpha)),
            beta=max(1.1, min(50.0, beta)))