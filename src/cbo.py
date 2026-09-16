"""CBO computations on a discretized grid. Log-space throughout for stability."""

from __future__ import annotations

import torch
from torch import Tensor

_LOG_NEG_INF: float = -1e30


def log_z_max(ell: Tensor, log_pi: Tensor) -> Tensor:
    """log Z_max = sup_theta { -ell(theta) + log_pi(theta) }."""
    return (-ell + log_pi).max()


def poss_posterior(ell: Tensor, log_pi: Tensor) -> Tensor:
    """Log possibilistic posterior normalized so sup_theta log g*(theta) = 0."""
    return -ell + log_pi - log_z_max(ell, log_pi)


def lcbo(log_g: Tensor, ell: Tensor, log_pi: Tensor) -> Tensor:
    """LCBO(g) = inf_theta { -ell(theta) - log_g(theta) + log_pi(theta) } <= log_z_max."""
    return (-ell - log_g + log_pi).min()


def ucbo(log_g: Tensor, ell: Tensor, log_pi: Tensor) -> Tensor:
    """UCBO(g) = sup_theta { -ell(theta) - log_g(theta) + log_pi(theta) } >= log_z_max."""
    return (-ell - log_g + log_pi).max()


def d_max(log_g: Tensor, log_f: Tensor) -> Tensor:
    """D_max(g || f) = sup_{f > 0} { log g - log f }. Equals 0 iff g <= f (with equal sups)."""
    in_support = log_f > _LOG_NEG_INF
    if not in_support.any():
        return torch.tensor(float("inf"))
    return (log_g[in_support] - log_f[in_support]).max()


def sandwich_check(log_g: Tensor, ell: Tensor, log_pi: Tensor) -> dict:
    """Sandwich identities: log Z_max = LCBO(g) + D_max(g||g*) = UCBO(g) - D_max(g*||g)."""
    _tol = 1e-4
    lzm = log_z_max(ell, log_pi)
    log_g_star = poss_posterior(ell, log_pi)
    lcbo_val = lcbo(log_g, ell, log_pi)
    ucbo_val = ucbo(log_g, ell, log_pi)
    dm_lower = d_max(log_g, log_g_star)
    dm_upper = d_max(log_g_star, log_g)
    return {
        "lcbo": lcbo_val,
        "ucbo": ucbo_val,
        "log_z_max": lzm,
        "d_max_lower": dm_lower,
        "d_max_upper": dm_upper,
        "sandwich_lower_ok": bool((lcbo_val + dm_lower - lzm).abs().item() < _tol),
        "sandwich_upper_ok": bool((ucbo_val - dm_upper - lzm).abs().item() < _tol),
    }