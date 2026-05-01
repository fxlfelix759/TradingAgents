"""Black-Scholes Greeks computation for options evaluation."""

from __future__ import annotations

import math
import logging

import yfinance as yf
from scipy.stats import norm

logger = logging.getLogger(__name__)

_FALLBACK_RATE = 0.05  # fallback when ^IRX is unavailable


def get_risk_free_rate() -> float:
    """Fetch the 3-month T-bill rate from ^IRX; fall back to 5% on error."""
    try:
        ticker = yf.Ticker("^IRX")
        rate_pct = ticker.fast_info.last_price
        if rate_pct and rate_pct > 0:
            return float(rate_pct) / 100.0
    except Exception as exc:
        logger.warning("Could not fetch ^IRX risk-free rate (%s); using %.0f%%", exc, _FALLBACK_RATE * 100)
    return _FALLBACK_RATE


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> dict:
    """Compute Black-Scholes Greeks for a single option contract.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiry in years (use at least 0.001 to avoid division by zero)
        r: Risk-free rate as a decimal (e.g. 0.05 for 5%)
        sigma: Implied volatility as a decimal (e.g. 0.30 for 30%)
        option_type: "call" or "put"

    Returns:
        dict with keys: delta, gamma, theta (daily), vega (per 1% IV move)
    """
    T = max(T, 1e-6)
    sigma = max(sigma, 1e-6)

    otype = option_type.lower()
    if otype not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% change in IV

    if otype == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:  # otype == "put"
        delta = norm.cdf(d1) - 1
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }
