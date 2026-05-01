import math
import pytest
from unittest.mock import patch, MagicMock
from tradingagents.dataflows.options_greeks import compute_greeks, get_risk_free_rate


def test_call_delta_atm():
    """ATM call delta should be close to 0.5."""
    greeks = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert 0.55 < greeks["delta"] < 0.65  # ATM call slightly above 0.5


def test_put_delta_atm():
    """ATM put delta should be negative and close to -0.5."""
    greeks = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert -0.5 < greeks["delta"] < -0.35


def test_call_put_delta_sum():
    """Call delta + |put delta| should be approximately 1 (put-call parity)."""
    call = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    put = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert abs(call["delta"] + abs(put["delta"]) - 1.0) < 0.01


def test_gamma_positive():
    """Gamma is always positive for both calls and puts."""
    for otype in ("call", "put"):
        g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type=otype)
        assert g["gamma"] > 0


def test_call_theta_negative():
    """Theta for calls is always negative."""
    g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type="call")
    assert g["theta"] < 0


def test_put_theta_atm_negative():
    """Theta for ATM puts is negative."""
    g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type="put")
    assert g["theta"] < 0


def test_invalid_option_type_raises():
    """compute_greeks raises ValueError for unknown option_type."""
    with pytest.raises(ValueError, match="option_type must be"):
        compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="long")


def test_vega_positive():
    """Vega is always positive for both calls and puts."""
    for otype in ("call", "put"):
        g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type=otype)
        assert g["vega"] > 0


def test_deep_itm_call_delta_near_one():
    """Deep ITM call delta should be close to 1."""
    greeks = compute_greeks(S=150, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
    assert greeks["delta"] > 0.95


def test_deep_otm_call_delta_near_zero():
    """Deep OTM call delta should be close to 0."""
    greeks = compute_greeks(S=50, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
    assert greeks["delta"] < 0.05


def test_near_expiry_raises_no_error():
    """Very short time to expiry (T=0.001) should not raise."""
    greeks = compute_greeks(S=100, K=100, T=0.001, r=0.05, sigma=0.2, option_type="call")
    assert "delta" in greeks


def test_get_risk_free_rate_mocked():
    """get_risk_free_rate returns a float between 0 and 1."""
    mock_ticker = MagicMock()
    mock_ticker.fast_info.last_price = 5.25  # 5.25% annualized
    with patch("tradingagents.dataflows.options_greeks.yf.Ticker", return_value=mock_ticker):
        r = get_risk_free_rate()
    assert abs(r - 0.0525) < 1e-6


def test_get_risk_free_rate_fallback():
    """get_risk_free_rate returns 0.05 when yfinance raises."""
    with patch("tradingagents.dataflows.options_greeks.yf.Ticker", side_effect=Exception("network")):
        r = get_risk_free_rate()
    assert r == 0.05
