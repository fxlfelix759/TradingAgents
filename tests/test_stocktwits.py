import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestStripExchangeSuffix:
    def test_strips_toronto_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("CNR.TO") == "CNR"

    def test_strips_london_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("SHEL.L") == "SHEL"

    def test_strips_hongkong_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("0700.HK") == "0700"

    def test_strips_tokyo_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("7203.T") == "7203"

    def test_plain_ticker_unchanged(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("AAPL") == "AAPL"

    def test_uppercase(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("aapl") == "AAPL"


@pytest.mark.unit
class TestGetStocktwitsMessages:
    def _mock_response(self, messages, status=200):
        mock = MagicMock()
        mock.status_code = status
        mock.json.return_value = {"response": {"status": status}, "messages": messages}
        return mock

    def test_returns_formatted_string_with_sentiment_summary(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        messages = [
            {
                "id": 1,
                "body": "Earnings beat!",
                "created_at": "2026-05-02T14:32:00Z",
                "user": {"username": "bull_trader"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
            {
                "id": 2,
                "body": "Overvalued imo",
                "created_at": "2026-05-02T13:00:00Z",
                "user": {"username": "bear_trader"},
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
            {
                "id": 3,
                "body": "Watching closely",
                "created_at": "2026-05-02T12:00:00Z",
                "user": {"username": "neutral_user"},
                "entities": {},
            },
        ]
        with patch("requests.get", return_value=self._mock_response(messages)):
            result = get_stocktwits_messages("AAPL", "2026-05-02")
        assert "Bullish: 1" in result
        assert "Bearish: 1" in result
        assert "Unlabelled: 1" in result
        assert "bull_trader" in result
        assert "Earnings beat!" in result

    def test_strips_exchange_suffix_before_api_call(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        with patch("requests.get", return_value=self._mock_response([])) as mock_get:
            get_stocktwits_messages("CNR.TO", "2026-05-02")
        called_url = mock_get.call_args[0][0]
        assert "CNR.TO" not in called_url
        assert "CNR" in called_url

    def test_empty_messages_returns_no_activity_string(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        with patch("requests.get", return_value=self._mock_response([])):
            result = get_stocktwits_messages("AAPL", "2026-05-02")
        assert "No Stocktwits messages found" in result

    def test_api_error_returns_error_string(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        with patch("requests.get", side_effect=Exception("connection refused")):
            result = get_stocktwits_messages("AAPL", "2026-05-02")
        assert "Error fetching Stocktwits" in result

    def test_non_200_status_returns_error_string(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        mock = MagicMock()
        mock.status_code = 404
        mock.json.return_value = {"response": {"status": 404}, "error": ["Not found"]}
        with patch("requests.get", return_value=mock):
            result = get_stocktwits_messages("ZZZZ", "2026-05-02")
        assert "Error fetching Stocktwits" in result
