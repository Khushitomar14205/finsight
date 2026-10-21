"""The chat must show people answers, not raw tool JSON."""
from src.agent.humanize import humanize

RETURNS = {"symbol": "RELIANCE", "start": "2023-01-02", "end": "2023-12-29", "observations": 245,
           "first_close": 2575.9, "last_close": 2584.95, "total_return": 0.0035133,
           "annualized_return": 0.0036287,
           "best_day": {"date": "2023-03-31", "return": 0.0431154},
           "worst_day": {"date": "2023-07-20", "return": -0.0781181}}

def test_returns_answer_is_plain_english():
    text = humanize("calculate_returns", RETURNS, {"symbols": ["RELIANCE"]})
    assert "+0.35%" in text and "RELIANCE" in text
    assert "{" not in text and "total_return" not in text and "Tool used" not in text

def test_compare_answer_lists_each_symbol():
    result = {"common_start": "2023-01-02", "common_end": "2023-12-29", "missing_observations": {},
              "symbols": {"RELIANCE": {"total_return": 0.0035, "annualized_return": 0.0036,
                                       "annualized_volatility": 0.13, "sharpe_ratio": -0.4, "max_drawdown": -0.11},
                          "TCS": {"total_return": 0.18, "annualized_return": 0.185,
                                  "annualized_volatility": 0.16, "sharpe_ratio": 0.78, "max_drawdown": -0.09}}}
    text = humanize("compare_stocks", result, {})
    assert "RELIANCE" in text and "TCS" in text and "+18.00%" in text
    assert "{" not in text

def test_generic_fallback_never_dumps_json():
    text = humanize("some_future_tool", {"alpha_beta": 0.123456, "count": 7}, {})
    assert "{" not in text and "Alpha beta" in text
