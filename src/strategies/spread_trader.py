"""Spread/pair trading engine adapted from IMC Prosperity Gift Basket strategy.

Core concept: trade the z-score of the spread between correlated assets.
When spread deviates from its mean, bet on reversion.

Adapted for Hyperliquid perpetual futures with pre-defined pair configurations.
"""

from __future__ import annotations
import math
from collections import deque
from src.strategies.quant_signals import rolling_mean, rolling_std, zscore


# ---------------------------------------------------------------------------
# Pre-defined correlated pairs for Hyperliquid
# ---------------------------------------------------------------------------

# Each pair: (asset_a, asset_b, hedge_ratio, spread_mean, description)
# hedge_ratio: how many units of B to trade per unit of A
# spread_mean: long-run estimated mean of the spread (calibrate from data)
DEFAULT_PAIRS = [
    {
        "name": "BTC_ETH",
        "asset_a": "BTC",
        "asset_b": "ETH",
        "hedge_ratio": None,    # Will be computed dynamically
        "spread_mean": None,    # Will be computed from rolling window
        "description": "BTC/ETH correlation pair",
    },
    {
        "name": "BTC_SOL",
        "asset_a": "BTC",
        "asset_b": "SOL",
        "hedge_ratio": None,
        "spread_mean": None,
        "description": "BTC/SOL correlation pair",
    },
]


class SpreadTracker:
    """Tracks spread history and generates pair trading signals.

    Adapted from IMC's Gift Basket z-score strategy:
    - Compute spread = price_A - hedge_ratio * price_B
    - Track rolling z-score of spread
    - Signal entry when z-score exceeds threshold
    - Signal exit when z-score reverts toward zero
    """

    def __init__(
        self,
        pair_name: str,
        asset_a: str,
        asset_b: str,
        spread_window: int = 45,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        max_history: int = 500,
    ):
        self.pair_name = pair_name
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.spread_window = spread_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit

        # Rolling history
        self.spread_history: deque[float] = deque(maxlen=max_history)
        self.price_a_history: deque[float] = deque(maxlen=max_history)
        self.price_b_history: deque[float] = deque(maxlen=max_history)
        self.hedge_ratio: float | None = None

    def _compute_hedge_ratio(self) -> float:
        """Dynamically compute hedge ratio from price history.

        Uses the ratio of means as a simple estimator.
        For more sophisticated: use rolling OLS regression.
        """
        if len(self.price_a_history) < 20 or len(self.price_b_history) < 20:
            return 1.0

        a_prices = list(self.price_a_history)[-50:]
        b_prices = list(self.price_b_history)[-50:]

        if len(a_prices) != len(b_prices):
            min_len = min(len(a_prices), len(b_prices))
            a_prices = a_prices[-min_len:]
            b_prices = b_prices[-min_len:]

        # Simple hedge ratio: mean(A) / mean(B)
        mean_a = sum(a_prices) / len(a_prices)
        mean_b = sum(b_prices) / len(b_prices)

        if mean_b == 0:
            return 1.0

        return mean_a / mean_b

    def _compute_ols_hedge_ratio(self) -> float:
        """Compute hedge ratio via OLS regression: A = beta * B + alpha.

        More robust than simple ratio. beta = Cov(A,B) / Var(B).
        """
        if len(self.price_a_history) < 30 or len(self.price_b_history) < 30:
            return self._compute_hedge_ratio()

        a_prices = list(self.price_a_history)[-100:]
        b_prices = list(self.price_b_history)[-100:]
        min_len = min(len(a_prices), len(b_prices))
        a_prices = a_prices[-min_len:]
        b_prices = b_prices[-min_len:]

        n = len(a_prices)
        mean_a = sum(a_prices) / n
        mean_b = sum(b_prices) / n

        cov = sum((a_prices[i] - mean_a) * (b_prices[i] - mean_b) for i in range(n)) / n
        var_b = sum((b_prices[i] - mean_b) ** 2 for i in range(n)) / n

        if var_b == 0:
            return self._compute_hedge_ratio()

        return cov / var_b

    def update(self, price_a: float, price_b: float):
        """Add new price observations and update spread history."""
        self.price_a_history.append(price_a)
        self.price_b_history.append(price_b)

        # Recompute hedge ratio periodically
        if len(self.price_a_history) >= 30:
            self.hedge_ratio = self._compute_ols_hedge_ratio()
        elif len(self.price_a_history) >= 10:
            self.hedge_ratio = self._compute_hedge_ratio()
        else:
            self.hedge_ratio = price_a / price_b if price_b > 0 else 1.0

        # Compute spread using log-returns for stationarity
        if price_a > 0 and price_b > 0:
            # Log spread is more stationary than raw spread
            log_spread = math.log(price_a) - self.hedge_ratio * math.log(price_b)
            self.spread_history.append(log_spread)

    def get_signal(self) -> dict:
        """Generate spread trading signal.

        Returns:
            {
                "pair": str,
                "spread_zscore": float,
                "signal": "long_spread"/"short_spread"/"hold",
                "strength": float,
                "hedge_ratio": float,
                "entry_triggered": bool,
                "exit_triggered": bool,
                "actions": {
                    "asset_a": {"action": "buy"/"sell"/"hold", "weight": float},
                    "asset_b": {"action": "buy"/"sell"/"hold", "weight": float},
                },
                "correlation": float,
            }
        """
        if len(self.spread_history) < self.spread_window + 1:
            return {
                "pair": self.pair_name,
                "spread_zscore": 0,
                "signal": "hold",
                "strength": 0,
                "hedge_ratio": self.hedge_ratio or 1.0,
                "entry_triggered": False,
                "exit_triggered": False,
                "actions": {
                    self.asset_a: {"action": "hold", "weight": 0},
                    self.asset_b: {"action": "hold", "weight": 0},
                },
                "correlation": 0,
                "data_points": len(self.spread_history),
                "required_points": self.spread_window + 1,
            }

        spreads = list(self.spread_history)
        z_scores = zscore(spreads, self.spread_window)
        current_z = z_scores[-1] if z_scores[-1] is not None else 0

        signal = "hold"
        entry_triggered = False
        exit_triggered = False

        if current_z <= -self.zscore_entry:
            # Spread is below mean → buy spread (buy A, sell B)
            signal = "long_spread"
            entry_triggered = True
        elif current_z >= self.zscore_entry:
            # Spread is above mean → sell spread (sell A, buy B)
            signal = "short_spread"
            entry_triggered = True
        elif abs(current_z) < self.zscore_exit:
            exit_triggered = True

        # Compute correlation
        correlation = self._compute_correlation()

        strength = min(abs(current_z) / (self.zscore_entry * 2), 1.0)

        # Translate spread signal to per-asset actions
        actions = {}
        if signal == "long_spread":
            actions[self.asset_a] = {"action": "buy", "weight": 1.0}
            actions[self.asset_b] = {"action": "sell", "weight": self.hedge_ratio or 1.0}
        elif signal == "short_spread":
            actions[self.asset_a] = {"action": "sell", "weight": 1.0}
            actions[self.asset_b] = {"action": "buy", "weight": self.hedge_ratio or 1.0}
        else:
            actions[self.asset_a] = {"action": "hold", "weight": 0}
            actions[self.asset_b] = {"action": "hold", "weight": 0}

        return {
            "pair": self.pair_name,
            "spread_zscore": round(current_z, 4),
            "signal": signal,
            "strength": round(strength, 4),
            "hedge_ratio": round(self.hedge_ratio, 6) if self.hedge_ratio else 1.0,
            "entry_triggered": entry_triggered,
            "exit_triggered": exit_triggered,
            "actions": actions,
            "correlation": round(correlation, 4),
            "data_points": len(self.spread_history),
        }

    def _compute_correlation(self) -> float:
        """Compute rolling correlation between the two assets."""
        if len(self.price_a_history) < 20 or len(self.price_b_history) < 20:
            return 0

        a = list(self.price_a_history)[-50:]
        b = list(self.price_b_history)[-50:]
        min_len = min(len(a), len(b))
        a = a[-min_len:]
        b = b[-min_len:]

        # Compute returns
        ret_a = [(a[i] - a[i - 1]) / a[i - 1] for i in range(1, len(a)) if a[i - 1] != 0]
        ret_b = [(b[i] - b[i - 1]) / b[i - 1] for i in range(1, len(b)) if b[i - 1] != 0]

        min_ret = min(len(ret_a), len(ret_b))
        if min_ret < 10:
            return 0

        ret_a = ret_a[-min_ret:]
        ret_b = ret_b[-min_ret:]

        mean_a = sum(ret_a) / len(ret_a)
        mean_b = sum(ret_b) / len(ret_b)

        cov = sum((ret_a[i] - mean_a) * (ret_b[i] - mean_b) for i in range(len(ret_a))) / len(ret_a)
        std_a = math.sqrt(sum((r - mean_a) ** 2 for r in ret_a) / len(ret_a))
        std_b = math.sqrt(sum((r - mean_b) ** 2 for r in ret_b) / len(ret_b))

        if std_a == 0 or std_b == 0:
            return 0

        return cov / (std_a * std_b)


class SpreadTradingEngine:
    """Manages multiple spread trackers and produces unified signals.

    Usage:
        engine = SpreadTradingEngine(assets=["BTC", "ETH", "SOL"])
        engine.update_prices({"BTC": 95000, "ETH": 3500, "SOL": 180})
        signals = engine.get_all_signals()
    """

    def __init__(self, assets: list[str], pairs: list[dict] | None = None):
        self.assets = set(assets)
        self.trackers: dict[str, SpreadTracker] = {}

        # Auto-detect valid pairs from configured assets
        pair_configs = pairs or DEFAULT_PAIRS
        for pc in pair_configs:
            a = pc["asset_a"]
            b = pc["asset_b"]
            if a in self.assets and b in self.assets:
                tracker = SpreadTracker(
                    pair_name=pc["name"],
                    asset_a=a,
                    asset_b=b,
                )
                self.trackers[pc["name"]] = tracker

    def update_prices(self, prices: dict[str, float]):
        """Update all trackers with current prices."""
        for name, tracker in self.trackers.items():
            a_price = prices.get(tracker.asset_a)
            b_price = prices.get(tracker.asset_b)
            if a_price and b_price and a_price > 0 and b_price > 0:
                tracker.update(a_price, b_price)

    def get_all_signals(self) -> dict:
        """Get signals from all active spread trackers."""
        signals = {}
        for name, tracker in self.trackers.items():
            signals[name] = tracker.get_signal()
        return signals

    def get_asset_spread_signals(self, asset: str) -> list[dict]:
        """Get all spread signals that involve a specific asset."""
        results = []
        for name, tracker in self.trackers.items():
            if asset in (tracker.asset_a, tracker.asset_b):
                signal = tracker.get_signal()
                results.append(signal)
        return results
