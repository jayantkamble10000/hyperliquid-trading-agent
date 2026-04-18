"""Quantitative signal engine adapted from IMC Prosperity competition strategies.

Computes hard mathematical signals BEFORE the LLM call so that Claude
validates/weights pre-computed signals rather than guessing from raw indicators.

Key techniques adapted:
- Z-score mean reversion (IMC Gift Basket spread strategy)
- Mean reversion with negative beta on lagged returns (IMC Starfruit)
- Bollinger Band %B for overbought/oversold
- Multi-timeframe signal alignment scoring
- Regime-aware signal gating
"""

from __future__ import annotations
import math
from collections import deque
from src.indicators.local_indicators import (
    ema, sma, rsi, macd, atr, bbands, adx, obv, vwap, compute_all, latest, last_n
)


# ---------------------------------------------------------------------------
# Rolling statistics helpers
# ---------------------------------------------------------------------------

def rolling_mean(values: list[float], window: int) -> list[float | None]:
    """Rolling mean over a fixed window."""
    result: list[float | None] = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(None)
        else:
            result.append(sum(values[i - window + 1: i + 1]) / window)
    return result


def rolling_std(values: list[float], window: int) -> list[float | None]:
    """Rolling standard deviation over a fixed window."""
    result: list[float | None] = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(None)
        else:
            w = values[i - window + 1: i + 1]
            mean = sum(w) / window
            var = sum((x - mean) ** 2 for x in w) / window
            result.append(math.sqrt(var) if var > 0 else 0.0)
    return result


def zscore(values: list[float], window: int) -> list[float | None]:
    """Rolling z-score: (value - rolling_mean) / rolling_std."""
    means = rolling_mean(values, window)
    stds = rolling_std(values, window)
    result: list[float | None] = []
    for i in range(len(values)):
        if means[i] is None or stds[i] is None or stds[i] == 0:
            result.append(None)
        else:
            result.append((values[i] - means[i]) / stds[i])
    return result


def bollinger_pct_b(candles: list[dict], period: int = 20, std_dev: float = 2.0) -> list[float | None]:
    """Bollinger %B: (price - lower) / (upper - lower). 0 = at lower band, 1 = at upper."""
    closes = [c["close"] for c in candles]
    bb = bbands(candles, period, std_dev)
    result: list[float | None] = []
    for i in range(len(closes)):
        upper = bb["upper"][i]
        lower = bb["lower"][i]
        if upper is None or lower is None or upper == lower:
            result.append(None)
        else:
            result.append((closes[i] - lower) / (upper - lower))
    return result


# ---------------------------------------------------------------------------
# Mean reversion signal (adapted from IMC Starfruit strategy)
# ---------------------------------------------------------------------------

def mean_reversion_signal(
    candles: list[dict],
    reversion_beta: float = -0.229,
    lookback: int = 1
) -> dict:
    """Compute mean reversion signal based on lagged returns.

    From IMC Prosperity: if price just went up, predict partial reversion.
    A negative beta means contrarian: recent up-move predicts down-move.

    Returns:
        {
            "predicted_return": float,  # Expected next-bar return
            "signal": float,           # -1 to +1 normalized signal
            "fair_value_offset": float, # Offset from current price
            "strength": str,           # "strong"/"moderate"/"weak"
        }
    """
    closes = [c["close"] for c in candles]
    if len(closes) < lookback + 2:
        return {"predicted_return": 0, "signal": 0, "fair_value_offset": 0, "strength": "none"}

    # Compute recent return
    recent_return = (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback]

    # Predicted return = reversion_beta * recent_return
    predicted_return = reversion_beta * recent_return

    # Fair value offset
    fair_value_offset = closes[-1] * predicted_return

    # Normalize signal to -1..+1 range using tanh scaling
    # Scale factor calibrated so a 1% move produces ~0.5 signal
    signal = math.tanh(predicted_return * 50)

    # Strength classification
    abs_signal = abs(signal)
    if abs_signal > 0.6:
        strength = "strong"
    elif abs_signal > 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "predicted_return": round(predicted_return, 6),
        "signal": round(signal, 4),
        "fair_value_offset": round(fair_value_offset, 4),
        "strength": strength,
    }


# ---------------------------------------------------------------------------
# Regime detection (adapted from IMC ADX-based approach)
# ---------------------------------------------------------------------------

def detect_regime(candles: list[dict], adx_period: int = 14) -> dict:
    """Classify market regime using ADX + price structure.

    Returns:
        {
            "regime": "trending_up" | "trending_down" | "ranging" | "volatile",
            "adx_value": float,
            "trend_strength": float,  # 0-1
            "ema_alignment": str,     # "bullish"/"bearish"/"neutral"
        }
    """
    from src.indicators.local_indicators import adx as compute_adx

    if len(candles) < 50:
        return {"regime": "unknown", "adx_value": 0, "trend_strength": 0, "ema_alignment": "neutral"}

    closes = [c["close"] for c in candles]
    adx_series = compute_adx(candles, adx_period)
    adx_val = latest(adx_series)

    ema20 = latest(ema(closes, 20))
    ema50 = latest(ema(closes, 50))

    if adx_val is None or ema20 is None or ema50 is None:
        return {"regime": "unknown", "adx_value": 0, "trend_strength": 0, "ema_alignment": "neutral"}

    # EMA alignment
    if ema20 > ema50:
        ema_alignment = "bullish"
    elif ema20 < ema50:
        ema_alignment = "bearish"
    else:
        ema_alignment = "neutral"

    # Trend strength normalized 0-1 (ADX 0-100 mapped)
    trend_strength = min(adx_val / 50.0, 1.0)

    # ATR-based volatility check
    atr_series = atr(candles, 14)
    atr_val = latest(atr_series)
    atr_pct = (atr_val / closes[-1] * 100) if atr_val and closes[-1] else 0

    # Regime classification
    if adx_val >= 25:
        if ema_alignment == "bullish":
            regime = "trending_up"
        elif ema_alignment == "bearish":
            regime = "trending_down"
        else:
            regime = "trending_up" if closes[-1] > closes[-10] else "trending_down"
    elif atr_pct > 3.0:
        regime = "volatile"
    else:
        regime = "ranging"

    return {
        "regime": regime,
        "adx_value": round(adx_val, 2),
        "trend_strength": round(trend_strength, 4),
        "ema_alignment": ema_alignment,
        "atr_pct": round(atr_pct, 4),
    }


# ---------------------------------------------------------------------------
# Multi-timeframe signal alignment (adapted from IMC cross-timeframe approach)
# ---------------------------------------------------------------------------

def compute_momentum_score(candles: list[dict]) -> dict:
    """Score momentum across multiple indicators.

    Returns a composite score from -1 (strong bearish) to +1 (strong bullish).
    """
    closes = [c["close"] for c in candles]
    if len(closes) < 30:
        return {"score": 0, "components": {}}

    components = {}

    # 1. EMA trend: price vs EMA20
    ema20_val = latest(ema(closes, 20))
    if ema20_val:
        ema_signal = (closes[-1] - ema20_val) / ema20_val
        components["ema_trend"] = round(math.tanh(ema_signal * 30), 4)
    else:
        components["ema_trend"] = 0

    # 2. MACD histogram direction
    macd_data = macd(candles)
    hist = [v for v in macd_data["histogram"] if v is not None]
    if len(hist) >= 2:
        # Histogram slope (accelerating or decelerating)
        hist_slope = hist[-1] - hist[-2]
        components["macd_momentum"] = round(math.tanh(hist_slope * 100), 4)
    else:
        components["macd_momentum"] = 0

    # 3. RSI position (normalized -1 to +1)
    rsi_series = rsi(candles, 14)
    rsi_val = latest(rsi_series)
    if rsi_val is not None:
        # Map 0-100 to -1..+1, with 50 as neutral
        components["rsi_position"] = round((rsi_val - 50) / 50, 4)
    else:
        components["rsi_position"] = 0

    # 4. OBV trend (rising/falling)
    obv_series = obv(candles)
    if len(obv_series) >= 10:
        obv_change = obv_series[-1] - obv_series[-10]
        obv_range = max(abs(obv_series[-1]), 1)
        components["obv_trend"] = round(math.tanh(obv_change / obv_range * 5), 4)
    else:
        components["obv_trend"] = 0

    # 5. Price vs VWAP
    vwap_series = vwap(candles)
    vwap_val = latest(vwap_series)
    if vwap_val and vwap_val > 0:
        vwap_signal = (closes[-1] - vwap_val) / vwap_val
        components["vwap_position"] = round(math.tanh(vwap_signal * 30), 4)
    else:
        components["vwap_position"] = 0

    # Weighted composite
    weights = {
        "ema_trend": 0.25,
        "macd_momentum": 0.25,
        "rsi_position": 0.15,
        "obv_trend": 0.15,
        "vwap_position": 0.20,
    }
    score = sum(components[k] * weights[k] for k in weights)

    return {"score": round(score, 4), "components": components}


def compute_timeframe_alignment(intraday_candles: list[dict], longterm_candles: list[dict]) -> dict:
    """Score alignment between intraday (5m) and long-term (4h) signals.

    From IMC: favor trades where both timeframes agree.
    Alignment > 0.5 = both bullish. < -0.5 = both bearish.
    Near 0 = conflict (prefer hold).
    """
    intra_mom = compute_momentum_score(intraday_candles)
    lt_mom = compute_momentum_score(longterm_candles)

    intra_score = intra_mom["score"]
    lt_score = lt_mom["score"]

    # Alignment = product of scores (positive when same direction)
    alignment = intra_score * lt_score

    # Composite direction weighted toward long-term
    direction = 0.6 * lt_score + 0.4 * intra_score

    if alignment > 0.1:
        alignment_label = "aligned"
    elif alignment < -0.1:
        alignment_label = "conflicting"
    else:
        alignment_label = "neutral"

    return {
        "alignment": round(alignment, 4),
        "alignment_label": alignment_label,
        "direction": round(direction, 4),
        "intraday_score": intra_score,
        "longterm_score": lt_score,
        "intraday_components": intra_mom["components"],
        "longterm_components": lt_mom["components"],
    }


# ---------------------------------------------------------------------------
# Volatility-adjusted position sizing (adapted from IMC ATR approach)
# ---------------------------------------------------------------------------

def compute_position_size(
    candles: list[dict],
    account_value: float,
    risk_per_trade_pct: float = 1.0,
    atr_multiplier: float = 2.0,
    max_position_pct: float = 20.0,
) -> dict:
    """ATR-based position sizing.

    Size = (account * risk_pct) / (ATR * multiplier)
    This gives smaller positions in volatile markets, larger in calm markets.

    Returns:
        {
            "suggested_usd": float,
            "atr_stop_distance": float,  # Distance for ATR-based stop
            "risk_per_unit": float,
        }
    """
    atr_series = atr(candles, 14)
    atr_val = latest(atr_series)
    current_price = candles[-1]["close"] if candles else 0

    if not atr_val or atr_val <= 0 or current_price <= 0 or account_value <= 0:
        return {"suggested_usd": 0, "atr_stop_distance": 0, "risk_per_unit": 0}

    # Risk budget per trade
    risk_budget = account_value * (risk_per_trade_pct / 100.0)

    # Stop distance = ATR * multiplier
    stop_distance = atr_val * atr_multiplier

    # Position size in units = risk_budget / stop_distance
    units = risk_budget / stop_distance

    # Convert to USD
    suggested_usd = units * current_price

    # Cap at max_position_pct
    max_usd = account_value * (max_position_pct / 100.0)
    suggested_usd = min(suggested_usd, max_usd)

    # Floor at Hyperliquid minimum
    suggested_usd = max(suggested_usd, 11.0)

    return {
        "suggested_usd": round(suggested_usd, 2),
        "atr_stop_distance": round(stop_distance, 4),
        "atr_stop_pct": round(stop_distance / current_price * 100, 2),
        "risk_per_unit": round(stop_distance, 4),
    }


# ---------------------------------------------------------------------------
# Z-score based entry/exit signals (adapted from IMC Gift Basket)
# ---------------------------------------------------------------------------

def zscore_signal(
    values: list[float],
    window: int = 45,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> dict:
    """Generate trading signals from z-score of a price series.

    Adapted from IMC's spread trading: enter when z-score exceeds threshold,
    exit when it reverts toward zero.

    Returns:
        {
            "zscore": float,
            "signal": "buy"/"sell"/"hold",
            "strength": float,  # 0-1
            "entry_triggered": bool,
            "exit_triggered": bool,
        }
    """
    if len(values) < window + 1:
        return {"zscore": 0, "signal": "hold", "strength": 0,
                "entry_triggered": False, "exit_triggered": False}

    zscores = zscore(values, window)
    current_z = zscores[-1]
    prev_z = zscores[-2] if len(zscores) >= 2 else None

    if current_z is None:
        return {"zscore": 0, "signal": "hold", "strength": 0,
                "entry_triggered": False, "exit_triggered": False}

    signal = "hold"
    entry_triggered = False
    exit_triggered = False

    if current_z <= -entry_threshold:
        signal = "buy"
        entry_triggered = True
    elif current_z >= entry_threshold:
        signal = "sell"
        entry_triggered = True
    elif prev_z is not None:
        # Exit signal: z-score crossing back toward zero
        if abs(current_z) < exit_threshold:
            exit_triggered = True

    strength = min(abs(current_z) / (entry_threshold * 2), 1.0)

    return {
        "zscore": round(current_z, 4),
        "signal": signal,
        "strength": round(strength, 4),
        "entry_triggered": entry_triggered,
        "exit_triggered": exit_triggered,
    }


# ---------------------------------------------------------------------------
# Master signal aggregator
# ---------------------------------------------------------------------------

def compute_all_signals(
    intraday_candles: list[dict],
    longterm_candles: list[dict],
    account_value: float = 10000,
    risk_per_trade_pct: float = 1.0,
) -> dict:
    """Compute all quantitative signals for a single asset.

    This is the main entry point called before the LLM decision.
    Returns a structured dict of pre-computed signals.
    """
    closes_5m = [c["close"] for c in intraday_candles] if intraday_candles else []
    closes_4h = [c["close"] for c in longterm_candles] if longterm_candles else []

    # 1. Mean reversion signal (Starfruit-style)
    mr_signal = mean_reversion_signal(intraday_candles) if intraday_candles else {}

    # 2. Regime detection
    regime = detect_regime(longterm_candles) if longterm_candles else {}

    # 3. Multi-timeframe alignment
    alignment = compute_timeframe_alignment(intraday_candles, longterm_candles) \
        if intraday_candles and longterm_candles else {}

    # 4. Z-score on 5m closes (short-term mean reversion)
    zscore_5m = zscore_signal(closes_5m, window=30, entry_threshold=2.0) if len(closes_5m) > 30 else {}

    # 5. Z-score on 4h closes (medium-term mean reversion)
    zscore_4h = zscore_signal(closes_4h, window=20, entry_threshold=1.5) if len(closes_4h) > 20 else {}

    # 6. Bollinger %B (overbought/oversold)
    bpb_5m = bollinger_pct_b(intraday_candles) if intraday_candles else []
    bpb_latest = latest(bpb_5m)

    # 7. Position sizing
    sizing = compute_position_size(
        longterm_candles if longterm_candles else intraday_candles,
        account_value, risk_per_trade_pct
    ) if (longterm_candles or intraday_candles) else {}

    # 8. Composite recommendation
    recommendation = _compute_recommendation(
        mr_signal, regime, alignment, zscore_5m, zscore_4h, bpb_latest
    )

    return {
        "mean_reversion": mr_signal,
        "regime": regime,
        "timeframe_alignment": alignment,
        "zscore_5m": zscore_5m,
        "zscore_4h": zscore_4h,
        "bollinger_pct_b": round(bpb_latest, 4) if bpb_latest is not None else None,
        "position_sizing": sizing,
        "recommendation": recommendation,
    }


def _compute_recommendation(
    mr_signal: dict,
    regime: dict,
    alignment: dict,
    zscore_5m: dict,
    zscore_4h: dict,
    bpb: float | None,
) -> dict:
    """Combine all signals into a single recommendation.

    This is the "brain" that weights signals based on the detected regime.
    In trending regimes: weight momentum higher.
    In ranging regimes: weight mean reversion higher.
    """
    regime_type = regime.get("regime", "unknown")

    # Signal scores (-1 to +1)
    scores = {}

    # Mean reversion score
    scores["mean_reversion"] = mr_signal.get("signal", 0)

    # Timeframe alignment direction
    scores["tf_direction"] = alignment.get("direction", 0)

    # Z-score signals mapped to -1..+1
    z5m = zscore_5m.get("zscore", 0)
    z4h = zscore_4h.get("zscore", 0)
    # Negative z-score = below mean = buy signal (positive score)
    scores["zscore_5m"] = round(math.tanh(-z5m / 2), 4) if z5m else 0
    scores["zscore_4h"] = round(math.tanh(-z4h / 2), 4) if z4h else 0

    # Bollinger %B mapped to signal (-1 at upper band, +1 at lower band)
    if bpb is not None:
        scores["bollinger"] = round(math.tanh(-(bpb - 0.5) * 3), 4)
    else:
        scores["bollinger"] = 0

    # Regime-dependent weighting
    if regime_type in ("trending_up", "trending_down"):
        # In trends: follow momentum, less mean reversion
        weights = {
            "tf_direction": 0.40,
            "mean_reversion": 0.10,
            "zscore_5m": 0.10,
            "zscore_4h": 0.15,
            "bollinger": 0.25,
        }
    elif regime_type == "ranging":
        # In ranges: heavy mean reversion
        weights = {
            "tf_direction": 0.15,
            "mean_reversion": 0.25,
            "zscore_5m": 0.20,
            "zscore_4h": 0.25,
            "bollinger": 0.15,
        }
    else:
        # Volatile / unknown: conservative balanced
        weights = {
            "tf_direction": 0.25,
            "mean_reversion": 0.20,
            "zscore_5m": 0.15,
            "zscore_4h": 0.20,
            "bollinger": 0.20,
        }

    composite = sum(scores[k] * weights[k] for k in weights)

    # Action determination with confidence
    # NOTE: threshold aligned with risk_manager signal-quality gate (0.2) as of Run 21.
    # Previously 0.25 (asymmetric with gate), which parked 0.20–0.25 composites in a
    # "dead zone" — strong enough to pass the gate but too weak for action label.
    confidence = abs(composite)
    if composite > 0.2:
        action = "buy"
    elif composite < -0.2:
        action = "sell"
    else:
        action = "hold"

    # Confidence classification
    if confidence > 0.5:
        confidence_label = "high"
    elif confidence > 0.25:
        confidence_label = "moderate"
    else:
        confidence_label = "low"

    return {
        "action": action,
        "composite_score": round(composite, 4),
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label,
        "regime_type": regime_type,
        "signal_scores": scores,
        "signal_weights": weights,
    }
