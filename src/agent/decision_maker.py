"""Decision-making agent that orchestrates LLM prompts and indicator lookups.

Uses the Anthropic Claude API directly for trade decisions.
"""

import asyncio
import anthropic
from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
import json
import logging
from datetime import datetime


class TradingAgent:
    """High-level trading agent that delegates reasoning to Claude."""

    def __init__(self, hyperliquid=None):
        self.model = CONFIG["llm_model"]
        self.client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        self.sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-4-5-20251001"
        self.max_tokens = int(CONFIG.get("max_tokens") or 4096)

    def decide_trade(self, assets, context):
        """Decide for multiple assets in one call."""
        return self._decide(context, assets=assets)

    def _decide(self, context, assets):
        """Dispatch decision request to Claude and enforce output contract."""
        system_prompt = (
            "You are a rigorous QUANTITATIVE TRADER operating a systematic strategy engine.\n"
            "You will receive PRE-COMPUTED QUANTITATIVE SIGNALS alongside raw indicators for SEVERAL assets.\n\n"
            f"Assets: {json.dumps(list(assets))}\n\n"
            "CRITICAL ARCHITECTURE: Three layers run BEFORE you:\n"
            "  Layer 1 — RESEARCH: Free news feeds (RSS), Reddit, CoinGecko, Fear & Greed, DeFi Llama,\n"
            "    on-chain funding/OI analysis. Cross-validated (events need 2+ independent sources).\n"
            "  Layer 2 — QUANT SIGNALS: Z-scores, mean reversion, regime detection, spread trading,\n"
            "    timeframe alignment, ATR-based position sizing.\n"
            "  Layer 3 — YOU: Validate, cross-reference, and execute.\n\n"
            "You receive:\n"
            "- research: Market sentiment, risk alerts, cross-validated events, per-asset sentiment,\n"
            "  macro context (Fear & Greed, BTC dominance, stablecoin flows), on-chain signals.\n"
            "- quant_signals: Pre-computed signals per asset with composite recommendation.\n"
            "- Raw indicators for additional validation.\n"
            "- Account state, positions, risk limits.\n\n"
            "YOUR ROLE: SIGNAL VALIDATOR + RISK-ADJUSTED EXECUTOR across all three data layers.\n\n"
            "Decision framework (priority order):\n"
            "1. READ RESEARCH FIRST (the 'research' section):\n"
            "   - risk_alerts: HIGH PRIORITY. If alerts mention liquidation risk or extreme fear/greed,\n"
            "     adjust position sizes down or hold. Risk alerts are cross-validated (2+ sources).\n"
            "   - asset_sentiment: Per-asset sentiment score (-1 to +1) from news analysis.\n"
            "   - key_events: Cross-validated events (appeared in 2+ independent sources).\n"
            "   - macro.fear_greed: Extreme values (<20 or >80) are contrarian signals.\n"
            "   - on_chain: Funding rate anomalies and OI signals from Hyperliquid.\n\n"
            "2. READ quant_signals.recommendation for each asset:\n"
            "   - action: buy/sell/hold (the system's mathematical recommendation)\n"
            "   - composite_score: -1 to +1 (strength and direction)\n"
            "   - confidence_label: high/moderate/low\n"
            "   - regime_type: trending_up/trending_down/ranging/volatile\n"
            "   - signal_scores: individual signal contributions\n\n"
            "3. CROSS-REFERENCE research with quant signals:\n"
            "   - research bearish + quant bearish = HIGH CONFIDENCE SHORT\n"
            "   - research bullish + quant bullish = HIGH CONFIDENCE LONG\n"
            "   - research conflicts quant = REDUCE CONFIDENCE, prefer HOLD\n"
            "   - research neutral + quant has signal = TRUST QUANT (normal operation)\n"
            "   - risk_alerts present = ALWAYS reduce position size or hold\n\n"
            "4. VALIDATE against:\n"
            "   - Current positions (avoid conflicting with active trades unless invalidated)\n"
            "   - Risk limits (the system enforces these, but you should be aware)\n"
            "   - Spread signals (if assets are in a correlated pair, respect spread signals)\n\n"
            "5. OVERRIDE only when you have strong evidence the quant signal is wrong:\n"
            "   - A cross-validated key_event directly contradicts the quant signal\n"
            "   - On-chain data shows extreme funding + surging OI (liquidation cascade risk)\n"
            "   - Position management (existing position makes the signal redundant or dangerous)\n\n"
            "4. USE the position_sizing recommendation:\n"
            "   - quant_signals.position_sizing.suggested_usd is ATR-calibrated\n"
            "   - You MAY adjust by +/-30% based on confidence, but respect the order of magnitude\n"
            "   - NEVER ignore the suggested size and substitute a random number\n\n"
            "5. SET TP/SL using ATR-based levels:\n"
            "   - TP: 2-3x ATR from entry in trade direction\n"
            "   - SL: 1-1.5x ATR from entry against trade direction\n"
            "   - The quant_signals.position_sizing.atr_stop_distance gives the calibrated distance\n\n"
            "Signal interpretation guide:\n"
            "- REGIME matters most for strategy selection:\n"
            "  • trending_up/down: Follow the composite_score direction. Mean reversion signals are LESS reliable.\n"
            "  • ranging: Mean reversion signals are MOST reliable. Z-score reversals are your primary edge.\n"
            "  • volatile: Reduce position sizes. Require HIGH confidence to enter.\n\n"
            "- SPREAD SIGNALS (if present in quant_signals.spread_signals):\n"
            "  • When a pair's spread_zscore exceeds 2.0: the statistical edge is strongest.\n"
            "  • Spread trades should be entered as pairs (long A + short B, or vice versa).\n"
            "  • Spread signals can OVERRIDE single-asset signals when correlation is > 0.6.\n\n"
            "- Z-SCORE signals:\n"
            "  • zscore_5m: Short-term mean reversion. Best in ranging markets.\n"
            "  • zscore_4h: Medium-term mean reversion. More reliable but slower.\n"
            "  • Entry when |z| > 2.0, exit when |z| < 0.5.\n"
            "  • IMPORTANT: z-score between 1.0 and 2.0 is NOT a 'sell gate' or entry blocker.\n"
            "    It means the asset is slightly extended — you can still enter with reduced size.\n"
            "    Only z-score > 2.5 should make you cautious about entering in the same direction.\n"
            "  • In trending markets (ADX > 25), z-score can stay elevated for extended periods.\n"
            "    Do NOT wait for z-score to drop to 0 before entering a trend.\n\n"
            "- MEAN REVERSION signal (Starfruit-style):\n"
            "  • Predicts partial reversion of recent price moves.\n"
            "  • Most useful for intraday scalping in ranging regimes.\n"
            "  • Ignore in strong trends (ADX > 30).\n\n"
            "Low-churn policies (STILL ENFORCED):\n"
            "1) Respect prior plans: Don't close/flip early unless invalidation occurred.\n"
            "2) Hysteresis: Require stronger evidence to CHANGE than to KEEP a decision.\n"
            "3) Cooldown: At least 3 bars between direction changes unless hard invalidation.\n"
            "4) Funding is a tilt, not a trigger.\n"
            "5) Overbought/oversold alone is not a reversal signal.\n"
            "6) Prefer adjustments over exits when thesis weakens but isn't invalidated.\n"
            "7) ANTI-PARALYSIS: If you have been HOLD for 10+ consecutive cycles with no position,\n"
            "   you MUST enter at least one trade. Sitting flat for hours is worse than a small\n"
            "   position with a stop-loss. Use market orders with reduced size (50% of suggested)\n"
            "   if you're uncertain, rather than waiting for perfect conditions that may never come.\n"
            "8) SCALE-OUT IS AUTOMATED — DO NOT EMIT IT YOURSELF.\n"
            "   The risk manager automatically closes 50% of any open position when it reaches\n"
            "   >= +0.5% unrealized PnL and has been held >= 10 cycles, then moves SL to breakeven.\n"
            "   You do NOT need to emit sell/buy actions for partial profit-taking. If you see\n"
            "   a position at breakeven SL with reduced size, it is a protected runner — hold it.\n"
            "   Only emit a sell/buy on an existing position if you have a genuine directional\n"
            "   reversal thesis (hard invalidation), not for profit-taking.\n"
            "9) SIGNAL-QUALITY GATE (NEW ENTRIES ONLY):\n"
            "   New entries will be rejected by the risk manager if confidence_label is 'low'\n"
            "   AND |composite_score| < 0.2. Exception: anti-paralysis (rule 7) bypasses this gate.\n"
            "   Practical implication: do NOT propose a buy/sell for a flat asset whose quant\n"
            "   signals show low confidence AND weak composite — it will be blocked anyway.\n"
            "   Prefer hold. This does NOT apply to positions you already hold (scale-out,\n"
            "   close, adjust SL are all still allowed regardless of signal quality).\n\n"
            "Order types:\n"
            "- order_type: \"market\" (immediate) or \"limit\" (resting at limit_price)\n"
            "- In RANGING regimes: prefer limit orders at z-score extremes for better fills.\n"
            "- In TRENDING regimes: USE MARKET ORDERS. Limit orders in trends often never fill\n"
            "  because the price keeps moving away. Missing the trend is worse than paying slippage.\n"
            "- If limit_price is more than 1% below current price for buys (or above for sells),\n"
            "  strongly consider using a market order instead — the order may never fill.\n\n"
            "Tool usage:\n"
            "- Use fetch_indicator for additional data if the pre-computed signals are inconclusive.\n"
            "- Available: ema, sma, rsi, macd, bbands, atr, adx, obv, vwap, stoch_rsi, all.\n\n"
            "Output contract:\n"
            "- Output ONLY a strict JSON object (no markdown, no code fences) with exactly two properties:\n"
            "  • \"reasoning\": Step-by-step: 1) Research summary, 2) Quant signals, 3) Cross-reference, 4) Decide, 5) Size.\n"
            "  • \"trade_decisions\": array ordered to match the provided assets list.\n"
            "- Each trade_decisions item: asset, action, allocation_usd, order_type, limit_price, tp_price, sl_price, exit_plan, rationale.\n"
            "  • order_type: \"market\" or \"limit\"\n"
            "  • limit_price: required if order_type is \"limit\", null otherwise\n"
            "- Do not emit Markdown or any extra properties.\n"
        )

        tools = [{
            "name": "fetch_indicator",
            "description": (
                "Fetch technical indicators computed locally from Hyperliquid candle data. "
                "Works for ALL Hyperliquid perp markets including crypto (BTC, ETH, SOL), "
                "commodities (OIL, GOLD, SILVER), indices (SPX), and more. "
                "Available indicators: ema, sma, rsi, macd, bbands, atr, adx, obv, vwap, stoch_rsi, all. "
                "Returns the latest values and recent series."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "obv", "vwap", "stoch_rsi", "all"],
                    },
                    "asset": {
                        "type": "string",
                        "description": "Hyperliquid asset symbol, e.g. BTC, ETH, OIL, GOLD, SPX",
                    },
                    "interval": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    },
                    "period": {
                        "type": "integer",
                        "description": "Indicator period (default varies by indicator)",
                    },
                },
                "required": ["indicator", "asset", "interval"],
            },
        }]

        messages = [{"role": "user", "content": context}]

        def _log_request(model, messages_to_log):
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {model}\n")
                f.write(f"Messages count: {len(messages_to_log)}\n")
                # Log last message content (truncated)
                last = messages_to_log[-1]
                content_str = str(last.get("content", ""))[:500]
                f.write(f"Last message role: {last.get('role')}\n")
                f.write(f"Last message content (truncated): {content_str}\n")

        enable_tools = CONFIG.get("enable_tool_calling", False)

        def _call_claude(msgs, use_tools=True):
            """Make a Claude API call with optional tool use."""
            _log_request(self.model, msgs)
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": msgs,
            }
            if use_tools and enable_tools:
                kwargs["tools"] = tools
            if CONFIG.get("thinking_enabled"):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(CONFIG.get("thinking_budget_tokens") or 10000),
                }
                # When thinking is enabled, max_tokens must be larger
                kwargs["max_tokens"] = max(self.max_tokens, 16000)

            # Retry on transient errors (529 overloaded, 500, network)
            # Falls back to Haiku if primary model stays overloaded
            import time as _time
            _fallback_model = "claude-haiku-4-5-20251001"
            last_err = None
            for _attempt in range(4):
                try:
                    # On attempt 4, fall back to Haiku
                    if _attempt == 3 and kwargs["model"] != _fallback_model:
                        logging.warning("Primary model overloaded after 3 attempts, falling back to %s", _fallback_model)
                        kwargs["model"] = _fallback_model
                    response = self.client.messages.create(**kwargs)
                    logging.info("Claude response (model=%s): stop_reason=%s, usage=%s",
                                kwargs["model"], response.stop_reason, response.usage)
                    with open("llm_requests.log", "a", encoding="utf-8") as f:
                        f.write(f"Response stop_reason: {response.stop_reason}\n")
                        f.write(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}\n")
                    return response
                except Exception as _e:
                    last_err = _e
                    err_str = str(_e)
                    if "529" in err_str or "overloaded" in err_str.lower() or "500" in err_str:
                        wait = 10 * (_attempt + 1)
                        logging.warning("Claude API transient error (attempt %d/4), retrying in %ds: %s",
                                       _attempt + 1, wait, err_str[:100])
                        _time.sleep(wait)
                    else:
                        raise
            raise last_err

        def _handle_tool_call(tool_name, tool_input):
            """Execute a tool call and return the result string."""
            if tool_name != "fetch_indicator":
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            try:
                asset = tool_input["asset"]
                interval = tool_input["interval"]
                indicator = tool_input["indicator"]

                # Fetch candles from Hyperliquid
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        candles = pool.submit(
                            asyncio.run,
                            self.hyperliquid.get_candles(asset, interval, 100)
                        ).result(timeout=30)
                else:
                    candles = asyncio.run(self.hyperliquid.get_candles(asset, interval, 100))

                all_indicators = compute_all(candles)

                if indicator == "all":
                    result = {k: {"latest": latest(v) if isinstance(v, list) else v,
                                  "series": last_n(v, 10) if isinstance(v, list) else v}
                              for k, v in all_indicators.items()}
                elif indicator == "macd":
                    result = {
                        "macd": {"latest": latest(all_indicators.get("macd", [])), "series": last_n(all_indicators.get("macd", []), 10)},
                        "signal": {"latest": latest(all_indicators.get("macd_signal", [])), "series": last_n(all_indicators.get("macd_signal", []), 10)},
                        "histogram": {"latest": latest(all_indicators.get("macd_histogram", [])), "series": last_n(all_indicators.get("macd_histogram", []), 10)},
                    }
                elif indicator == "bbands":
                    result = {
                        "upper": {"latest": latest(all_indicators.get("bbands_upper", [])), "series": last_n(all_indicators.get("bbands_upper", []), 10)},
                        "middle": {"latest": latest(all_indicators.get("bbands_middle", [])), "series": last_n(all_indicators.get("bbands_middle", []), 10)},
                        "lower": {"latest": latest(all_indicators.get("bbands_lower", [])), "series": last_n(all_indicators.get("bbands_lower", []), 10)},
                    }
                elif indicator in ("ema", "sma"):
                    period = tool_input.get("period", 20)
                    from src.indicators.local_indicators import ema as _ema, sma as _sma
                    closes = [c["close"] for c in candles]
                    series = _ema(closes, period) if indicator == "ema" else _sma(closes, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "rsi":
                    period = tool_input.get("period", 14)
                    from src.indicators.local_indicators import rsi as _rsi
                    series = _rsi(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "atr":
                    period = tool_input.get("period", 14)
                    from src.indicators.local_indicators import atr as _atr
                    series = _atr(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                else:
                    key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap", "stoch_rsi": "stoch_rsi"}
                    mapped = key_map.get(indicator, indicator)
                    series = all_indicators.get(mapped, [])
                    result = {"latest": latest(series) if isinstance(series, list) else series,
                              "series": last_n(series, 10) if isinstance(series, list) else series}

                return json.dumps(result, default=str)
            except Exception as ex:
                logging.error("Tool call error: %s", ex)
                return json.dumps({"error": str(ex)})

        def _sanitize_output(raw_content: str, assets_list):
            """Use a cheap Claude model to normalize malformed output."""
            try:
                response = self.client.messages.create(
                    model=self.sanitize_model,
                    max_tokens=2048,
                    system=(
                        "You are a strict JSON normalizer. Return ONLY a JSON object with two keys: "
                        "\"reasoning\" (string) and \"trade_decisions\" (array). "
                        "Each trade_decisions item must have: asset, action (buy/sell/hold), "
                        "allocation_usd (number), order_type (\"market\" or \"limit\"), "
                        "limit_price (number or null), tp_price (number or null), sl_price (number or null), "
                        "exit_plan (string), rationale (string). "
                        f"Valid assets: {json.dumps(list(assets_list))}. "
                        "If input is wrapped in markdown or has prose, extract just the JSON. Do not add fields."
                    ),
                    messages=[{"role": "user", "content": raw_content}],
                )
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "trade_decisions" in parsed:
                    return parsed
                return {"reasoning": "", "trade_decisions": []}
            except Exception as se:
                logging.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        # Main loop: up to 6 iterations to handle tool calls
        for iteration in range(6):
            try:
                response = _call_claude(messages)
            except anthropic.APIError as e:
                logging.error("Claude API error: %s", e)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"API Error: {e}\n")
                break

            # Check if the response contains tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_use_blocks and response.stop_reason == "tool_use":
                # Build assistant message with all content blocks
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif block.type == "thinking":
                        assistant_content.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                # Process each tool call
                tool_results = []
                for block in tool_use_blocks:
                    result_str = _handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # No tool calls — parse the text response as JSON
            raw_text = ""
            for block in text_blocks:
                raw_text += block.text

            if not raw_text.strip():
                logging.error("Empty response from Claude")
                break

            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    logging.error("Expected dict, got: %s; attempting sanitize", type(parsed))
                    return _sanitize_output(raw_text, assets)

                reasoning_text = parsed.get("reasoning", "") or ""
                decisions = parsed.get("trade_decisions")

                if isinstance(decisions, list):
                    normalized = []
                    for item in decisions:
                        if isinstance(item, dict):
                            item.setdefault("allocation_usd", 0.0)
                            item.setdefault("order_type", "market")
                            item.setdefault("limit_price", None)
                            item.setdefault("tp_price", None)
                            item.setdefault("sl_price", None)
                            item.setdefault("exit_plan", "")
                            item.setdefault("rationale", "")
                            normalized.append(item)
                    return {"reasoning": reasoning_text, "trade_decisions": normalized}

                logging.error("trade_decisions missing or invalid; attempting sanitize")
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, raw_text[:200])
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {
                    "reasoning": "Parse error",
                    "trade_decisions": [{
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error"
                    } for a in assets]
                }

        # Exhausted tool loop
        return {
            "reasoning": "tool loop cap",
            "trade_decisions": [{
                "asset": a,
                "action": "hold",
                "allocation_usd": 0.0,
                "tp_price": None,
                "sl_price": None,
                "exit_plan": "",
                "rationale": "tool loop cap"
            } for a in assets]
        }
