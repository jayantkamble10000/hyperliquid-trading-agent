"""Entry-point script that wires together the trading agent, data feeds, and API.

Enhanced with IMC Prosperity quantitative strategy engine:
- Pre-computed quantitative signals (z-scores, mean reversion, regime detection)
- Spread/pair trading between correlated assets
- ATR-based position sizing
- Multi-timeframe signal alignment
"""

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.agent.decision_maker import TradingAgent
from src.indicators.local_indicators import compute_all, last_n, latest
from src.risk_manager import RiskManager
from src.trading.hyperliquid_api import HyperliquidAPI
from src.trading.paper_trader import PaperTrader
from src.strategies.quant_signals import compute_all_signals
from src.strategies.spread_trader import SpreadTradingEngine
from src.research.research_engine import ResearchEngine
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone
import math  # For Sharpe
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_interval_seconds(interval_str):
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    from src.config_loader import CONFIG
    _cfg = CONFIG
    assets_env = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")
    if (not args.assets or len(args.assets) == 0) and assets_env:
        # Support space or comma separated
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    # Choose real or paper trading based on config
    paper_mode = _cfg.get("paper_trade", False) if isinstance(_cfg.get("paper_trade"), bool) else str(_cfg.get("paper_trade", "false")).lower() in ("1", "true", "yes", "on")
    if paper_mode:
        paper_balance = float(_cfg.get("paper_initial_balance", "10000"))
        hyperliquid = PaperTrader(initial_balance=paper_balance)
        logging.info("=" * 60)
        logging.info("PAPER TRADING MODE — No real money, simulated execution")
        logging.info("Initial balance: $%.2f", paper_balance)
        logging.info("=" * 60)
    else:
        hyperliquid = HyperliquidAPI()

    agent = TradingAgent(hyperliquid=hyperliquid)
    risk_mgr = RiskManager()

    # Initialize spread trading engine with configured assets
    spread_engine = SpreadTradingEngine(
        assets=args.assets,
        pairs=json.loads(_cfg.get("spread_pairs") or "[]") or None
    )

    # Initialize research engine (free sources, runs before everything else)
    use_ollama = _cfg.get("use_ollama", "true").lower() in ("1", "true", "yes", "on") \
        if isinstance(_cfg.get("use_ollama"), str) else bool(_cfg.get("use_ollama", True))
    research_engine = ResearchEngine(assets=args.assets, use_ollama=use_ollama)

    # Strategy parameters from config
    risk_per_trade_pct = float(_cfg.get("risk_per_trade_pct") or 1.0)

    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = []  # For Sharpe: list of returns
    active_trades = []  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    recent_events = deque(maxlen=200)
    diary_path = "diary.jsonl"
    initial_account_value = None
    # Mandatory scale-out tracking
    position_cycle_count = {}  # coin -> cycles open (reset when position closes)
    scaled_out_coins = set()   # coins already scaled out on this position instance
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
        logging.info(msg)

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        nonlocal invocation_count, initial_account_value

        # Pre-load meta cache for correct order sizing
        await hyperliquid.get_meta_and_ctxs()
        # Pre-load HIP-3 dex meta for any dex:asset in the asset list
        hip3_dexes = set()
        for a in args.assets:
            if ":" in a:
                hip3_dexes.add(a.split(":")[0])
        for dex in hip3_dexes:
            await hyperliquid.get_meta_and_ctxs(dex=dex)
            add_event(f"Loaded HIP-3 meta for dex: {dex}")

        while True:
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state
            state = await hyperliquid.get_user_state()
            total_value = state.get('total_value') or state['balance'] + sum(p.get('pnl', 0) for p in state['positions'])
            sharpe = calculate_sharpe(trade_log)

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            positions = []
            for pos_wrap in state['positions']:
                pos = pos_wrap
                coin = pos.get('coin')
                current_px = await hyperliquid.get_current_price(coin) if coin else None
                qty = pos.get('szi')
                qty_float = float(qty) if qty else 0
                positions.append({
                    "symbol": coin,
                    "direction": "LONG" if qty_float > 0 else "SHORT" if qty_float < 0 else "FLAT",
                    "quantity": round_or_none(qty, 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })

            # --- RISK: Force-close positions that exceed max loss ---
            try:
                positions_to_close = risk_mgr.check_losing_positions(state['positions'])
                for ptc in positions_to_close:
                    coin = ptc["coin"]
                    size = ptc["size"]
                    is_long = ptc["is_long"]
                    add_event(f"RISK FORCE-CLOSE: {coin} at {ptc['loss_pct']}% loss (PnL: ${ptc['pnl']})")
                    try:
                        if is_long:
                            await hyperliquid.place_sell_order(coin, size)
                        else:
                            await hyperliquid.place_buy_order(coin, size)
                        await hyperliquid.cancel_all_orders(coin)
                        # Remove from active trades
                        for tr in active_trades[:]:
                            if tr.get('asset') == coin:
                                active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": coin,
                                "action": "risk_force_close",
                                "loss_pct": ptc["loss_pct"],
                                "pnl": ptc["pnl"],
                            }) + "\n")
                    except Exception as fc_err:
                        add_event(f"Force-close error for {coin}: {fc_err}")
            except Exception as risk_err:
                add_event(f"Risk check error: {risk_err}")

            # --- RISK: Update per-coin cycle counters and run mandatory scale-out ---
            try:
                open_coins_now = set()
                for pos in state['positions']:
                    coin = pos.get('coin')
                    size = float(pos.get('szi') or 0)
                    if coin and abs(size) > 0:
                        open_coins_now.add(coin)
                        position_cycle_count[coin] = position_cycle_count.get(coin, 0) + 1
                # Reset counters + scaled_out flag for any coin that closed
                for coin in list(position_cycle_count.keys()):
                    if coin not in open_coins_now:
                        position_cycle_count.pop(coin, None)
                        scaled_out_coins.discard(coin)

                scale_outs = risk_mgr.check_mandatory_scale_outs(
                    state['positions'], position_cycle_count, scaled_out_coins
                )
                for so in scale_outs:
                    coin = so["coin"]
                    size = so["size"]
                    is_long = so["is_long"]
                    breakeven = so["breakeven_sl"]
                    add_event(
                        f"RISK SCALE-OUT: {coin} 50% "
                        f"(+{so['unrealized_pct']}%, {so['cycles_held']} cycles, PnL ${so['pnl']})"
                    )
                    try:
                        # Close 50% at market (opposite-side order)
                        if is_long:
                            await hyperliquid.place_sell_order(coin, size)
                        else:
                            await hyperliquid.place_buy_order(coin, size)
                        # Cancel existing SL and replace at breakeven for remaining 50%
                        try:
                            await hyperliquid.cancel_all_orders(coin)
                        except Exception:
                            pass
                        try:
                            remaining = size  # remaining half equals the closed half
                            sl_order = await hyperliquid.place_stop_loss(
                                coin, is_long, remaining, breakeven
                            )
                            add_event(f"RISK SCALE-OUT: breakeven SL for {coin} at ${breakeven}")
                        except Exception as sl_err:
                            add_event(f"Breakeven SL place error for {coin}: {sl_err}")
                        scaled_out_coins.add(coin)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": coin,
                                "action": "scale_out",
                                "unrealized_pct": so["unrealized_pct"],
                                "cycles_held": so["cycles_held"],
                                "pnl": so["pnl"],
                                "breakeven_sl": breakeven,
                            }) + "\n")
                    except Exception as so_err:
                        add_event(f"Scale-out error for {coin}: {so_err}")
            except Exception as so_outer:
                add_event(f"Scale-out check error: {so_outer}")

            recent_diary = []
            consecutive_holds = 0
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        entry = json.loads(line)
                        recent_diary.append(entry)
                    # Count consecutive hold-only cycles from the end
                    for line in reversed(lines):
                        try:
                            entry = json.loads(line)
                            if entry.get("action") in ("hold", "risk_blocked"):
                                consecutive_holds += 1
                            else:
                                break
                        except Exception:
                            continue
            except Exception:
                pass

            open_orders_struct = []
            try:
                open_orders = await hyperliquid.get_open_orders()
                for o in open_orders[:50]:
                    open_orders_struct.append({
                        "coin": o.get('coin'),
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "order_type": o.get('orderType')
                    })
            except Exception:
                open_orders = []

            # Reconcile active trades
            try:
                assets_with_positions = set()
                for pos in state['positions']:
                    try:
                        if abs(float(pos.get('szi') or 0)) > 0:
                            assets_with_positions.add(pos.get('coin'))
                    except Exception:
                        continue
                assets_with_orders = {o.get('coin') for o in (open_orders or []) if o.get('coin')}
                for tr in active_trades[:]:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions and asset not in assets_with_orders:
                        add_event(f"Reconciling stale active trade for {asset} (no position, no orders)")
                        active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "reconcile_close",
                                "reason": "no_position_no_orders",
                                "opened_at": tr.get('opened_at')
                            }) + "\n")
            except Exception:
                pass

            recent_fills_struct = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=50)
                for f_entry in fills[-20:]:
                    try:
                        t_raw = f_entry.get('time') or f_entry.get('timestamp')
                        timestamp = None
                        if t_raw is not None:
                            try:
                                t_int = int(t_raw)
                                if t_int > 1e12:
                                    timestamp = datetime.fromtimestamp(t_int / 1000, tz=timezone.utc).isoformat()
                                else:
                                    timestamp = datetime.fromtimestamp(t_int, tz=timezone.utc).isoformat()
                            except Exception:
                                timestamp = str(t_raw)
                        recent_fills_struct.append({
                            "timestamp": timestamp,
                            "coin": f_entry.get('coin') or f_entry.get('asset'),
                            "is_buy": f_entry.get('isBuy'),
                            "size": round_or_none(f_entry.get('sz') or f_entry.get('size'), 6),
                            "price": round_or_none(f_entry.get('px') or f_entry.get('price'), 2)
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
                "sharpe_ratio": round_or_none(sharpe, 3),
                "consecutive_holds": consecutive_holds,
                "anti_paralysis_active": consecutive_holds >= 10 and len(positions) == 0,
                "positions": positions,
                "active_trades": [
                    {
                        "asset": tr.get('asset'),
                        "is_long": tr.get('is_long'),
                        "amount": round_or_none(tr.get('amount'), 6),
                        "entry_price": round_or_none(tr.get('entry_price'), 2),
                        "tp_oid": tr.get('tp_oid'),
                        "sl_oid": tr.get('sl_oid'),
                        "exit_plan": tr.get('exit_plan'),
                        "opened_at": tr.get('opened_at')
                    }
                    for tr in active_trades
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
            }

            # --- RESEARCH LAYER: Run FIRST, before any analysis ---
            research_briefing = None
            try:
                # Collect funding/OI for on-chain analysis
                _funding_rates = {}
                _open_interests = {}
                for _asset in args.assets:
                    try:
                        _fr = await hyperliquid.get_funding_rate(_asset)
                        _oi = await hyperliquid.get_open_interest(_asset)
                        if _fr is not None:
                            _funding_rates[_asset] = _fr
                        if _oi is not None:
                            _open_interests[_asset] = _oi
                    except Exception:
                        continue

                research_briefing = await research_engine.get_briefing(
                    funding_rates=_funding_rates,
                    open_interests=_open_interests,
                )
                add_event(
                    f"Research: {research_briefing.data_quality['total_items']} items, "
                    f"{research_briefing.data_quality['fresh_items_24h']} fresh, "
                    f"{len(research_briefing.risk_alerts)} alerts, "
                    f"sentiment={research_briefing.market_sentiment['market_label']}"
                )
                if research_briefing.risk_alerts:
                    for alert in research_briefing.risk_alerts:
                        add_event(f"RESEARCH ALERT: {alert}")
            except Exception as re_err:
                add_event(f"Research layer error (non-fatal): {re_err}")

            # Gather data for ALL assets first (using Hyperliquid candles + local indicators)
            market_sections = []
            asset_prices = {}
            asset_quant_lookup = {}  # asset -> {confidence_label, composite_score} for signal-quality gate
            asset_candles = {}  # Store candles for quant signal computation
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    # Fetch candles from Hyperliquid and compute indicators locally
                    candles_5m = await hyperliquid.get_candles(asset, "5m", 100)
                    candles_4h = await hyperliquid.get_candles(asset, "4h", 100)
                    asset_candles[asset] = {"5m": candles_5m, "4h": candles_4h}

                    intra = compute_all(candles_5m)
                    lt = compute_all(candles_4h)

                    # --- QUANT SIGNALS: Compute pre-LLM quantitative signals ---
                    quant_signals = compute_all_signals(
                        intraday_candles=candles_5m,
                        longterm_candles=candles_4h,
                        account_value=account_value or 10000,
                        risk_per_trade_pct=risk_per_trade_pct,
                    )

                    # Add spread signals for this asset
                    spread_signals = spread_engine.get_asset_spread_signals(asset)
                    quant_signals["spread_signals"] = spread_signals

                    # Index quant signals for signal-quality gate in risk manager.
                    # NOTE: composite_score and confidence_label live inside the
                    # nested "recommendation" dict returned by compute_all_signals,
                    # not at the top level. Run 15 post-mortem caught this — reading
                    # the top level silently returned the defaults on every cycle,
                    # making the gate evaluate every asset as low/0.0 regardless.
                    _rec = quant_signals.get("recommendation", {}) or {}
                    asset_quant_lookup[asset] = {
                        "confidence_label": _rec.get("confidence_label", "low"),
                        "composite_score": _rec.get("composite_score", 0.0),
                    }

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-10:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        # PRE-COMPUTED QUANT SIGNALS (the main decision input)
                        "quant_signals": quant_signals,
                        "intraday": {
                            "ema20": round_or_none(latest(intra.get("ema20", [])), 2),
                            "macd": round_or_none(latest(intra.get("macd", [])), 2),
                            "rsi7": round_or_none(latest(intra.get("rsi7", [])), 2),
                            "rsi14": round_or_none(latest(intra.get("rsi14", [])), 2),
                            "bollinger_pct_b": round_or_none(latest(intra.get("bollinger_pct_b", [])), 4),
                            "price_zscore": round_or_none(latest(intra.get("price_zscore", [])), 4),
                            "ema20_slope": round_or_none(latest(intra.get("ema20_slope", [])), 4),
                            "series": {
                                "ema20": round_series(last_n(intra.get("ema20", []), 10), 2),
                                "macd": round_series(last_n(intra.get("macd", []), 10), 2),
                                "rsi7": round_series(last_n(intra.get("rsi7", []), 10), 2),
                                "rsi14": round_series(last_n(intra.get("rsi14", []), 10), 2),
                            }
                        },
                        "long_term": {
                            "ema20": round_or_none(latest(lt.get("ema20", [])), 2),
                            "ema50": round_or_none(latest(lt.get("ema50", [])), 2),
                            "atr3": round_or_none(latest(lt.get("atr3", [])), 2),
                            "atr14": round_or_none(latest(lt.get("atr14", [])), 2),
                            "ema20_slope": round_or_none(latest(lt.get("ema20_slope", [])), 4),
                            "ema50_slope": round_or_none(latest(lt.get("ema50_slope", [])), 4),
                            "bollinger_pct_b": round_or_none(latest(lt.get("bollinger_pct_b", [])), 4),
                            "macd_series": round_series(last_n(lt.get("macd", []), 10), 2),
                            "rsi_series": round_series(last_n(lt.get("rsi14", []), 10), 2),
                        },
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids,
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # Update spread engine with latest prices
            spread_engine.update_prices(asset_prices)

            # Get global spread signals summary
            all_spread_signals = spread_engine.get_all_signals()

            # Build research context for LLM (compact version)
            research_context = None
            if research_briefing:
                research_context = {
                    "market_sentiment": research_briefing.market_sentiment,
                    "risk_alerts": research_briefing.risk_alerts,
                    "key_events": [
                        {"event": e["sample_title"][:80], "sources": e["source_count"], "impact": e["impact"]}
                        for e in research_briefing.key_events[:5]
                    ],
                    "asset_sentiment": {
                        asset: sig["final_sentiment"]
                        for asset, sig in research_briefing.asset_signals.items()
                    },
                    "macro": {
                        "fear_greed": research_briefing.market_sentiment.get("fear_greed_index"),
                        "fear_greed_trend": research_briefing.market_sentiment.get("fear_greed_trend"),
                        "btc_dominance": research_briefing.macro_context.get("global_market", {}).get("btc_dominance"),
                        "market_cap_change_24h": research_briefing.macro_context.get("global_market", {}).get("market_cap_change_24h_pct"),
                        "trending_coins": research_briefing.macro_context.get("trending_coins", [])[:5],
                    },
                    "on_chain": research_briefing.on_chain_context,
                    "data_quality": {
                        "sources": research_briefing.data_quality.get("sources_count", 0),
                        "fresh_ratio": research_briefing.data_quality.get("freshness_ratio", 0),
                        "events_validated": research_briefing.data_quality.get("cross_validated_events", 0),
                    },
                }

            # Single LLM call with all assets + quant signals + research
            context_payload = OrderedDict([
                ("invocation", {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count
                }),
                ("research", research_context),
                ("account", dashboard),
                ("risk_limits", risk_mgr.get_risk_summary()),
                ("spread_signals", all_spread_signals),
                ("market_data", market_sections),
                ("instructions", {
                    "assets": args.assets,
                    "requirement": (
                        "STEP 1: Read the 'research' section for macro context, risk alerts, and sentiment. "
                        "STEP 2: For each asset, READ quant_signals.recommendation. "
                        "STEP 3: Cross-reference research sentiment with quant signals. "
                        "  - If research shows bearish sentiment + risk alerts but quant says buy → reduce confidence or hold. "
                        "  - If research confirms quant direction → increase confidence. "
                        "STEP 4: Use position_sizing.suggested_usd for allocation. "
                        "Override the recommendation ONLY with strong justification. "
                        "Return a strict JSON object matching the schema."
                    )
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            with open("prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

            def _is_failed_outputs(outs):
                """Return True when outputs are missing or clearly invalid."""
                if not isinstance(outs, dict):
                    return True
                decisions = outs.get("trade_decisions")
                if not isinstance(decisions, list) or not decisions:
                    return True
                try:
                    return all(
                        isinstance(o, dict)
                        and (o.get('action') == 'hold')
                        and ('parse error' in (o.get('rationale', '').lower()))
                        for o in decisions
                    )
                except Exception:
                    return True

            try:
                outputs = agent.decide_trade(args.assets, context)
                if not isinstance(outputs, dict):
                    add_event(f"Invalid output format (expected dict): {outputs}")
                    outputs = {}
            except Exception as e:
                import traceback
                add_event(f"Agent error: {e}")
                add_event(f"Traceback: {traceback.format_exc()}")
                outputs = {}

            # Retry once on failure/parse error with a stricter instruction prefix
            if _is_failed_outputs(outputs):
                add_event("Retrying LLM once due to invalid/parse-error output")
                context_retry_payload = OrderedDict([
                    ("retry_instruction", "Return ONLY the JSON array per schema with no prose."),
                    ("original_context", context_payload)
                ])
                context_retry = json.dumps(context_retry_payload, default=json_default)
                try:
                    outputs = agent.decide_trade(args.assets, context_retry)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

            # Haiku sometimes returns reasoning as a dict (structured steps) instead
            # of a string. Normalize to string so downstream slicing/logging works.
            _raw_reasoning = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
            if isinstance(_raw_reasoning, str):
                reasoning_text = _raw_reasoning
            elif _raw_reasoning:
                try:
                    reasoning_text = json.dumps(_raw_reasoning, default=str)
                except Exception:
                    reasoning_text = str(_raw_reasoning)
            else:
                reasoning_text = ""
            if reasoning_text:
                add_event(f"LLM reasoning summary: {reasoning_text}")

            # Log full cycle decisions for the dashboard
            cycle_decisions = []
            for d in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                cycle_decisions.append({
                    "asset": d.get("asset"),
                    "action": d.get("action", "hold"),
                    "allocation_usd": d.get("allocation_usd", 0),
                    "rationale": d.get("rationale", ""),
                })
            cycle_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": invocation_count,
                "reasoning": reasoning_text[:2000] if reasoning_text else "",
                "decisions": cycle_decisions,
                "account_value": round_or_none(account_value, 2),
                "balance": round_or_none(state['balance'], 2),
                "positions_count": len([p for p in state['positions'] if abs(float(p.get('szi') or 0)) > 0]),
            }
            try:
                with open("decisions.jsonl", "a") as f:
                    f.write(json.dumps(cycle_log) + "\n")
            except Exception:
                pass

            # Execute trades for each asset
            for output in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue
                    action = output.get("action")
                    current_price = asset_prices.get(asset, 0)
                    action = output["action"]
                    rationale = output.get("rationale", "")
                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")
                    if action in ("buy", "sell"):
                        is_buy = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue
                        # Minimum trade size to prevent micro-position churn
                        MIN_TRADE_USD = 100
                        if alloc_usd < MIN_TRADE_USD:
                            add_event(f"Skipping {asset}: allocation ${alloc_usd:.0f} below minimum ${MIN_TRADE_USD}")
                            continue

                        # --- RISK: Signal-quality gate (skip for positions already open) ---
                        already_open = any(
                            (p.get("coin") == asset and abs(float(p.get("szi") or 0)) > 0)
                            for p in state['positions']
                        )
                        if not already_open:
                            anti_par = consecutive_holds >= 10 and len(
                                [p for p in state['positions'] if abs(float(p.get('szi') or 0)) > 0]
                            ) == 0
                            sig_ok, sig_reason = risk_mgr.check_signal_quality(
                                asset, asset_quant_lookup.get(asset), anti_par
                            )
                            if not sig_ok:
                                add_event(f"RISK SIGNAL-QUALITY BLOCK {asset}: {sig_reason}")
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "risk_blocked",
                                        "reason": sig_reason,
                                        "original_alloc_usd": alloc_usd,
                                    }) + "\n")
                                continue

                        # --- RISK: Validate trade before execution ---
                        output["current_price"] = current_price
                        allowed, reason, output = risk_mgr.validate_trade(
                            output, state, initial_account_value or 0
                        )
                        if not allowed:
                            add_event(f"RISK BLOCKED {asset}: {reason}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "risk_blocked",
                                    "reason": reason,
                                    "original_alloc_usd": alloc_usd,
                                }) + "\n")
                            continue
                        # Use potentially adjusted values from risk manager
                        alloc_usd = float(output.get("allocation_usd", alloc_usd))
                        amount = alloc_usd / current_price

                        # Place market or limit order
                        order_type = output.get("order_type", "market")
                        limit_price = output.get("limit_price")

                        if order_type == "limit" and limit_price:
                            limit_price = float(limit_price)
                            if is_buy:
                                order = await hyperliquid.place_limit_buy(asset, amount, limit_price)
                            else:
                                order = await hyperliquid.place_limit_sell(asset, amount, limit_price)
                            add_event(f"LIMIT {action.upper()} {asset} amount {amount:.4f} at limit ${limit_price}")
                        else:
                            order = await hyperliquid.place_buy_order(asset, amount) if is_buy else await hyperliquid.place_sell_order(asset, amount)

                        # Confirm by checking recent fills for this asset shortly after placing
                        await asyncio.sleep(1)
                        fills_check = await hyperliquid.get_recent_fills(limit=10)
                        filled = False
                        for fc in reversed(fills_check):
                            try:
                                if (fc.get('coin') == asset or fc.get('asset') == asset):
                                    filled = True
                                    break
                            except Exception:
                                continue
                        trade_log.append({"type": action, "price": current_price, "amount": amount, "exit_plan": output["exit_plan"], "filled": filled})
                        tp_oid = None
                        sl_oid = None
                        if output.get("tp_price"):
                            tp_order = await hyperliquid.place_take_profit(asset, is_buy, amount, output["tp_price"])
                            tp_oids = hyperliquid.extract_oids(tp_order)
                            tp_oid = tp_oids[0] if tp_oids else None
                            add_event(f"TP placed {asset} at {output['tp_price']}")
                        if output.get("sl_price"):
                            sl_order = await hyperliquid.place_stop_loss(asset, is_buy, amount, output["sl_price"])
                            sl_oids = hyperliquid.extract_oids(sl_order)
                            sl_oid = sl_oids[0] if sl_oids else None
                            add_event(f"SL placed {asset} at {output['sl_price']}")
                        # Reconcile: if opposite-side position exists or TP/SL just filled, clear stale active_trades for this asset
                        for existing in active_trades[:]:
                            if existing.get('asset') == asset:
                                try:
                                    active_trades.remove(existing)
                                except ValueError:
                                    pass
                        active_trades.append({
                            "asset": asset,
                            "is_long": is_buy,
                            "amount": amount,
                            "entry_price": current_price,
                            "tp_oid": tp_oid,
                            "sl_oid": sl_oid,
                            "exit_plan": output["exit_plan"],
                            "opened_at": datetime.now().isoformat()
                        })
                        add_event(f"{action.upper()} {asset} amount {amount:.4f} at ~{current_price}")
                        if rationale:
                            add_event(f"Post-trade rationale for {asset}: {rationale}")
                        # Write to diary after confirming fills status
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": action,
                                "order_type": order_type,
                                "limit_price": limit_price,
                                "allocation_usd": alloc_usd,
                                "amount": amount,
                                "entry_price": current_price,
                                "tp_price": output.get("tp_price"),
                                "tp_oid": tp_oid,
                                "sl_price": output.get("sl_price"),
                                "sl_oid": sl_oid,
                                "exit_plan": output.get("exit_plan", ""),
                                "rationale": output.get("rationale", ""),
                                "order_result": str(order),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "filled": filled
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                    else:
                        add_event(f"Hold {asset}: {output.get('rationale', '')}")
                        # Write hold to diary
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "asset": asset,
                                "action": "hold",
                                "rationale": output.get("rationale", "")
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                except Exception as e:
                    import traceback
                    add_event(f"Execution error {asset}: {e}")
                    add_event(f"Traceback: {traceback.format_exc()}")

            # Paper trading: check trigger orders (TP/SL) and log performance
            if paper_mode and hasattr(hyperliquid, 'check_trigger_orders'):
                triggered = await hyperliquid.check_trigger_orders()
                if triggered:
                    for trig in triggered:
                        add_event(f"PAPER {trig.get('orderType','')} triggered: {trig.get('coin')} at ${trig.get('triggerPx')}")
                perf = hyperliquid.get_performance_summary()
                add_event(
                    f"PAPER PERFORMANCE: Balance ${perf['current_balance']}, "
                    f"PnL ${perf['realized_pnl']} ({perf['realized_pnl_pct']}%), "
                    f"Trades: {perf['total_trades']}, Open: {perf['open_positions']}"
                )

            await asyncio.sleep(get_interval_seconds(args.interval))

    async def handle_diary(request):
        """Return diary entries as JSON or newline-delimited text."""
        try:
            raw = request.query.get('raw')
            download = request.query.get('download')
            if raw or download:
                if not os.path.exists(diary_path):
                    return web.Response(text="", content_type="text/plain")
                with open(diary_path, "r") as f:
                    data = f.read()
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename=diary.jsonl"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(request.query.get('limit', '200'))
            with open(diary_path, "r") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            entries = [json.loads(l) for l in lines[start:]]
            return web.json_response({"entries": entries})
        except FileNotFoundError:
            return web.json_response({"entries": []})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path = request.query.get('path', 'llm_requests.log')
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_param) if limit_param else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/logs', handle_logs)

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        app = web.Application()
        await start_api(app)
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        await run_loop()

    def calculate_total_return(state, trade_log):
        """Compute percent return relative to an assumed initial balance."""
        initial = 10000
        current = state['balance'] + sum(p.get('pnl', 0) for p in state.get('positions', []))
        return ((current - initial) / initial) * 100 if initial else 0

    def calculate_sharpe(returns):
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    async def check_exit_condition(trade, hyperliquid_api):
        """Evaluate whether a given trade's exit plan triggers a close."""
        plan = (trade.get("exit_plan") or "").lower()
        if not plan:
            return False
        try:
            candles_4h = await hyperliquid_api.get_candles(trade["asset"], "4h", 60)
            indicators = compute_all(candles_4h)
            if "macd" in plan and "below" in plan:
                macd_val = latest(indicators.get("macd", []))
                threshold = float(plan.split("below")[-1].strip())
                return macd_val is not None and macd_val < threshold
            if "close above ema50" in plan:
                ema50_val = latest(indicators.get("ema50", []))
                current = await hyperliquid_api.get_current_price(trade["asset"])
                return ema50_val is not None and current > ema50_val
        except Exception:
            return False
        return False

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
