"""Paper trading simulator that mirrors the HyperliquidAPI interface.

Runs the full pipeline (research + quant signals + LLM decisions) but
executes trades in-memory instead of on-chain. No wallet, no testnet,
no real money. Perfect for validating strategy before going live.

Fetches REAL market data from Hyperliquid mainnet (prices, candles,
funding, OI) but simulates order execution locally.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timezone
from collections import defaultdict

from hyperliquid.info import Info
from hyperliquid.utils import constants

logger = logging.getLogger(__name__)


class PaperTrader:
    """Drop-in replacement for HyperliquidAPI that simulates trades locally.

    Uses real Hyperliquid mainnet data for prices/candles but tracks
    positions, PnL, and orders in-memory.
    """

    # Hyperliquid fee schedule (realistic)
    TAKER_FEE = 0.00035   # 0.035% taker fee
    MAKER_FEE = 0.0002    # 0.02% maker rebate (we charge, not rebate, for safety)
    MARKET_SLIPPAGE = 0.001  # 0.1% simulated slippage on market orders

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.total_fees_paid = 0.0
        self.positions: dict[str, dict] = {}  # coin -> {size, entry_price, side}
        self.open_orders: list[dict] = []
        self.fills: list[dict] = []
        self.order_counter = 0
        self._meta_cache = None
        self._hip3_meta_cache = {}

        # Real Hyperliquid mainnet info client for market data
        self.info = Info(constants.MAINNET_API_URL)
        # Dummy values for interface compatibility
        self.account_address = "0xPAPER_TRADER"
        self.query_address = "0xPAPER_TRADER"

        logger.info(
            "PAPER TRADER initialized with $%.2f balance. "
            "Using REAL mainnet prices, simulated execution.",
            initial_balance
        )

    def round_size(self, asset, amount):
        """Round order size using metadata from mainnet."""
        meta = self._meta_cache[0] if self._meta_cache else None
        if meta:
            universe = meta.get("universe", [])
            asset_info = next((u for u in universe if u.get("name") == asset), None)
            if asset_info:
                decimals = asset_info.get("szDecimals", 8)
                return round(amount, decimals)
        return round(amount, 8)

    def _next_oid(self) -> int:
        self.order_counter += 1
        return self.order_counter

    def _record_fill(self, asset: str, is_buy: bool, amount: float, price: float):
        """Record a simulated fill with correct PnL for longs AND shorts.

        Key logic:
        - ADDING to position (buy on long, sell on short) → average entry price
        - REDUCING position (sell on long, buy on short) → realize PnL on closed portion
        - CLOSING position (size goes to zero) → realize full PnL
        - FLIPPING direction → close old, open new remainder at current price
        """
        if price <= 0:
            logger.error("PAPER: Rejecting fill for %s — price is $%.2f", asset, price)
            return

        fill = {
            "coin": asset,
            "isBuy": is_buy,
            "sz": str(amount),
            "px": str(price),
            "time": int(time.time() * 1000),
            "oid": self._next_oid(),
        }
        self.fills.append(fill)

        # Update position
        if asset in self.positions:
            pos = self.positions[asset]
            old_size = pos["size"]
            if is_buy:
                new_size = old_size + amount
            else:
                new_size = old_size - amount

            # Determine if we're adding or reducing
            adding = (old_size > 0 and is_buy) or (old_size < 0 and not is_buy)

            if abs(new_size) < 1e-10:
                # Fully closed
                closed_amount = abs(old_size)
                if old_size > 0:
                    pnl = (price - pos["entry_price"]) * closed_amount
                else:
                    pnl = (pos["entry_price"] - price) * closed_amount
                self.balance += pnl
                del self.positions[asset]
                logger.info(
                    "PAPER: Closed %s %s position. PnL: $%.2f. Balance: $%.2f",
                    asset, "LONG" if old_size > 0 else "SHORT", pnl, self.balance
                )
            elif adding:
                # Adding to position in same direction → average entry
                total_cost = pos["entry_price"] * abs(old_size) + price * amount
                pos["entry_price"] = total_cost / abs(new_size)
                pos["size"] = new_size
                logger.info(
                    "PAPER: Added to %s %s. New avg entry: $%.2f, size: %.6f",
                    asset, "LONG" if new_size > 0 else "SHORT",
                    pos["entry_price"], new_size
                )
            elif (old_size > 0 and new_size > 0) or (old_size < 0 and new_size < 0):
                # Reducing position (partial close) — realize PnL on closed portion
                closed_amount = amount
                if old_size > 0:
                    pnl = (price - pos["entry_price"]) * closed_amount
                else:
                    pnl = (pos["entry_price"] - price) * closed_amount
                self.balance += pnl
                pos["size"] = new_size
                # Entry price stays the same for remaining position
                logger.info(
                    "PAPER: Partially closed %s. Realized PnL: $%.2f. Remaining: %.6f @ $%.2f",
                    asset, pnl, new_size, pos["entry_price"]
                )
            else:
                # Direction flipped — close old position fully, open new at current price
                closed_amount = abs(old_size)
                if old_size > 0:
                    pnl = (price - pos["entry_price"]) * closed_amount
                else:
                    pnl = (pos["entry_price"] - price) * closed_amount
                self.balance += pnl
                remainder = abs(new_size)
                pos["entry_price"] = price
                pos["size"] = new_size
                logger.info(
                    "PAPER: Flipped %s. Closed PnL: $%.2f. New %s %.6f @ $%.2f",
                    asset, pnl, "LONG" if new_size > 0 else "SHORT",
                    remainder, price
                )
        else:
            # New position
            self.positions[asset] = {
                "size": amount if is_buy else -amount,
                "entry_price": price,
            }

        # Deduct trading fee from balance
        notional = amount * price
        fee = notional * self.TAKER_FEE
        self.balance -= fee
        self.total_fees_paid += fee
        logger.info(
            "PAPER FILL: %s %s %.6f @ $%.2f (notional $%.2f, fee $%.4f)",
            "BUY" if is_buy else "SELL", asset, amount, price, notional, fee
        )

    # ------------------------------------------------------------------
    # HyperliquidAPI-compatible interface
    # ------------------------------------------------------------------

    async def place_buy_order(self, asset, amount, slippage=0.01):
        amount = self.round_size(asset, amount)
        price = await self.get_current_price(asset)
        if not price or price <= 0:
            logger.error("PAPER: Cannot buy %s — no valid price available", asset)
            return {"status": "error", "response": {"data": {"statuses": [
                {"error": "no_price_available"}
            ]}}}
        fill_price = price * (1 + self.MARKET_SLIPPAGE)  # Buyer pays above mid
        self._record_fill(asset, True, amount, fill_price)
        return {"status": "ok", "response": {"data": {"statuses": [
            {"filled": {"oid": self.order_counter}}
        ]}}}

    async def place_sell_order(self, asset, amount, slippage=0.01):
        amount = self.round_size(asset, amount)
        price = await self.get_current_price(asset)
        if not price or price <= 0:
            logger.error("PAPER: Cannot sell %s — no valid price available", asset)
            return {"status": "error", "response": {"data": {"statuses": [
                {"error": "no_price_available"}
            ]}}}
        fill_price = price * (1 - self.MARKET_SLIPPAGE)  # Seller gets below mid
        self._record_fill(asset, False, amount, fill_price)
        return {"status": "ok", "response": {"data": {"statuses": [
            {"filled": {"oid": self.order_counter}}
        ]}}}

    async def place_limit_buy(self, asset, amount, limit_price, tif="Gtc"):
        amount = self.round_size(asset, amount)
        price = await self.get_current_price(asset)
        # Fill if price is at or below limit, OR within 0.5% (realistic fill in ranging market)
        if price <= limit_price or (price - limit_price) / price < 0.005:
            fill_price = min(price, limit_price)
            self._record_fill(asset, True, amount, fill_price)
            return {"status": "ok", "response": {"data": {"statuses": [
                {"filled": {"oid": self._next_oid()}}
            ]}}}
        else:
            # Resting order
            oid = self._next_oid()
            self.open_orders.append({
                "coin": asset, "oid": oid, "isBuy": True,
                "sz": str(amount), "px": str(limit_price),
                "orderType": "Limit", "triggerPx": None,
            })
            return {"status": "ok", "response": {"data": {"statuses": [
                {"resting": {"oid": oid}}
            ]}}}

    async def place_limit_sell(self, asset, amount, limit_price, tif="Gtc"):
        amount = self.round_size(asset, amount)
        price = await self.get_current_price(asset)
        # Fill if price is at or above limit, OR within 0.5%
        if price >= limit_price or (limit_price - price) / price < 0.005:
            fill_price = max(price, limit_price)
            self._record_fill(asset, False, amount, fill_price)
            return {"status": "ok", "response": {"data": {"statuses": [
                {"filled": {"oid": self._next_oid()}}
            ]}}}
        else:
            oid = self._next_oid()
            self.open_orders.append({
                "coin": asset, "oid": oid, "isBuy": False,
                "sz": str(amount), "px": str(limit_price),
                "orderType": "Limit", "triggerPx": None,
            })
            return {"status": "ok", "response": {"data": {"statuses": [
                {"resting": {"oid": oid}}
            ]}}}

    async def place_take_profit(self, asset, is_buy, amount, tp_price):
        amount = self.round_size(asset, amount)
        oid = self._next_oid()
        self.open_orders.append({
            "coin": asset, "oid": oid, "isBuy": not is_buy,
            "sz": str(amount), "px": str(tp_price),
            "orderType": "Take Profit", "triggerPx": str(tp_price),
        })
        logger.info("PAPER: TP order placed for %s at $%.2f", asset, tp_price)
        return {"status": "ok", "response": {"data": {"statuses": [
            {"resting": {"oid": oid}}
        ]}}}

    async def place_stop_loss(self, asset, is_buy, amount, sl_price):
        amount = self.round_size(asset, amount)
        oid = self._next_oid()
        self.open_orders.append({
            "coin": asset, "oid": oid, "isBuy": not is_buy,
            "sz": str(amount), "px": str(sl_price),
            "orderType": "Stop Loss", "triggerPx": str(sl_price),
        })
        logger.info("PAPER: SL order placed for %s at $%.2f", asset, sl_price)
        return {"status": "ok", "response": {"data": {"statuses": [
            {"resting": {"oid": oid}}
        ]}}}

    async def cancel_order(self, asset, oid):
        self.open_orders = [o for o in self.open_orders if o.get("oid") != oid]
        return {"status": "ok"}

    async def cancel_all_orders(self, asset):
        count = len([o for o in self.open_orders if o.get("coin") == asset])
        self.open_orders = [o for o in self.open_orders if o.get("coin") != asset]
        return {"status": "ok", "cancelled_count": count}

    async def get_open_orders(self):
        return list(self.open_orders)

    async def get_recent_fills(self, limit: int = 50):
        return self.fills[-limit:]

    def extract_oids(self, order_result):
        oids = []
        try:
            statuses = order_result["response"]["data"]["statuses"]
            for st in statuses:
                if "resting" in st and "oid" in st["resting"]:
                    oids.append(st["resting"]["oid"])
                if "filled" in st and "oid" in st["filled"]:
                    oids.append(st["filled"]["oid"])
        except (KeyError, TypeError):
            pass
        return oids

    async def get_user_state(self):
        """Compute simulated account state from positions + real prices."""
        enriched = []
        total_pnl = 0

        for coin, pos in self.positions.items():
            current_price = await self.get_current_price(coin)
            if not current_price or current_price <= 0:
                current_price = pos["entry_price"]  # Fallback to entry if price unavailable
            size = pos["size"]
            entry = pos["entry_price"]
            pnl = (current_price - entry) * size  # Works for both long/short
            total_pnl += pnl

            enriched.append({
                "coin": coin,
                "szi": str(size),
                "entryPx": str(entry),
                "liquidationPx": None,
                "leverage": {"type": "cross"},
                "pnl": pnl,
                "notional_entry": abs(size) * entry,
            })

        total_value = self.balance + total_pnl
        return {
            "balance": self.balance,
            "total_value": total_value,
            "positions": enriched,
        }

    async def get_current_price(self, asset):
        """Get REAL price from Hyperliquid mainnet."""
        try:
            mids = await asyncio.to_thread(self.info.all_mids)
            return float(mids.get(asset, 0.0))
        except Exception as e:
            logger.error("Price fetch error for %s: %s", asset, e)
            return 0.0

    async def get_meta_and_ctxs(self, dex=None):
        if dex:
            if dex not in self._hip3_meta_cache:
                response = await asyncio.to_thread(
                    self.info.post, "/info", {"type": "metaAndAssetCtxs", "dex": dex}
                )
                if isinstance(response, list) and len(response) >= 2:
                    self._hip3_meta_cache[dex] = response
            return self._hip3_meta_cache.get(dex)
        if not self._meta_cache:
            self._meta_cache = await asyncio.to_thread(self.info.meta_and_asset_ctxs)
        return self._meta_cache

    async def get_open_interest(self, asset):
        """Get REAL OI from mainnet."""
        try:
            dex = asset.split(":")[0] if ":" in asset else None
            data = await self.get_meta_and_ctxs(dex=dex)
            if isinstance(data, list) and len(data) >= 2:
                meta, ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                idx = next((i for i, u in enumerate(universe) if u.get("name") == asset), None)
                if idx is not None and idx < len(ctxs):
                    oi = ctxs[idx].get("openInterest")
                    return round(float(oi), 2) if oi else None
        except Exception as e:
            logger.error("OI fetch error for %s: %s", asset, e)
        return None

    async def get_candles(self, asset, interval="5m", count=100):
        """Get REAL candles from mainnet."""
        interval_ms_map = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
            "1d": 86_400_000,
        }
        interval_ms = interval_ms_map.get(interval, 300_000)
        end_time = int(time.time() * 1000)
        start_time = end_time - (count * interval_ms)

        try:
            raw = await asyncio.to_thread(
                self.info.candles_snapshot, asset, interval, start_time, end_time
            )
            return [
                {
                    "t": c.get("t"),
                    "open": float(c.get("o", 0)),
                    "high": float(c.get("h", 0)),
                    "low": float(c.get("l", 0)),
                    "close": float(c.get("c", 0)),
                    "volume": float(c.get("v", 0)),
                }
                for c in raw
            ]
        except Exception as e:
            logger.error("Candle fetch error for %s: %s", asset, e)
            return []

    async def get_funding_rate(self, asset):
        """Get REAL funding rate from mainnet."""
        try:
            dex = asset.split(":")[0] if ":" in asset else None
            data = await self.get_meta_and_ctxs(dex=dex)
            if isinstance(data, list) and len(data) >= 2:
                meta, ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                idx = next((i for i, u in enumerate(universe) if u.get("name") == asset), None)
                if idx is not None and idx < len(ctxs):
                    funding = ctxs[idx].get("funding")
                    return round(float(funding), 8) if funding else None
        except Exception as e:
            logger.error("Funding fetch error for %s: %s", asset, e)
        return None

    async def check_trigger_orders(self):
        """Check if any resting limit orders OR TP/SL trigger orders should fire."""
        triggered = []
        for order in self.open_orders[:]:
            coin = order["coin"]
            price = await self.get_current_price(coin)

            if price <= 0:
                continue

            is_buy = order["isBuy"]
            order_type = order.get("orderType", "")
            should_trigger = False

            if order.get("triggerPx") is not None:
                # TP/SL trigger orders
                trigger = float(order["triggerPx"])
                if "Take Profit" in order_type:
                    if is_buy and price <= trigger:  # Short TP: price dropped to target
                        should_trigger = True
                    elif not is_buy and price >= trigger:  # Long TP: price rose to target
                        should_trigger = True
                elif "Stop Loss" in order_type:
                    if is_buy and price >= trigger:  # Short SL: price rose above stop
                        should_trigger = True
                    elif not is_buy and price <= trigger:  # Long SL: price dropped below stop
                        should_trigger = True
            elif order_type == "Limit":
                # Resting limit orders — check if price crossed the limit
                limit_px = float(order["px"])
                if is_buy and price <= limit_px:
                    # Price dropped to or below limit buy price → fill
                    should_trigger = True
                elif not is_buy and price >= limit_px:
                    # Price rose to or above limit sell price → fill
                    should_trigger = True

            if should_trigger:
                amount = float(order["sz"])
                fill_price = price  # Fill at current market price (realistic)
                if order_type == "Limit":
                    # Limit orders fill at the limit price (price improvement)
                    fill_price = float(order["px"])
                self._record_fill(coin, is_buy, amount, fill_price)
                self.open_orders.remove(order)
                triggered.append(order)
                logger.info(
                    "PAPER: %s triggered for %s at $%.2f (market $%.2f)",
                    order_type or "Order", coin, fill_price, price
                )

        return triggered

    def get_performance_summary(self) -> dict:
        """Get overall paper trading performance."""
        total_pnl = self.balance - self.initial_balance
        return {
            "initial_balance": self.initial_balance,
            "current_balance": round(self.balance, 2),
            "realized_pnl": round(total_pnl, 2),
            "realized_pnl_pct": round(total_pnl / self.initial_balance * 100, 2),
            "total_fees_paid": round(self.total_fees_paid, 4),
            "total_trades": len(self.fills),
            "open_positions": len(self.positions),
            "open_orders": len(self.open_orders),
        }
