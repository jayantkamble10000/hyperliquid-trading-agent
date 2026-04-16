"""Centralized risk management for the trading agent.

All safety guards are enforced here, independent of LLM decisions.
The LLM cannot override these limits — they are hard-coded checks
applied before every trade execution.
"""

import logging
from datetime import datetime, timezone

from src.config_loader import CONFIG


class RiskManager:
    """Enforces risk limits on every trade before execution."""

    def __init__(self):
        self.max_position_pct = float(CONFIG.get("max_position_pct") or 10)
        self.max_loss_per_position_pct = float(CONFIG.get("max_loss_per_position_pct") or 20)
        self.max_leverage = float(CONFIG.get("max_leverage") or 10)
        self.max_total_exposure_pct = float(CONFIG.get("max_total_exposure_pct") or 50)
        self.daily_loss_circuit_breaker_pct = float(CONFIG.get("daily_loss_circuit_breaker_pct") or 10)
        self.mandatory_sl_pct = float(CONFIG.get("mandatory_sl_pct") or 5)
        self.max_concurrent_positions = int(CONFIG.get("max_concurrent_positions") or 10)
        self.min_balance_reserve_pct = float(CONFIG.get("min_balance_reserve_pct") or 20)

        # Daily tracking
        self.daily_high_value = None
        self.daily_high_date = None
        self.circuit_breaker_active = False
        self.circuit_breaker_date = None

    def _reset_daily_if_needed(self, account_value: float):
        """Reset daily high watermark at UTC day boundary."""
        today = datetime.now(timezone.utc).date()
        if self.daily_high_date != today:
            self.daily_high_value = account_value
            self.daily_high_date = today
            self.circuit_breaker_active = False
            self.circuit_breaker_date = None
        elif account_value > self.daily_high_value:
            self.daily_high_value = account_value

    # ------------------------------------------------------------------
    # Individual checks — each returns (allowed: bool, reason: str)
    # ------------------------------------------------------------------

    def check_position_size(self, alloc_usd: float, account_value: float) -> tuple[bool, str]:
        """Single position cannot exceed max_position_pct of account."""
        if account_value <= 0:
            return False, "Account value is zero or negative"
        max_alloc = account_value * (self.max_position_pct / 100.0)
        if alloc_usd > max_alloc:
            return False, (
                f"Allocation ${alloc_usd:.2f} exceeds {self.max_position_pct}% "
                f"of account (${max_alloc:.2f})"
            )
        return True, ""

    def check_total_exposure(self, positions: list[dict], new_alloc: float,
                              account_value: float) -> tuple[bool, str]:
        """Sum of all position notionals + new allocation cannot exceed max_total_exposure_pct."""
        current_exposure = 0.0
        for pos in positions:
            qty = abs(float(pos.get("quantity") or pos.get("szi") or 0))
            entry = float(pos.get("entry_price") or pos.get("entryPx") or 0)
            current_exposure += qty * entry
        total = current_exposure + new_alloc
        max_exposure = account_value * (self.max_total_exposure_pct / 100.0)
        if total > max_exposure:
            return False, (
                f"Total exposure ${total:.2f} would exceed {self.max_total_exposure_pct}% "
                f"of account (${max_exposure:.2f})"
            )
        return True, ""

    def check_leverage(self, alloc_usd: float, balance: float) -> tuple[bool, str]:
        """Effective leverage of new trade cannot exceed max_leverage."""
        if balance <= 0:
            return False, "Balance is zero or negative"
        effective_lev = alloc_usd / balance
        if effective_lev > self.max_leverage:
            return False, (
                f"Effective leverage {effective_lev:.1f}x exceeds max {self.max_leverage}x"
            )
        return True, ""

    def check_daily_drawdown(self, account_value: float) -> tuple[bool, str]:
        """Activate circuit breaker if account drops max % from daily high."""
        self._reset_daily_if_needed(account_value)
        if self.circuit_breaker_active:
            return False, "Daily loss circuit breaker is active — no new trades until tomorrow (UTC)"
        if self.daily_high_value and self.daily_high_value > 0:
            drawdown_pct = ((self.daily_high_value - account_value) / self.daily_high_value) * 100
            if drawdown_pct >= self.daily_loss_circuit_breaker_pct:
                self.circuit_breaker_active = True
                self.circuit_breaker_date = datetime.now(timezone.utc).date()
                return False, (
                    f"Daily drawdown {drawdown_pct:.2f}% exceeds circuit breaker "
                    f"threshold of {self.daily_loss_circuit_breaker_pct}%"
                )
        return True, ""

    def check_concurrent_positions(self, current_count: int) -> tuple[bool, str]:
        """Limit number of simultaneous open positions."""
        if current_count >= self.max_concurrent_positions:
            return False, (
                f"Already at max concurrent positions ({self.max_concurrent_positions})"
            )
        return True, ""

    def check_balance_reserve(self, balance: float, initial_balance: float) -> tuple[bool, str]:
        """Don't trade if balance falls below reserve threshold."""
        if initial_balance <= 0:
            return True, ""
        min_balance = initial_balance * (self.min_balance_reserve_pct / 100.0)
        if balance < min_balance:
            return False, (
                f"Balance ${balance:.2f} below minimum reserve "
                f"${min_balance:.2f} ({self.min_balance_reserve_pct}% of initial)"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Stop-loss enforcement
    # ------------------------------------------------------------------

    def enforce_stop_loss(self, sl_price: float | None, entry_price: float,
                           is_buy: bool) -> float:
        """Ensure every trade has a stop-loss within mandatory distance.

        If SL is missing: auto-set at mandatory_sl_pct from entry.
        If SL is provided but too far: cap at mandatory_sl_pct distance.
        """
        max_sl_distance = entry_price * (self.mandatory_sl_pct / 100.0)

        if sl_price is None:
            # No SL provided — auto-set at mandatory distance
            if is_buy:
                return round(entry_price - max_sl_distance, 2)
            else:
                return round(entry_price + max_sl_distance, 2)

        # SL provided — validate it's not too far from entry
        if is_buy:
            actual_distance = entry_price - sl_price
            if actual_distance > max_sl_distance:
                capped_sl = round(entry_price - max_sl_distance, 2)
                logging.warning(
                    "RISK: SL %.2f is %.1f%% from entry (max %.1f%%). Capping to %.2f",
                    sl_price, (actual_distance / entry_price) * 100,
                    self.mandatory_sl_pct, capped_sl
                )
                return capped_sl
        else:
            actual_distance = sl_price - entry_price
            if actual_distance > max_sl_distance:
                capped_sl = round(entry_price + max_sl_distance, 2)
                logging.warning(
                    "RISK: SL %.2f is %.1f%% from entry (max %.1f%%). Capping to %.2f",
                    sl_price, (actual_distance / entry_price) * 100,
                    self.mandatory_sl_pct, capped_sl
                )
                return capped_sl

        return sl_price

    # ------------------------------------------------------------------
    # Force-close losing positions
    # ------------------------------------------------------------------

    def check_losing_positions(self, positions: list[dict]) -> list[dict]:
        """Return positions that should be force-closed due to excessive loss.

        Args:
            positions: List of position dicts with keys:
                coin/symbol, szi/quantity, entryPx/entry_price,
                pnl/unrealized_pnl

        Returns:
            List of positions that exceed the max loss threshold.
        """
        to_close = []
        for pos in positions:
            coin = pos.get("coin") or pos.get("symbol")
            entry_px = float(pos.get("entryPx") or pos.get("entry_price") or 0)
            size = float(pos.get("szi") or pos.get("quantity") or 0)
            pnl = float(pos.get("pnl") or pos.get("unrealized_pnl") or 0)

            if entry_px == 0 or size == 0:
                continue

            notional = abs(size) * entry_px
            if notional == 0:
                continue

            loss_pct = abs(pnl / notional) * 100 if pnl < 0 else 0

            if loss_pct >= self.max_loss_per_position_pct:
                logging.warning(
                    "RISK: Force-closing %s — loss %.2f%% exceeds max %.2f%%",
                    coin, loss_pct, self.max_loss_per_position_pct
                )
                to_close.append({
                    "coin": coin,
                    "size": abs(size),
                    "is_long": size > 0,
                    "loss_pct": round(loss_pct, 2),
                    "pnl": round(pnl, 2),
                })
        return to_close

    # ------------------------------------------------------------------
    # Signal-quality gate for new entries (pre-LLM in effect)
    # ------------------------------------------------------------------

    def check_signal_quality(
        self,
        asset: str,
        quant_signal: dict,
        anti_paralysis_active: bool,
        min_composite_abs: float = 0.2,
    ) -> tuple[bool, str]:
        """Reject low-conviction new entries. Anti-paralysis bypasses the gate.

        Rejects entries where confidence_label is 'low' AND |composite_score| < min_composite_abs.
        This catches the weakest signals while letting moderate/high-conviction trades through.
        """
        if anti_paralysis_active:
            return True, ""  # anti-paralysis overrides — must enter something
        if not quant_signal:
            return True, ""  # no signal data — let LLM call through (backward-compat)
        confidence = str(quant_signal.get("confidence_label") or "low").lower()
        composite = abs(float(quant_signal.get("composite_score") or 0.0))
        if confidence == "low" and composite < min_composite_abs:
            return False, (
                f"Signal too weak for {asset}: confidence={confidence}, "
                f"|composite|={composite:.3f} < {min_composite_abs}"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Mandatory partial scale-out on winning positions (hard-coded)
    # ------------------------------------------------------------------

    def check_mandatory_scale_outs(
        self,
        positions: list[dict],
        cycle_counts_by_coin: dict,
        already_scaled: set,
        min_unrealized_pct: float = 0.5,
        min_cycles_held: int = 10,
    ) -> list[dict]:
        """Return positions that must be partially closed (50%) due to profit trigger.

        Hard-coded, runs post-LLM, cannot be overridden by the LLM.
        Trigger: unrealized_pct >= min_unrealized_pct AND cycles_held >= min_cycles_held
                 AND position not already scaled out.

        After scale-out the caller sets SL on the remaining 50% to breakeven.
        """
        to_scale = []
        for pos in positions:
            coin = pos.get("coin") or pos.get("symbol")
            if not coin or coin in already_scaled:
                continue
            entry_px = float(pos.get("entryPx") or pos.get("entry_price") or 0)
            size = float(pos.get("szi") or pos.get("quantity") or 0)
            pnl = float(pos.get("pnl") or pos.get("unrealized_pnl") or 0)
            if entry_px == 0 or size == 0:
                continue
            notional = abs(size) * entry_px
            if notional == 0:
                continue
            unrealized_pct = (pnl / notional) * 100
            cycles_held = cycle_counts_by_coin.get(coin, 0)
            if unrealized_pct >= min_unrealized_pct and cycles_held >= min_cycles_held:
                logging.warning(
                    "RISK: Mandatory scale-out %s — +%.2f%% unrealized after %d cycles",
                    coin, unrealized_pct, cycles_held
                )
                to_scale.append({
                    "coin": coin,
                    "size": abs(size) * 0.5,
                    "is_long": size > 0,
                    "breakeven_sl": round(entry_px, 2),
                    "unrealized_pct": round(unrealized_pct, 2),
                    "cycles_held": cycles_held,
                    "pnl": round(pnl, 2),
                })
        return to_scale

    # ------------------------------------------------------------------
    # Correlation-aware exposure check (adapted from IMC Prosperity)
    # ------------------------------------------------------------------

    CORRELATED_GROUPS = [
        # Assets that tend to move together - treat as single exposure bucket
        {"group": "major_crypto", "assets": {"BTC", "ETH", "SOL", "AVAX", "DOT", "MATIC", "LINK"}},
        {"group": "meme_crypto", "assets": {"DOGE", "SHIB", "PEPE", "WIF", "BONK"}},
        {"group": "defi_crypto", "assets": {"UNI", "AAVE", "MKR", "CRV", "SUSHI"}},
    ]

    def check_correlated_exposure(
        self, asset: str, new_alloc: float, positions: list[dict], account_value: float,
        max_group_exposure_pct: float = 50.0
    ) -> tuple[bool, str]:
        """Check that correlated assets don't collectively exceed safe exposure.

        Adapted from IMC Prosperity: their basket strategy tracked net exposure
        across correlated components. We apply the same principle to crypto groups.
        """
        # Find which group this asset belongs to
        asset_group = None
        for group in self.CORRELATED_GROUPS:
            if asset in group["assets"]:
                asset_group = group
                break

        if asset_group is None:
            return True, ""  # Uncorrelated asset, no group check needed

        # Sum existing exposure in this group
        group_exposure = 0.0
        for pos in positions:
            pos_coin = pos.get("coin") or pos.get("symbol") or ""
            if pos_coin in asset_group["assets"]:
                qty = abs(float(pos.get("szi") or pos.get("quantity") or 0))
                entry = float(pos.get("entryPx") or pos.get("entry_price") or 0)
                group_exposure += qty * entry

        total_group = group_exposure + new_alloc
        max_group = account_value * (max_group_exposure_pct / 100.0)

        if total_group > max_group:
            return False, (
                f"Correlated group '{asset_group['group']}' exposure ${total_group:.2f} "
                f"would exceed {max_group_exposure_pct}% of account (${max_group:.2f}). "
                f"Existing group exposure: ${group_exposure:.2f}"
            )
        return True, ""

    def validate_trade(self, trade: dict, account_state: dict,
                        initial_balance: float) -> tuple[bool, str, dict]:
        """Run all safety checks on a proposed trade.

        Args:
            trade: LLM trade decision with keys:
                asset, action, allocation_usd, tp_price, sl_price
            account_state: Current account with keys:
                balance, total_value, positions
            initial_balance: Starting balance for reserve check

        Returns:
            (allowed, reason, adjusted_trade)
            adjusted_trade may have modified sl_price if it was missing.
        """
        action = trade.get("action", "hold")
        if action == "hold":
            return True, "", trade

        alloc_usd = float(trade.get("allocation_usd", 0))
        if alloc_usd <= 0:
            return False, "Zero or negative allocation", trade

        # Hyperliquid minimum order size is $10
        if alloc_usd < 11.0:
            alloc_usd = 11.0
            trade = {**trade, "allocation_usd": alloc_usd}
            logging.info("RISK: Bumped allocation to $11 (Hyperliquid $10 minimum)")

        account_value = float(account_state.get("total_value", 0))
        balance = float(account_state.get("balance", 0))
        positions = account_state.get("positions", [])
        is_buy = action == "buy"
        asset = trade.get("asset", "")

        # 1. Daily drawdown circuit breaker
        ok, reason = self.check_daily_drawdown(account_value)
        if not ok:
            return False, reason, trade

        # 2. Balance reserve
        ok, reason = self.check_balance_reserve(balance, initial_balance)
        if not ok:
            return False, reason, trade

        # 3. Position size limit
        ok, reason = self.check_position_size(alloc_usd, account_value)
        if not ok:
            # Cap allocation instead of rejecting
            max_alloc = account_value * (self.max_position_pct / 100.0)
            # But never below Hyperliquid's $10 minimum
            if max_alloc < 11.0:
                max_alloc = 11.0
            logging.warning("RISK: Capping allocation from $%.2f to $%.2f", alloc_usd, max_alloc)
            alloc_usd = max_alloc
            trade = {**trade, "allocation_usd": alloc_usd}

        # 4. Total exposure
        ok, reason = self.check_total_exposure(positions, alloc_usd, account_value)
        if not ok:
            return False, reason, trade

        # 5. Leverage check
        ok, reason = self.check_leverage(alloc_usd, balance)
        if not ok:
            return False, reason, trade

        # 6. Concurrent positions
        active_count = sum(
            1 for p in positions
            if abs(float(p.get("szi") or p.get("quantity") or 0)) > 0
        )
        ok, reason = self.check_concurrent_positions(active_count)
        if not ok:
            return False, reason, trade

        # 7. Correlated exposure check (NEW - IMC Prosperity inspired)
        ok, reason = self.check_correlated_exposure(asset, alloc_usd, positions, account_value)
        if not ok:
            # Cap to remaining group budget instead of blocking
            logging.warning("RISK: Correlated exposure warning for %s: %s", asset, reason)
            # Reduce allocation by 50% as compromise
            alloc_usd = alloc_usd * 0.5
            if alloc_usd < 11.0:
                return False, reason, trade
            trade = {**trade, "allocation_usd": alloc_usd}
            logging.warning("RISK: Reduced allocation to $%.2f due to correlated exposure", alloc_usd)

        # 8. Enforce mandatory stop-loss
        current_price = float(trade.get("current_price", 0))
        entry_price = current_price if current_price > 0 else 1.0
        sl_price = trade.get("sl_price")
        enforced_sl = self.enforce_stop_loss(sl_price, entry_price, is_buy)
        if sl_price is None:
            logging.info("RISK: Auto-setting SL at %.2f (%.1f%% from entry)",
                        enforced_sl, self.mandatory_sl_pct)
        trade = {**trade, "sl_price": enforced_sl}

        return True, "", trade

    def get_risk_summary(self) -> dict:
        """Return current risk parameters for inclusion in LLM context."""
        return {
            "max_position_pct": self.max_position_pct,
            "max_loss_per_position_pct": self.max_loss_per_position_pct,
            "max_leverage": self.max_leverage,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "daily_loss_circuit_breaker_pct": self.daily_loss_circuit_breaker_pct,
            "mandatory_sl_pct": self.mandatory_sl_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "min_balance_reserve_pct": self.min_balance_reserve_pct,
            "circuit_breaker_active": self.circuit_breaker_active,
            "correlated_groups": [g["group"] for g in self.CORRELATED_GROUPS],
        }
