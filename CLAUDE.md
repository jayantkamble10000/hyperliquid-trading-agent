# CLAUDE.md — Hyperliquid Paper-Trading Agent

> **Purpose of this file:** Claude Code auto-loads `CLAUDE.md` when you open the
> project. It is a handoff document so a new chat can pick up where the last one
> left off without re-deriving everything. Keep it terse and factual. Update it
> at the end of any material design change or run.

---

## 1. Project goal and constraints

- Paper-trading bot for **Hyperliquid perps** (real prices, simulated fills).
- **Primary objective:** minimize risk, capture small consistent profits. Not a
  moonshot bot — guardrails come first.
- Paper-first. No live capital until paper shows a positive, boring equity curve
  over multiple multi-hour runs in different market regimes.
- The owner iterates by running multi-hour windows, reading diary/decision
  logs, and hardening rules that the model abused or that the market exposed.

## 2. Architecture (three layers, strict separation)

```
 Free Research   →   Quant Signals   →   Claude LLM (Haiku 4.5)   →   Risk Manager   →   PaperTrader
 (news/macro)        (z-score, ADX,      (JSON decisions)              (hard rules)       (sim fills)
                      Bollinger %B)
```

- **Research:** `src/research/` — pulls free context (news, macro) used only as
  narrative input to the LLM.
- **Quant signals:** `src/indicators/`, `src/strategies/` — produce a
  `composite_score` in [-1, 1] and a `confidence_label` in {low, medium, high}
  per asset per cycle.
- **LLM (decision maker):** `src/agent/decision_maker.py` — Haiku 4.5
  (`claude-haiku-4-5-20251001`). Emits JSON trade decisions. It is **not**
  trusted to enforce risk rules; it is trusted to *propose* trades.
- **Risk manager:** `src/risk_manager.py` — hard-coded, deterministic.
  Everything safety-critical lives here (SL cap, scale-out, signal quality,
  correlated exposure, min-notional, etc.).
- **Paper trader:** `src/trading/` — simulates fills against live prices.
- **Orchestrator:** `src/main.py` — wires the cycle: research → signals →
  LLM → risk manager → paper trader, writes diary + decisions logs.

**Design principle learned the hard way:** prompt-based rules are *leaky*.
Haiku will co-opt labels (see Run 11 below). If a rule is safety-critical,
hard-code it in `risk_manager.py`. The prompt only *tells the LLM* about the
rule so its proposals don't fight automation.

## 3. Models

- Trading model: **Haiku 4.5** (`claude-haiku-4-5-20251001`).
- Sanitizer / overload fallback: also Haiku.
- **Opus 4.6 question:** considered and rejected for now. Haiku is ~12× cheaper
  per call, and the bot's losses in early runs were from *rule design*, not
  model reasoning. Swapping to Opus doesn't fix leaky prompts or flat markets.
  Revisit once rules are stable and costs are negligible.

## 4. Run history (what we learned)

Each run writes `diary_run<N>.jsonl` (events + rationale) and
`decisions_run<N>.jsonl` (raw LLM outputs). These are excluded from git but
kept locally for post-mortems.

| Run | Window | What was tested | Outcome / lesson |
|-----|--------|-----------------|------------------|
| 1–7 | various | Baseline cycles, plumbing fixes | Infra hardening — HIP-3 round_size, risk_manager min order, unified accounts, agent wallet auth. |
| 8   | 13h    | Long window with original TP/SL | ~98.6% hold rate — TP/SL too wide for the window, enter-and-hold-forever. |
| 9   | 40m    | Short smoke test | Confirmed plumbing after fixes. |
| 10  | 3h     | Post-fix behavior check | Still too passive; model question raised (Haiku vs Opus). |
| 11  | 4h     | **Prompt-based scale-out rule** | **FAILED.** Haiku co-opted the "Mandatory scale-out" label to justify selling an underwater SOL position. Proved prompt rules are unsafe for money-moving logic. |
| 12  | 4h     | **Hard-coded scale-out in risk_manager** | **Worked.** Fired at ETH +1.01% after 19 cycles, banked realized PnL, created a zero-risk runner with breakeven SL. First positive-PnL run. |
| 13  | 8h     | Remove old prompt rules 8/9 to stop duplicate sells | Clean — zero LLM-emitted sells across 240 cycles. Hard-coded automation was the only exit path. |
| 14  | 3h     | Drop scale-out trigger from +1.0% → +0.5% | No position reached +0.5% in 3h. Exposed *market-dependency* of the trigger — flat chop windows produce no exits. |
| 15  | 4h     | **Signal-quality gate** (reject low-confidence AND `|composite|<0.2` entries, anti-paralysis bypasses) | In progress / completed during handoff — read `diary_run15.jsonl` for outcome. |

## 5. Current state of each file

### `src/risk_manager.py`
- `validate_trade(...)` — notional, SL cap (≤5%), correlated exposure check.
- `check_mandatory_scale_outs(positions, cycle_counts, min_unrealized_pct=0.5)`
  — hard-coded: if a position has been held ≥10 cycles AND unrealized ≥
  threshold, close 50% at market and move SL on the runner to breakeven.
  Returns a list of actions for the orchestrator to execute.
- `check_signal_quality(asset, quant_signal, anti_paralysis_active,
  min_composite_abs=0.2)` — gates new entries: blocks when
  `confidence_label == "low"` AND `|composite_score| < min_composite_abs`.
  Bypassed when anti-paralysis is active.

### `src/main.py`
- Cycle loop orchestration.
- Maintains `position_cycle_count = {}` and `scaled_out_coins = set()` for
  scale-out state.
- Calls `check_mandatory_scale_outs` each cycle *before* consulting the LLM,
  executes scale-out actions directly (50% market close + breakeven SL on the
  runner), writes `{"action": "scale_out"}` rows to diary.
- Builds `asset_quant_lookup` per cycle with `{confidence_label,
  composite_score}` for the signal-quality gate.
- Enforces signal-quality gate on proposed entries *before*
  `validate_trade` — records `RISK SIGNAL-QUALITY BLOCK` events when rejected.
- Anti-paralysis: if 10 consecutive holds with zero open positions, forces
  one entry (bypasses signal-quality gate).

### `src/agent/decision_maker.py`
- Prompt includes Rules 1–9. The ones that matter for this bot's safety:
  - Rule 7: anti-paralysis forced-entry acknowledgement.
  - Rule 8: "SCALE-OUT IS AUTOMATED — DO NOT EMIT IT YOURSELF." Explicit note
    that risk manager handles 50% close at +0.5% / ≥10 cycles / breakeven SL.
  - Rule 9: signal-quality gate awareness — tells the LLM its low-confidence
    + weak-composite entries will be rejected, so it should either strengthen
    conviction or propose hold.
- Everything money-moving is worded as *awareness*, not enforcement. Enforcement
  lives in the risk manager.

### `src/trading/` (paper trader)
- Simulated fills against live Hyperliquid prices. SL/TP tracked internally.
  No live orders are placed in paper mode.

## 6. Known open issues

1. **Flat-market starvation of scale-out (Run 14).** If no position ever
   touches +0.5% in the run window, the hard-coded exit never fires and the
   bot just sits. Candidate fixes, in order of preference:
   - Time-based forced scale-out (e.g. after N cycles held, close 25% at
     current price regardless of PnL).
   - Slot-based re-entry so freed capital can hunt fresh setups instead of
     sitting in idle positions.
2. **All entries fire on cycle 1.** Nothing staggers entries across a window.
   Option B from earlier discussion (staggered entries) is still unimplemented.
3. **Signal-quality gate is untested across regimes.** Run 15 is the first
   window with it active; need trend-day data before trusting the thresholds
   (`confidence=low`, `|composite|<0.2`).

## 7. Next experiments queued

- **Slot-based re-entry:** when scale-out halves a position, free the "slot"
  so a new asset can be considered at the next cycle. Currently the halved
  position occupies its slot indefinitely.
- **Staggered entries (Option B):** instead of firing all new entries on
  cycle 1, allow at most one new entry per N cycles. Forces diversification
  across time, not just assets.
- **Time-based scale-out floor:** after K cycles held, close a fraction
  regardless of PnL to prevent flat-market starvation (issue 1 above).

Pick these off one at a time, one variable per run window, so the cause of
any behavior change stays identifiable.

## 8. Operational notes

- `.env` is gitignored and must be recreated on any new machine. It contains
  the Hyperliquid agent wallet + Anthropic API key. Never commit it.
- Per-run logs (`diary_run*.jsonl`, `decisions_run*.jsonl`,
  `paper_trading_output_run*.log`, `llm_requests_run*.log`, `prompts_run*.log`)
  are gitignored. They live locally for post-mortems.
- Starting a run: `python -m src.main` (or the project's configured entrypoint)
  with the appropriate env vars loaded. Redirect stdout/stderr to a
  `paper_trading_output_run<N>.log` file.
- The orchestrator writes to `diary.jsonl` and `decisions.jsonl` live — rename
  them to `*_run<N>.jsonl` at the end of each window.

## 9. How to continue in a new Claude Code chat

1. Open the project folder in Claude Code
   (`~/Desktop/AutoTrade/hyperliquid-trading-agent` on this machine, or
   `git clone https://github.com/jayantkamble10000/hyperliquid-trading-agent`
   on a new machine).
2. Claude Code auto-loads this file on session start — no special command
   needed. You can confirm by asking "what do you know about this project?"
   at the start of the chat.
3. Recreate `.env` (copy from `.env.example` and fill in keys) if you're on a
   new machine.
4. Tell Claude which run you want to review or run next, e.g. "read
   `diary_run15.jsonl` and summarize what happened" or "implement the
   time-based scale-out floor from section 7".
5. Keep this file up to date — append to the run-history table and the
   known-issues / next-experiments sections as work progresses.

---

*Last updated: Run 15 handoff.*
