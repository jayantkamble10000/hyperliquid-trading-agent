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
| 15  | 4h     | **Signal-quality gate** (reject low-confidence AND `|composite|<0.2` entries, anti-paralysis bypasses) | **Gate silently broken.** Post-mortem found `asset_quant_lookup` was reading `composite_score` from the top-level dict returned by `compute_all_signals`, but those keys live inside the nested `"recommendation"` sub-dict. Every block showed `|composite|=0.000` regardless of real signal strength. LLM-cited composites (e.g. SOL +0.1784) confirmed the mismatch. Anti-paralysis still fired (bypasses gate), forcing 1 BTC entry. 0 scale-outs — BTC never went positive. Flat-market starvation third consecutive run. |
| 16  | 4h+15m | **Fix gate lookup bug** (`_rec = quant_signals["recommendation"]`), re-test gate | **Gate working.** First block showed `\|composite\|=0.156` (real value, not 0.000). BTC passed gate (strong composite, Schwab catalyst), SOL blocked at 0.156. 2 entries, **2 scale-outs** (SOL +0.92%/$9.17 at cycle 10, BTC +0.57%/$5.71 at cycle 32), **final PnL +$5.42**. First double-scale-out run. New issue surfaced: 21/42 cycles had JSON parse errors (Haiku hitting max_tokens=4096, truncating mid-JSON). Gracefully defaulted to hold; scale-outs still fired (hard-coded). Fix: bump max_tokens to 8192. |
| 17  | 2h     | **Bump max_tokens 4096→8192** — but fix landed in wrong file | **Fix didn't apply.** `decision_maker.py` used `CONFIG.get("max_tokens") or 8192` but `config_loader.py` returned 4096 as default (not None), so `4096 or 8192 = 4096`. All 7 truncations still hit exactly 4096 output tokens. 2 entries (ETH $1500 + BTC $1000), 1 block (BTC `\|composite\|=0.187 < 0.2`), 0 scale-outs, final PnL −$0.88. Real fix: change default in `config_loader.py:86` to 8192. Committed `73e06c9`. |
| 18  | 4h+5m  | **Verify max_tokens=8192 fix in config_loader** | **Fix confirmed.** Zero `stop_reason=max_tokens` hits (was 7/21 in Run 17). 44/44 successful end_turn cycles. 2 residual parse errors — NOT truncation, different causes (malformed JSON from model, sanitizer retry then defaulted to hold; both cycles recovered). 2 entries (ETH $1k + BTC $1k), 2 blocks (BTC `\|composite\|=0.133`, SOL `\|composite\|=0.018`), **2 scale-outs** (ETH +0.58%/$5.84 at cycle ~16; BTC +0.59%/$5.92 at cycle ~37), **final PnL +$3.84**. Parse error issue shifts from truncation → occasional malformed output; needs a more robust JSON extraction strategy in a later run. |

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

1. **Residual JSON parse errors (~5% of cycles) from malformed model output.**
   Truncation is fixed (max_tokens=8192 confirmed working in Run 18). Remaining
   errors are malformed JSON (e.g. trailing commas, extra data) from the model
   occasionally breaking JSON syntax mid-reasoning. Sanitizer retry also fails
   because the retry prompt doesn't receive a truncated input to fix — it receives
   empty content. Fix options: (a) regex-extract the `trade_decisions` array
   directly from the raw output even if overall JSON is broken; (b) reduce prompt
   length to give the model more headroom within 8192 tokens.
2. **Flat-market starvation of scale-out (Runs 14–15, now resolved in 16).**
   Two scale-outs fired in Run 16 when the market moved. Issue may re-surface
   in genuinely flat windows. Queued fix: time-based scale-out floor (close a
   fraction after K cycles regardless of PnL).
3. **All entries fire on cycle 1 (or earliest trigger).** Nothing staggers
   entries across a window. Option B (staggered entries) still unimplemented.
4. **Quant threshold (0.25) is higher than gate threshold (0.2).** Effective
   entry threshold is max(0.25 quant action, 0.2 gate) = 0.25. Gate only bites
   when LLM overrides quant's hold (anti-paralysis or narrative). Thresholds
   should be aligned (both 0.2 or both 0.25).

## 7. Next experiments queued

- **[RUN 19 — NEXT] Time-based scale-out floor.**
  After K cycles held with no PnL movement (e.g. 20 cycles, ~2h), close 25% at
  market regardless of unrealized PnL, to prevent flat-market capital starvation.
  Implement in `risk_manager.check_mandatory_scale_outs` — add a second condition
  branch: `elif cycles_held >= K and not already_scaled`. Threshold-based scale-out
  still fires first if +0.5% is hit; time-based is the fallback for stagnant positions.
  Run 4h window to see if positions that never hit +0.5% now exit partially instead
  of sitting dead for the full window.
- **Align quant/gate thresholds:** quant uses 0.25 to define action=buy/sell;
  gate uses 0.2 to block. Lower quant threshold to 0.2 so they're consistent —
  OR accept the current asymmetry (gate is redundant for quant-driven holds,
  only bites on LLM-override proposals).
- **Time-based scale-out floor:** after K cycles held with no PnL movement,
  close a fraction regardless to prevent flat-market starvation.
- **Slot-based re-entry:** free the slot after scale-out so a new asset can
  enter next cycle. Currently the runner occupies its slot indefinitely.
- **Staggered entries (Option B):** at most one new entry per N cycles.

Pick these off one at a time, one variable per run window.

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

*Last updated: Run 18 post-mortem. Next: Run 19 — time-based scale-out floor.*
