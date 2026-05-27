"""
entry_filter.py — Directional Confirmation Entry Filter

Problem
-------
1H signals fire at candle close and execute at market immediately.  The first
5–20 minutes of the next candle often features a sharp spike AGAINST the new
position (liquidity sweep / stop hunt) before the true direction plays out.
On GOLD this is commonly $10–20; on BTC $400–600 — enough to hit the stop loss
on a trade that would otherwise have been correct.

Solution — Price-Action Confirmation (NOT a clock delay)
---------------------------------------------------------
After a signal fires, do NOT execute immediately.  Instead, watch the next
bot cycles (~5 min each) and execute only when price starts CONFIRMING the
direction.  The key thresholds:

  min_confirm_atr  — execute as soon as price moves this many ATR units IN
                     the signal direction from the original entry price.
                     Default 0.15 (15% of ATR).  For GOLD at ATR=$15 this is
                     a $2.25 move — small enough to catch the signal early,
                     large enough to filter out tick noise.

  max_adverse_atr  — cancel immediately if price moves this many ATR units
                     AGAINST the signal.  Default 1.2.  A sweep this deep
                     means either the direction is wrong or the stop would be
                     hit before recovery anyway.

  max_wait_minutes — backstop only.  If neither confirm nor cancel has
                     triggered after this many minutes, abandon the entry.
                     Default 20 min.  NOT the primary gate.

Typical wait time: 5–10 minutes (one or two bot cycles), not a flat clock.
The entry fires the moment price starts moving in the right direction.

Configuration (per-asset in config.json under assets.<ASSET>):
--------------------------------------------------------------
  "entry_confirmation": {
    "enabled": true,
    "min_confirm_atr": 0.15,   -- execute when price moves N * ATR toward signal
    "max_adverse_atr": 1.2,    -- cancel if price moves N * ATR against signal
    "max_wait_minutes": 20     -- backstop timeout; NOT the primary trigger
  }
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PendingEntry:
    asset: str
    signal: int              # 1 = BUY, -1 = SELL
    entry_price: float       # Close price at the moment the signal fired
    atr: float               # ATR-14 at signal generation time
    registered_at: datetime
    min_confirm_atr: float   # Confirm when price moves this fraction of ATR in signal dir
    max_adverse_atr: float   # Cancel if adverse move exceeds N * ATR
    max_wait_minutes: int    # Backstop timeout
    details: dict = field(default_factory=dict)


@dataclass
class ConfirmationResult:
    """Returned by check_and_update() to tell the caller what to do."""
    status: str              # 'BYPASS' | 'WAITING' | 'CONFIRMED' | 'CANCELLED'
    reason: str
    pending: Optional[PendingEntry] = None


# ── Main class ─────────────────────────────────────────────────────────────────

class EntryConfirmationFilter:
    """
    Price-action-driven entry confirmation.  One instance lives on the bot
    for the entire session.

    Call check_and_update() just before each execute_signal() call.
    Call expire_stale_entries() once at the start of each trading cycle.
    Call clear_pending() after a position is successfully opened.
    """

    def __init__(self):
        self._pending: Dict[str, PendingEntry] = {}

    # ── Primary API ────────────────────────────────────────────────────────────

    def check_and_update(
        self,
        asset: str,
        signal: int,
        current_price: float,
        atr: float,
        asset_cfg: dict,
        signal_details: Optional[dict] = None,
    ) -> ConfirmationResult:
        """
        Called just before executing a trade.

        Returns
        -------
        BYPASS    — filter not enabled for this asset; proceed immediately
        WAITING   — signal registered this cycle; do NOT execute yet
        CONFIRMED — price has moved in signal direction; execute now
        CANCELLED — adverse sweep too deep, or timeout; abort entry
        """
        conf_cfg = asset_cfg.get("entry_confirmation", {})
        if not conf_cfg.get("enabled", False):
            return ConfirmationResult(status="BYPASS", reason="not enabled for this asset")

        min_confirm  = float(conf_cfg.get("min_confirm_atr",  0.15))
        max_adverse  = float(conf_cfg.get("max_adverse_atr",  1.2))
        max_wait     = int(conf_cfg.get("max_wait_minutes",   20))
        safe_atr     = max(float(atr) if atr else 0.0, 1e-8)

        existing = self._pending.get(asset)

        # ── Signal direction changed — discard stale pending ──────────────
        if existing is not None and existing.signal != signal:
            old_dir = "BUY" if existing.signal == 1 else "SELL"
            new_dir = "BUY" if signal == 1 else "SELL"
            logger.info(
                f"[ENTRY FILTER] {asset}: Direction flipped "
                f"({old_dir} → {new_dir}) — old pending cleared."
            )
            del self._pending[asset]
            existing = None

        # ── First time this signal fires — register, skip execution ───────
        if existing is None:
            self._pending[asset] = PendingEntry(
                asset=asset,
                signal=signal,
                entry_price=current_price,
                atr=safe_atr,
                registered_at=datetime.now(),
                min_confirm_atr=min_confirm,
                max_adverse_atr=max_adverse,
                max_wait_minutes=max_wait,
                details=signal_details or {},
            )
            dir_str = "BUY" if signal == 1 else "SELL"
            confirm_dist = min_confirm * safe_atr
            adverse_dist = max_adverse * safe_atr
            logger.info(
                f"[ENTRY FILTER] ⏳ {asset} {dir_str}: Signal registered at {current_price:.5f}. "
                f"Will execute when price moves ≥{confirm_dist:.5f} in signal direction "
                f"(confirm={min_confirm}x ATR). "
                f"Cancel if adverse move >{adverse_dist:.5f} ({max_adverse}x ATR). "
                f"Backstop timeout: {max_wait}min."
            )
            return ConfirmationResult(
                status="WAITING",
                reason="first cycle — waiting for directional confirmation",
                pending=self._pending[asset],
            )

        # ── Pending entry exists — run all three checks ───────────────────
        now = datetime.now()
        age_minutes = (now - existing.registered_at).total_seconds() / 60.0

        # Check 1 — Backstop timeout
        if age_minutes > existing.max_wait_minutes:
            del self._pending[asset]
            dir_str = "BUY" if existing.signal == 1 else "SELL"
            logger.info(
                f"[ENTRY FILTER] ❌ {asset} {dir_str}: CANCELLED — "
                f"timeout ({age_minutes:.0f}min > {existing.max_wait_minutes}min). "
                f"Market direction unclear — skipping entry."
            )
            return ConfirmationResult(
                status="CANCELLED",
                reason=f"timeout after {age_minutes:.0f}min — market direction unclear",
                pending=existing,
            )

        # Measure directional and adverse move from original entry price
        if existing.signal == 1:       # BUY
            confirm_move = current_price - existing.entry_price   # positive = good
            adverse_move = existing.entry_price - current_price   # positive = bad
        else:                           # SELL
            confirm_move = existing.entry_price - current_price   # positive = good
            adverse_move = current_price - existing.entry_price   # positive = bad

        confirm_atr_units = confirm_move / existing.atr if existing.atr > 0 else 0.0
        adverse_atr_units = adverse_move / existing.atr if existing.atr > 0 else 0.0

        dir_str = "BUY" if existing.signal == 1 else "SELL"

        # Check 2 — Deep adverse sweep → cancel
        if adverse_atr_units > existing.max_adverse_atr:
            del self._pending[asset]
            logger.info(
                f"[ENTRY FILTER] ❌ {asset} {dir_str}: CANCELLED — "
                f"deep sweep: price moved {adverse_atr_units:.2f}x ATR against signal "
                f"(threshold {existing.max_adverse_atr}x). Entry skipped."
            )
            return ConfirmationResult(
                status="CANCELLED",
                reason=(
                    f"adverse sweep {adverse_atr_units:.2f}x ATR "
                    f"(limit {existing.max_adverse_atr}x)"
                ),
                pending=existing,
            )

        # Check 3 — Directional confirmation → execute now
        if confirm_atr_units >= existing.min_confirm_atr:
            del self._pending[asset]
            logger.info(
                f"[ENTRY FILTER] ✅ {asset} {dir_str}: CONFIRMED after {age_minutes:.0f}min — "
                f"price moved {confirm_atr_units:.2f}x ATR in signal direction "
                f"(threshold {existing.min_confirm_atr}x). Executing."
            )
            return ConfirmationResult(
                status="CONFIRMED",
                reason=(
                    f"directional move {confirm_atr_units:.2f}x ATR confirmed after {age_minutes:.0f}min"
                ),
                pending=existing,
            )

        # Neither confirmed nor cancelled — still waiting
        logger.info(
            f"[ENTRY FILTER] ⏳ {asset} {dir_str}: Still waiting ({age_minutes:.0f}min). "
            f"Confirm progress: {confirm_atr_units:.2f}x ATR (need {existing.min_confirm_atr}x). "
            f"Adverse: {adverse_atr_units:.2f}x ATR (limit {existing.max_adverse_atr}x)."
        )
        return ConfirmationResult(
            status="WAITING",
            reason=(
                f"confirm {confirm_atr_units:.2f}x/{existing.min_confirm_atr}x ATR, "
                f"adverse {adverse_atr_units:.2f}x/{existing.max_adverse_atr}x ATR"
            ),
            pending=existing,
        )

    # ── Lifecycle helpers ──────────────────────────────────────────────────────

    def clear_pending(self, asset: str) -> None:
        """Call after a position is successfully opened or cancelled externally."""
        if asset in self._pending:
            del self._pending[asset]
            logger.debug(f"[ENTRY FILTER] Cleared pending entry for {asset}.")

    def expire_stale_entries(self) -> None:
        """
        Call at the start of every main trading cycle.
        Cleans up entries whose signal went to 0 and never re-fired,
        so they don't sit in memory past their timeout window.
        """
        now = datetime.now()
        to_remove = []
        for asset, entry in self._pending.items():
            age_minutes = (now - entry.registered_at).total_seconds() / 60.0
            if age_minutes > entry.max_wait_minutes:
                to_remove.append(asset)
                dir_str = "BUY" if entry.signal == 1 else "SELL"
                logger.info(
                    f"[ENTRY FILTER] 🗑️  {asset} {dir_str}: Stale pending expired "
                    f"({age_minutes:.0f}min) — signal never re-fired."
                )
        for asset in to_remove:
            del self._pending[asset]

    def has_pending(self, asset: str) -> bool:
        return asset in self._pending

    def get_pending_summary(self) -> dict:
        """For dashboard / logging."""
        now = datetime.now()
        return {
            asset: {
                "signal":           "BUY" if e.signal == 1 else "SELL",
                "registered_at":    e.registered_at.strftime("%H:%M:%S"),
                "age_minutes":      round((now - e.registered_at).total_seconds() / 60, 1),
                "entry_price":      e.entry_price,
                "atr":              e.atr,
                "min_confirm_atr":  e.min_confirm_atr,
                "max_adverse_atr":  e.max_adverse_atr,
                "max_wait_minutes": e.max_wait_minutes,
            }
            for asset, e in self._pending.items()
        }
