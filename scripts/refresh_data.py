#!/usr/bin/env python3
"""
scripts/refresh_data.py
=======================
Standalone historical data refresh script.

Run this BEFORE backtesting, or whenever the bot's CSV files look stale/short.
It connects to MT5 (and Binance if BTC is on Binance mode), then force-downloads
1H / 4H / 1D data for every enabled asset.

Usage:
    # Audit only — show what's in each file, no changes
    python scripts/refresh_data.py --audit

    # Full refresh all enabled assets (deletes and re-downloads)
    python scripts/refresh_data.py --full

    # Refresh a single asset
    python scripts/refresh_data.py --full --asset GOLD

    # Incremental update only (append missing bars)
    python scripts/refresh_data.py --update

Requirements:
    - MetaTrader 5 must be running and logged in before executing this script.
    - Run from the TBOT root directory: python scripts/refresh_data.py --full
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ── Make sure imports resolve from the project root ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Load .env so MT5/Binance credentials are available ───────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on credentials in config.json

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Expected minimum bar counts ───────────────────────────────────────────────
MIN_BARS = {
    "1h": 2_000,   # ~83 days — enough for 200-EMA + strategy warmup
    "4h":   500,   # ~83 days
    "1d":   250,   # ~1 year — minimum for 200-day EMA
}

# ── Recommended bar counts (what force-refresh should achieve) ────────────────
TARGET_BARS = {
    "1h": 8_760,   # 365 days
    "4h": 4_380,   # 730 days
    "1d": 1_095,   # 3 years
}


# ─────────────────────────────────────────────────────────────────────────────
# Audit helpers
# ─────────────────────────────────────────────────────────────────────────────

def _audit_file(csv_path: Path, timeframe: str) -> dict:
    """Return a status dict for a single CSV file."""
    if not csv_path.exists():
        return {"status": "MISSING", "bars": 0, "start": None, "end": None, "interval_ok": None}

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        # Find date column
        date_col = next(
            (c for c in df.columns if c in ("date", "timestamp", "datetime", "time")),
            None
        )
        if date_col is None:
            return {"status": "NO_DATE_COL", "bars": len(df), "start": None, "end": None, "interval_ok": None}

        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        bars    = len(df)
        start   = df[date_col].iloc[0]
        end     = df[date_col].iloc[-1]

        # Check interval consistency (sample the first 20 gaps)
        if bars > 2:
            gaps = df[date_col].diff().dropna().head(20)
            modal_gap = gaps.mode()[0]
            expected = {"1h": pd.Timedelta("1h"), "4h": pd.Timedelta("4h"), "1d": pd.Timedelta("1d")}
            exp = expected.get(timeframe, pd.Timedelta("1h"))
            # Allow up to 2× expected gap (weekends add gaps in FX/commodity 1D)
            interval_ok = modal_gap <= exp * 2
        else:
            interval_ok = None

        min_needed = MIN_BARS[timeframe]
        if bars < min_needed // 4:
            status = "CRITICAL"   # < 25% of minimum
        elif bars < min_needed:
            status = "LOW"        # below minimum but usable
        else:
            status = "OK"

        return {
            "status": status,
            "bars": bars,
            "start": start.strftime("%Y-%m-%d") if start else None,
            "end":   end.strftime("%Y-%m-%d")   if end   else None,
            "interval_ok": interval_ok,
        }

    except Exception as e:
        return {"status": f"ERROR: {e}", "bars": 0, "start": None, "end": None, "interval_ok": None}


def audit_all(data_dir: Path, config: dict):
    """Print an audit table for all enabled assets × all timeframes."""
    print()
    print("=" * 90)
    print("  DATA AUDIT")
    print("=" * 90)
    print(f"  {'Asset':<10} {'TF':<5} {'Status':<10} {'Bars':>7} {'Start':<12} {'End':<12} {'Interval'}")
    print("-" * 90)

    issues = []

    for asset_name, cfg in config["assets"].items():
        if not cfg.get("enabled", False):
            continue

        exchange = cfg.get("exchange", "binance")
        if exchange == "mt5":
            symbol = cfg.get("mt5_symbol", cfg.get("symbol", asset_name))
        else:
            symbol = cfg.get("symbol", asset_name)

        for tf in ("1h", "4h", "1d"):
            fname   = f"{symbol}_{tf}.csv"
            fpath   = data_dir / fname
            result  = _audit_file(fpath, tf)

            status     = result["status"]
            bars       = result["bars"]
            start      = result["start"] or "—"
            end        = result["end"]   or "—"
            interval   = ("✅" if result["interval_ok"] else "⚠️ WRONG") if result["interval_ok"] is not None else "?"

            flag = ""
            if status in ("MISSING", "CRITICAL", "NO_DATE_COL") or result["interval_ok"] is False:
                flag = " ← FIX NEEDED"
                issues.append(f"  {asset_name} {tf.upper()}: {status} ({fname})")

            print(f"  {asset_name:<10} {tf:<5} {status:<10} {bars:>7}  {start:<12} {end:<12} {interval}{flag}")

    print("=" * 90)

    if issues:
        print()
        print(f"  ⚠️  {len(issues)} issue(s) found — run with --full to fix:")
        for i in issues:
            print(i)
    else:
        print()
        print("  ✅ All files look healthy")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Refresh logic
# ─────────────────────────────────────────────────────────────────────────────

def _apply_env_credentials(config: dict) -> dict:
    """
    Inject credentials from environment variables into config, mirroring the
    same logic used by main.py's TradingBot.__init__.  This ensures the refresh
    script works even when config.json has blank credential fields.
    """
    mt5 = config.setdefault("api", {}).setdefault("mt5", {})
    if os.getenv("MT5_LOGIN"):
        mt5["login"] = int(os.getenv("MT5_LOGIN"))
    if os.getenv("MT5_PASSWORD"):
        mt5["password"] = os.getenv("MT5_PASSWORD")
    if os.getenv("MT5_SERVER"):
        mt5["server"] = os.getenv("MT5_SERVER")
    if os.getenv("MT5_PATH"):
        mt5["path"] = os.getenv("MT5_PATH")

    # Binance (non-critical for MT5-only setups)
    binance = config.setdefault("api", {}).setdefault("binance", {})
    if os.getenv("BINANCE_API_KEY"):
        binance["api_key"] = os.getenv("BINANCE_API_KEY")
    if os.getenv("BINANCE_API_SECRET"):
        binance["api_secret"] = os.getenv("BINANCE_API_SECRET")

    return config


def _init_data_manager(config: dict):
    """
    Initialise the DataManager with Binance and MT5 connections.
    Returns (data_manager, ok: bool).
    """
    from src.data.data_manager import DataManager

    config = _apply_env_credentials(config)
    dm = DataManager(config)

    # Try Binance (non-fatal if BTC is MT5-only)
    binance_ok = False
    try:
        binance_ok = dm.initialize_binance()
        if binance_ok:
            logger.info("✅ Binance connection ready")
        else:
            logger.warning("⚠️  Binance initialisation failed — BTC data will use MT5")
    except Exception as e:
        logger.warning(f"⚠️  Binance error (non-fatal): {e}")

    # Try MT5 (required for FX/commodity assets)
    mt5_ok = False
    try:
        mt5_ok = dm.initialize_mt5()
        if mt5_ok:
            logger.info("✅ MT5 connection ready")
        else:
            logger.error("❌ MT5 initialisation failed")
            logger.error("   Make sure MetaTrader 5 is running and you are logged in.")
    except Exception as e:
        logger.error(f"❌ MT5 error: {e}")

    if not binance_ok and not mt5_ok:
        logger.error("Both connections failed — cannot refresh data.")
        return dm, False

    return dm, True


def refresh(
    config: dict,
    data_dir: Path,
    force_full: bool = False,
    asset_filter: str = None,
):
    """
    Initialise connections and run the historical updater.

    Args:
        config:       Loaded config dict
        data_dir:     data/raw/ Path
        force_full:   True = wipe and re-download; False = incremental only
        asset_filter: If set, only refresh this asset (e.g. "GOLD")
    """
    from src.update.historical_updater import HistoricalDataUpdater

    dm, ok = _init_data_manager(config)
    if not ok:
        logger.error("Aborting refresh — no connections available.")
        sys.exit(1)

    updater = HistoricalDataUpdater(data_manager=dm, config=config)

    if force_full:
        logger.info("Mode: FULL REFRESH — existing CSVs will be overwritten")
    else:
        logger.info("Mode: INCREMENTAL — only appending missing bars")

    if asset_filter:
        asset_filter = asset_filter.upper()
        logger.info(f"Scope: single asset → {asset_filter}")
        if asset_filter not in config["assets"]:
            logger.error(f"Unknown asset '{asset_filter}'. "
                         f"Available: {list(config['assets'].keys())}")
            sys.exit(1)
        results = {
            asset_filter: updater.update_asset_history(
                asset_name=asset_filter,
                force_full_refresh=force_full,
            )
        }
    else:
        logger.info("Scope: all enabled assets")
        results = updater.update_all_enabled_assets(force_full_refresh=force_full)

    # Post-refresh audit
    print()
    print("─" * 60)
    print("  POST-REFRESH STATUS")
    print("─" * 60)
    all_ok = True
    for asset_name, tf_results in results.items():
        for tf, success in tf_results.items():
            exchange = config["assets"].get(asset_name, {}).get("exchange", "binance")
            symbol = (
                config["assets"][asset_name].get("mt5_symbol",
                config["assets"][asset_name].get("symbol", asset_name))
                if exchange == "mt5"
                else config["assets"][asset_name].get("symbol", asset_name)
            )
            fname   = f"{symbol}_{tf}.csv"
            fpath   = data_dir / fname
            audit   = _audit_file(fpath, tf)
            icon    = "✅" if audit["status"] == "OK" and success else "⚠️"
            if audit["status"] != "OK" or not success:
                all_ok = False
            print(
                f"  {icon} {asset_name:<8} {tf.upper():<4}  "
                f"{audit['bars']:>6} bars  "
                f"{audit['start'] or '—'} → {audit['end'] or '—'}"
            )

    print("─" * 60)
    if all_ok:
        print("  ✅ All refreshed successfully — ready for backtesting and live trading")
    else:
        print("  ⚠️  Some assets may need attention (see above)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Refresh historical OHLCV data for all TBOT assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current data health — no changes made
  python scripts/refresh_data.py --audit

  # Force full re-download for all enabled assets
  python scripts/refresh_data.py --full

  # Full refresh for one asset only
  python scripts/refresh_data.py --full --asset GOLD

  # Incremental update (append only new bars)
  python scripts/refresh_data.py --update
        """,
    )
    parser.add_argument("--audit",  action="store_true", help="Audit only — show bar counts and status, no download")
    parser.add_argument("--full",   action="store_true", help="Force full re-download (overwrites existing CSVs)")
    parser.add_argument("--update", action="store_true", help="Incremental update — append missing bars only")
    parser.add_argument("--asset",  type=str, default=None, help="Restrict to a single asset (e.g. GOLD)")
    args = parser.parse_args()

    if not any([args.audit, args.full, args.update]):
        parser.print_help()
        sys.exit(0)

    # Load config
    config_path = PROJECT_ROOT / "config" / "config.json"
    if not config_path.exists():
        logger.error(f"config/config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  TBOT DATA REFRESH")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if args.audit:
        audit_all(data_dir, config)
        return

    if args.full:
        audit_all(data_dir, config)   # show before state
        refresh(config, data_dir, force_full=True,  asset_filter=args.asset)
    elif args.update:
        refresh(config, data_dir, force_full=False, asset_filter=args.asset)


if __name__ == "__main__":
    main()
