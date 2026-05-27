#!/usr/bin/env python3
"""
Backtesting Script - Council vs Performance Aggregator Comparison
Supports all assets defined in config/config.json
"""
import json
import logging
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import talib as ta
import pickle
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.ema_strategy import EMAStrategy
from src.execution.signal_aggregator import PerformanceWeightedAggregator
from src.execution.council_aggregator import InstitutionalCouncilAggregator
from src.ai import DynamicAnalyst, OHLCSniper, HybridSignalValidator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _setup_run_log(run_tag: str) -> Path:
    """
    Add a FileHandler to the root logger so every logger.info / logger.warning
    call — including per-asset results and the final comparison table — is
    automatically written to  logs/backtest_<run_tag>.log

    Returns the log file path so the CLI can print it at the end.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"backtest_{run_tag}.log"

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Attach to the root logger so ALL loggers in this process write to it
    logging.getLogger().addHandler(fh)

    return log_path

# ─────────────────────────────────────────────────────────────────────────────
# Data file mapping: derived from config/config.json — single source of truth.
#
# Uses the same symbol-resolution logic as HistoricalDataUpdater and the live
# regime detector so that backtesting always reads the same CSV files the live
# system writes to.
#
#   MT5 assets : prefer mt5_symbol (e.g. "BTCUSDm"), fall back to symbol
#   Other      : use symbol directly
#
# Assets do NOT need to be enabled=true to appear here; disabled assets can
# still be backtested.
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.json from the project root, tolerating missing file."""
    cfg_path = Path("config/config.json")
    if not cfg_path.exists():
        logger.warning("config/config.json not found — DATA_FILE_MAP will be empty")
        return {"assets": {}}
    with open(cfg_path) as f:
        return json.load(f)


def _derive_csv_symbol(asset_name: str, cfg: dict) -> str:
    """
    Return the MT5/Binance symbol string that HistoricalDataUpdater uses as
    the CSV filename stem (e.g. "BTCUSDm", "XAUUSDm", "EURUSDm").
    """
    exchange = cfg.get("exchange", "binance")
    if exchange == "mt5":
        # BTC carries a separate mt5_symbol; other MT5 assets use symbol directly
        return cfg.get("mt5_symbol", cfg.get("symbol", asset_name))
    return cfg.get("symbol", asset_name)


# Build at module load time so argparse choices are populated correctly
_BOOT_CONFIG = _load_config()

DATA_FILE_MAP: dict = {
    name.upper(): f"{_derive_csv_symbol(name, cfg)}_1h.csv"
    for name, cfg in _BOOT_CONFIG.get("assets", {}).items()
}

SUPPORTED_ASSETS = list(DATA_FILE_MAP.keys())

logger.debug("DATA_FILE_MAP: %s", DATA_FILE_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# Asset-specific presets for PerformanceWeightedAggregator
# ─────────────────────────────────────────────────────────────────────────────
AGGREGATOR_PRESETS = {
    "BTC": {
        "conservative": {
            "buy_threshold": 0.35, "sell_threshold": 0.40, "two_strategy_bonus": 0.18,
            "three_strategy_bonus": 0.20, "bull_buy_boost": 0.10, "bull_sell_penalty": 0.12,
            "bear_sell_boost": 0.10, "bear_buy_penalty": 0.12, "min_confidence_to_use": 0.12,
            "min_signal_quality": 0.32, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.75,
            "opposition_penalty": 0.5, "verbose": False,
        },
        "balanced": {
            "buy_threshold": 0.30, "sell_threshold": 0.36, "two_strategy_bonus": 0.20,
            "three_strategy_bonus": 0.22, "bull_buy_boost": 0.11, "bull_sell_penalty": 0.11,
            "bear_sell_boost": 0.11, "bear_buy_penalty": 0.11, "min_confidence_to_use": 0.10,
            "min_signal_quality": 0.28, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.72,
            "opposition_penalty": 0.5, "verbose": False,
        },
        "aggressive": {
            "buy_threshold": 0.25, "sell_threshold": 0.30, "two_strategy_bonus": 0.22,
            "three_strategy_bonus": 0.25, "bull_buy_boost": 0.12, "bull_sell_penalty": 0.12,
            "bear_sell_boost": 0.12, "bear_buy_penalty": 0.12, "min_confidence_to_use": 0.09,
            "min_signal_quality": 0.25, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.70,
            "opposition_penalty": 0.5, "verbose": False,
        },
        "scalper": {
            "buy_threshold": 0.24, "sell_threshold": 0.30, "two_strategy_bonus": 0.25,
            "three_strategy_bonus": 0.30, "bull_buy_boost": 0.15, "bull_sell_penalty": 0.10,
            "bear_sell_boost": 0.15, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.08,
            "min_signal_quality": 0.20, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.65,
            "verbose": False, "min_quality_margin": 0.05, "opposition_penalty": 0.5,
        },
    },
    "GOLD": {
        "conservative": {
            "buy_threshold": 0.38, "sell_threshold": 0.42, "two_strategy_bonus": 0.18,
            "three_strategy_bonus": 0.25, "bull_buy_boost": 0.07, "bull_sell_penalty": 0.09,
            "bear_sell_boost": 0.07, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.12,
            "min_signal_quality": 0.30, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.75,
            "opposition_penalty": 0.5, "verbose": False,
        },
        "balanced": {
            "buy_threshold": 0.33, "sell_threshold": 0.36, "two_strategy_bonus": 0.20,
            "three_strategy_bonus": 0.25, "bull_buy_boost": 0.08, "bull_sell_penalty": 0.08,
            "bear_sell_boost": 0.08, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.10,
            "min_signal_quality": 0.28, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.72,
            "opposition_penalty": 0.5, "verbose": False,
        },
        "aggressive": {
            "buy_threshold": 0.28, "sell_threshold": 0.30, "two_strategy_bonus": 0.22,
            "three_strategy_bonus": 0.30, "bull_buy_boost": 0.09, "bull_sell_penalty": 0.09,
            "bear_sell_boost": 0.09, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.08,
            "min_signal_quality": 0.22, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.70,
            "opposition_penalty": 0.5, "verbose": False,
        },
        "scalper": {
            "buy_threshold": 0.23, "sell_threshold": 0.30, "two_strategy_bonus": 0.25,
            "three_strategy_bonus": 0.35, "bull_buy_boost": 0.12, "bull_sell_penalty": 0.08,
            "bear_sell_boost": 0.12, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.06,
            "min_signal_quality": 0.18, "hold_contribution_pct": 0.0,
            "allow_single_override": True, "single_override_threshold": 0.65,
            "verbose": False, "min_quality_margin": 0.06, "opposition_penalty": 0.5,
        },
    },
    "EURUSD": {
        "conservative": {"buy_threshold": 0.38, "sell_threshold": 0.42, "two_strategy_bonus": 0.18, "three_strategy_bonus": 0.25, "bull_buy_boost": 0.07, "bull_sell_penalty": 0.09, "bear_sell_boost": 0.07, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.12, "min_signal_quality": 0.30, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.75, "opposition_penalty": 0.5, "verbose": False},
        "balanced":     {"buy_threshold": 0.32, "sell_threshold": 0.36, "two_strategy_bonus": 0.20, "three_strategy_bonus": 0.25, "bull_buy_boost": 0.09, "bull_sell_penalty": 0.09, "bear_sell_boost": 0.09, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.10, "min_signal_quality": 0.27, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.72, "opposition_penalty": 0.5, "verbose": False},
        "aggressive":   {"buy_threshold": 0.26, "sell_threshold": 0.30, "two_strategy_bonus": 0.22, "three_strategy_bonus": 0.30, "bull_buy_boost": 0.10, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.10, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.08, "min_signal_quality": 0.22, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.70, "opposition_penalty": 0.5, "verbose": False},
        "scalper":      {"buy_threshold": 0.22, "sell_threshold": 0.28, "two_strategy_bonus": 0.25, "three_strategy_bonus": 0.35, "bull_buy_boost": 0.12, "bull_sell_penalty": 0.08, "bear_sell_boost": 0.12, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.06, "min_signal_quality": 0.18, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.65, "verbose": False, "min_quality_margin": 0.06, "opposition_penalty": 0.5},
    },
    "GBPUSD": {
        "conservative": {"buy_threshold": 0.36, "sell_threshold": 0.40, "two_strategy_bonus": 0.18, "three_strategy_bonus": 0.24, "bull_buy_boost": 0.08, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.08, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.12, "min_signal_quality": 0.30, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.75, "opposition_penalty": 0.5, "verbose": False},
        "balanced":     {"buy_threshold": 0.30, "sell_threshold": 0.35, "two_strategy_bonus": 0.20, "three_strategy_bonus": 0.25, "bull_buy_boost": 0.10, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.10, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.10, "min_signal_quality": 0.26, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.72, "opposition_penalty": 0.5, "verbose": False},
        "aggressive":   {"buy_threshold": 0.25, "sell_threshold": 0.29, "two_strategy_bonus": 0.22, "three_strategy_bonus": 0.28, "bull_buy_boost": 0.11, "bull_sell_penalty": 0.11, "bear_sell_boost": 0.11, "bear_buy_penalty": 0.11, "min_confidence_to_use": 0.08, "min_signal_quality": 0.21, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.70, "opposition_penalty": 0.5, "verbose": False},
        "scalper":      {"buy_threshold": 0.21, "sell_threshold": 0.27, "two_strategy_bonus": 0.25, "three_strategy_bonus": 0.33, "bull_buy_boost": 0.13, "bull_sell_penalty": 0.08, "bear_sell_boost": 0.13, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.06, "min_signal_quality": 0.17, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.65, "verbose": False, "min_quality_margin": 0.06, "opposition_penalty": 0.5},
    },
    "USDJPY": {
        "conservative": {"buy_threshold": 0.36, "sell_threshold": 0.40, "two_strategy_bonus": 0.18, "three_strategy_bonus": 0.24, "bull_buy_boost": 0.08, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.08, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.12, "min_signal_quality": 0.30, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.75, "opposition_penalty": 0.5, "verbose": False},
        "balanced":     {"buy_threshold": 0.30, "sell_threshold": 0.34, "two_strategy_bonus": 0.20, "three_strategy_bonus": 0.25, "bull_buy_boost": 0.10, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.10, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.10, "min_signal_quality": 0.25, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.72, "opposition_penalty": 0.5, "verbose": False},
        "aggressive":   {"buy_threshold": 0.24, "sell_threshold": 0.28, "two_strategy_bonus": 0.22, "three_strategy_bonus": 0.28, "bull_buy_boost": 0.11, "bull_sell_penalty": 0.11, "bear_sell_boost": 0.11, "bear_buy_penalty": 0.11, "min_confidence_to_use": 0.08, "min_signal_quality": 0.20, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.70, "opposition_penalty": 0.5, "verbose": False},
        "scalper":      {"buy_threshold": 0.20, "sell_threshold": 0.26, "two_strategy_bonus": 0.25, "three_strategy_bonus": 0.33, "bull_buy_boost": 0.13, "bull_sell_penalty": 0.08, "bear_sell_boost": 0.13, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.06, "min_signal_quality": 0.16, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.65, "verbose": False, "min_quality_margin": 0.06, "opposition_penalty": 0.5},
    },
    "USTEC": {
        "conservative": {"buy_threshold": 0.36, "sell_threshold": 0.40, "two_strategy_bonus": 0.18, "three_strategy_bonus": 0.22, "bull_buy_boost": 0.09, "bull_sell_penalty": 0.11, "bear_sell_boost": 0.09, "bear_buy_penalty": 0.11, "min_confidence_to_use": 0.12, "min_signal_quality": 0.30, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.75, "opposition_penalty": 0.5, "verbose": False},
        "balanced":     {"buy_threshold": 0.31, "sell_threshold": 0.35, "two_strategy_bonus": 0.20, "three_strategy_bonus": 0.24, "bull_buy_boost": 0.10, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.10, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.10, "min_signal_quality": 0.27, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.72, "opposition_penalty": 0.5, "verbose": False},
        "aggressive":   {"buy_threshold": 0.26, "sell_threshold": 0.30, "two_strategy_bonus": 0.22, "three_strategy_bonus": 0.27, "bull_buy_boost": 0.11, "bull_sell_penalty": 0.11, "bear_sell_boost": 0.11, "bear_buy_penalty": 0.11, "min_confidence_to_use": 0.09, "min_signal_quality": 0.22, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.70, "opposition_penalty": 0.5, "verbose": False},
        "scalper":      {"buy_threshold": 0.22, "sell_threshold": 0.28, "two_strategy_bonus": 0.25, "three_strategy_bonus": 0.32, "bull_buy_boost": 0.13, "bull_sell_penalty": 0.09, "bear_sell_boost": 0.13, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.07, "min_signal_quality": 0.18, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.65, "verbose": False, "min_quality_margin": 0.06, "opposition_penalty": 0.5},
    },
    "USOIL": {
        "conservative": {"buy_threshold": 0.37, "sell_threshold": 0.41, "two_strategy_bonus": 0.18, "three_strategy_bonus": 0.24, "bull_buy_boost": 0.08, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.08, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.12, "min_signal_quality": 0.30, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.75, "opposition_penalty": 0.5, "verbose": False},
        "balanced":     {"buy_threshold": 0.32, "sell_threshold": 0.36, "two_strategy_bonus": 0.20, "three_strategy_bonus": 0.25, "bull_buy_boost": 0.09, "bull_sell_penalty": 0.09, "bear_sell_boost": 0.09, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.10, "min_signal_quality": 0.27, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.72, "opposition_penalty": 0.5, "verbose": False},
        "aggressive":   {"buy_threshold": 0.27, "sell_threshold": 0.31, "two_strategy_bonus": 0.22, "three_strategy_bonus": 0.29, "bull_buy_boost": 0.10, "bull_sell_penalty": 0.10, "bear_sell_boost": 0.10, "bear_buy_penalty": 0.10, "min_confidence_to_use": 0.08, "min_signal_quality": 0.22, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.70, "opposition_penalty": 0.5, "verbose": False},
        "scalper":      {"buy_threshold": 0.23, "sell_threshold": 0.29, "two_strategy_bonus": 0.25, "three_strategy_bonus": 0.34, "bull_buy_boost": 0.12, "bull_sell_penalty": 0.08, "bear_sell_boost": 0.12, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.06, "min_signal_quality": 0.18, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.65, "verbose": False, "min_quality_margin": 0.06, "opposition_penalty": 0.5},
    },
    "GBPAUD": {
        "conservative": {"buy_threshold": 0.38, "sell_threshold": 0.42, "two_strategy_bonus": 0.18, "three_strategy_bonus": 0.25, "bull_buy_boost": 0.07, "bull_sell_penalty": 0.09, "bear_sell_boost": 0.07, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.13, "min_signal_quality": 0.32, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.75, "opposition_penalty": 0.5, "verbose": False},
        "balanced":     {"buy_threshold": 0.33, "sell_threshold": 0.37, "two_strategy_bonus": 0.20, "three_strategy_bonus": 0.26, "bull_buy_boost": 0.08, "bull_sell_penalty": 0.08, "bear_sell_boost": 0.08, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.11, "min_signal_quality": 0.28, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.73, "opposition_penalty": 0.5, "verbose": False},
        "aggressive":   {"buy_threshold": 0.28, "sell_threshold": 0.32, "two_strategy_bonus": 0.22, "three_strategy_bonus": 0.30, "bull_buy_boost": 0.09, "bull_sell_penalty": 0.09, "bear_sell_boost": 0.09, "bear_buy_penalty": 0.09, "min_confidence_to_use": 0.09, "min_signal_quality": 0.23, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.70, "opposition_penalty": 0.5, "verbose": False},
        "scalper":      {"buy_threshold": 0.24, "sell_threshold": 0.30, "two_strategy_bonus": 0.25, "three_strategy_bonus": 0.35, "bull_buy_boost": 0.11, "bull_sell_penalty": 0.08, "bear_sell_boost": 0.11, "bear_buy_penalty": 0.08, "min_confidence_to_use": 0.07, "min_signal_quality": 0.19, "hold_contribution_pct": 0.0, "allow_single_override": True, "single_override_threshold": 0.65, "verbose": False, "min_quality_margin": 0.07, "opposition_penalty": 0.5},
    },
}

_DEFAULT_PRESET = {
    "conservative": AGGREGATOR_PRESETS["GOLD"]["conservative"],
    "balanced":     AGGREGATOR_PRESETS["GOLD"]["balanced"],
    "aggressive":   AGGREGATOR_PRESETS["GOLD"]["aggressive"],
    "scalper":      AGGREGATOR_PRESETS["GOLD"]["scalper"],
}


# Council aggregator thresholds per preset level.
# The AGGREGATOR_PRESETS above only cover PerformanceWeightedAggregator keys.
# InstitutionalCouncilAggregator reads `trend_aligned_threshold` and
# `counter_trend_threshold` from the same config dict.  Without these keys it
# falls back to its hard-coded defaults (3.0 / 3.5), which require ~75% weighted
# agreement from the three strategies — far too strict for backtesting.
_COUNCIL_THRESHOLDS = {
    "conservative": {"trend_aligned_threshold": 3.0, "counter_trend_threshold": 3.5},
    "balanced":     {"trend_aligned_threshold": 2.5, "counter_trend_threshold": 3.0},
    "aggressive":   {"trend_aligned_threshold": 2.0, "counter_trend_threshold": 2.5},
    "scalper":      {"trend_aligned_threshold": 1.8, "counter_trend_threshold": 2.2},
}


def get_aggregator_preset(asset_key: str, preset_name: str) -> dict:
    presets = AGGREGATOR_PRESETS.get(asset_key, _DEFAULT_PRESET)
    if preset_name not in presets:
        logger.warning(f"Unknown preset '{preset_name}' for {asset_key}, using 'balanced'")
        preset_name = "balanced"
    cfg = presets[preset_name].copy()
    # Inject council thresholds so InstitutionalCouncilAggregator picks them up
    cfg.update(_COUNCIL_THRESHOLDS.get(preset_name, _COUNCIL_THRESHOLDS["balanced"]))
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Robust CSV loader — handles different column naming conventions
# ─────────────────────────────────────────────────────────────────────────────
def load_ohlcv_csv(filepath: str) -> pd.DataFrame:
    """
    Load an OHLCV CSV with flexible column handling.
    Normalises the datetime index so backtrader can consume it.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Find the datetime column (first column with valid datetime-parseable values)
    date_col = None
    for col in df.columns:
        if col in ("timestamp", "date", "datetime", "time"):
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample)
                    date_col = col
                    break
                except Exception:
                    continue

    if date_col is None:
        raise ValueError(f"No datetime column found in {filepath}. Columns: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df.index = df.index.tz_localize(None)  # backtrader doesn't handle tz-aware dates
    df = df.sort_index()

    # Ensure required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {filepath}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required))
    return df[["open", "high", "low", "close", "volume"]]


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# BacktestGovernor — point-in-time 4H regime context with no lookahead
# ─────────────────────────────────────────────────────────────────────────────
class BacktestGovernor:
    """
    Reads the asset's 4H CSV, pre-computes EMA-based regime labels for every
    bar, and provides point-in-time governor_data dicts that match the exact
    structure returned by MTFIntegration.get_regime_for_trading() in live
    trading.

    Zero lookahead: for each 1H bar at timestamp T, we return the regime
    from the latest 4H bar whose timestamp is ≤ T.
    """

    EMA_FAST      = 20
    EMA_SLOW      = 50
    EMA_BASELINE  = 200
    SLOPE_WINDOW  = 5        # bars to measure EMA200 slope direction
    SLOPE_THRESH  = 0.0003   # minimum % change per bar to classify as directional

    def __init__(self, csv_path: str, asset_name: str):
        self.asset_name = asset_name
        raw = load_ohlcv_csv(csv_path)
        if raw.empty:
            logger.warning(f"[GOVERNOR] ⚠️  Empty 4H CSV for {asset_name}: {csv_path}")
        self._df = self._compute_regime(raw)
        logger.info(
            f"[GOVERNOR] ✅ {asset_name} — {len(self._df)} 4H bars loaded "
            f"({csv_path})"
        )

    # ── Internal regime computation ───────────────────────────────────────────

    def _compute_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy().sort_index()
        close = df["close"].values.astype(float)

        df["ema20"]  = ta.EMA(close, timeperiod=self.EMA_FAST)
        df["ema50"]  = ta.EMA(close, timeperiod=self.EMA_SLOW)
        df["ema200"] = ta.EMA(close, timeperiod=self.EMA_BASELINE)
        df["ema200_slope"] = pd.Series(df["ema200"].values, index=df.index).pct_change(
            periods=self.SLOPE_WINDOW
        )

        regimes, scores = [], []
        for _, row in df.iterrows():
            r, s = self._classify(row)
            regimes.append(r)
            scores.append(s)
        df["regime"]       = regimes
        df["regime_score"] = scores
        return df

    def _classify(self, row) -> tuple:
        close  = row["close"]
        e200   = row.get("ema200")
        e50    = row.get("ema50")
        slope  = row.get("ema200_slope")

        if pd.isna(e200) or pd.isna(slope):
            return "NEUTRAL", 0.0

        above_200  = close > e200
        slope_up   = slope >  self.SLOPE_THRESH
        slope_down = slope < -self.SLOPE_THRESH
        above_50   = (close > e50) if (not pd.isna(e50)) else above_200

        if above_200 and slope_up and above_50:
            return "BULLISH", 1.0
        if not above_200 and slope_down and not above_50:
            return "BEARISH", -1.0
        if above_200 and (slope_up or above_50):
            return "SLIGHTLY_BULLISH", 0.5
        if not above_200 and (slope_down or not above_50):
            return "SLIGHTLY_BEARISH", -0.5
        return "NEUTRAL", 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_regime_at(self, dt: datetime) -> dict:
        """
        Return governor_data for the 1H bar at `dt`.
        Uses the last completed 4H bar whose timestamp ≤ dt (no lookahead).
        """
        if self._df.empty:
            return self._neutral()

        # Normalise timezone on dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Normalise index timezone to match dt
        idx = self._df.index
        if hasattr(idx, "tz"):
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")

        # searchsorted gives the insertion point; subtract 1 for "last bar ≤ dt"
        last_pos = int(idx.searchsorted(dt, side="right")) - 1
        if last_pos < 0:
            return self._neutral()

        row = self._df.iloc[last_pos]
        regime     = row.get("regime", "NEUTRAL")
        score      = float(row.get("regime_score", 0.0))
        ema200     = row.get("ema200")
        ema50      = row.get("ema50")
        is_bullish = regime in ("BULLISH", "SLIGHTLY_BULLISH")
        is_bearish = regime in ("BEARISH", "SLIGHTLY_BEARISH")
        confidence = abs(score)
        allow_ct   = regime in ("NEUTRAL", "SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH")
        trade_type = "TRANSITION" if regime == "NEUTRAL" else "TREND"

        # Pass a trailing 4H slice so aggregators that read df_4h get real data
        df_4h_slice = self._df.iloc[max(0, last_pos - 199) : last_pos + 1].copy()

        # council_aggregator._check_governor_filter() does:
        #   governor = governor_data.get('governor') or governor_data.get('full_regime_status')
        #   regime_name = getattr(governor, 'consensus_regime', ...)
        #   is_bullish  = getattr(governor, 'is_bullish', ...)
        #   is_bearish  = getattr(governor, 'is_bearish', ...)
        # So we must supply an object with those attributes, not just dict keys.
        proxy = SimpleNamespace(
            consensus_regime  = regime,
            is_bullish        = is_bullish,
            is_bearish        = is_bearish,
            is_bull           = is_bullish,
            score             = score,
            trade_type        = trade_type,
            confidence        = confidence,
            allow_counter_trend = allow_ct,
            ema_1d_200        = None,
            ema_4h_200        = float(ema200) if ema200 is not None and not pd.isna(ema200) else None,
            ema_4h_50         = float(ema50)  if ema50  is not None and not pd.isna(ema50)  else None,
            risk_level        = "low" if regime in ("BULLISH", "BEARISH") else "medium",
            volatility        = "high" if regime == "NEUTRAL" else "normal",
            reasoning         = f"4H EMA regime: {regime} (score={score:+.1f})",
        )

        return {
            "regime":              regime,
            "consensus_regime":    regime,
            "trade_type":          trade_type,
            "regime_score":        score,
            "is_bullish":          is_bullish,
            "is_bearish":          is_bearish,
            "is_bull":             is_bullish,
            "confidence":          confidence,
            "timeframe_agreement": confidence,
            "allow_counter_trend": allow_ct,
            "ema_4h_200":          float(ema200) if ema200 is not None and not pd.isna(ema200) else None,
            "ema_4h_50":           float(ema50)  if ema50  is not None and not pd.isna(ema50)  else None,
            "ema_1d_200":          None,
            "recommended_mode":    "council",
            "risk_level":          "low" if regime in ("BULLISH", "BEARISH") else "medium",
            "volatility":          "high" if regime == "NEUTRAL" else "normal",
            "max_positions":       3,
            "reasoning":           f"4H EMA regime: {regime} (score={score:+.1f})",
            "timestamp":           dt.isoformat(),
            "df_4h":               df_4h_slice,
            "h1_momentum_dir":     "FLAT",
            "h1_momentum_pct":     0.0,
            "h1_lower_highs":      False,
            "h1_higher_lows":      False,
            # BTC-specific fields — not available offline, set safe defaults
            "funding_rate_zscore": 0.0,
            "cvd_trend":           0,          # int: +1 buying, -1 selling, 0 flat/unknown
            # Object required by council_aggregator._check_governor_filter()
            "governor":            proxy,
            "full_regime_status":  proxy,
        }

    def _neutral(self) -> dict:
        proxy = SimpleNamespace(
            consensus_regime    = "NEUTRAL",
            is_bullish          = False,
            is_bearish          = False,
            is_bull             = False,
            score               = 0.0,
            trade_type          = "TRANSITION",
            confidence          = 0.0,
            allow_counter_trend = True,
            ema_1d_200          = None,
            ema_4h_200          = None,
            ema_4h_50           = None,
            risk_level          = "high",
            volatility          = "high",
            reasoning           = "No 4H data available yet",
        )
        return {
            "regime": "NEUTRAL", "consensus_regime": "NEUTRAL",
            "trade_type": "TRANSITION", "regime_score": 0.0,
            "is_bullish": False, "is_bearish": False, "is_bull": False,
            "confidence": 0.0, "timeframe_agreement": 0.0,
            "allow_counter_trend": True,
            "ema_4h_200": None, "ema_4h_50": None, "ema_1d_200": None,
            "recommended_mode": "council",
            "risk_level": "high", "volatility": "high",
            "max_positions": 0, "reasoning": "No 4H data available yet",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "df_4h": None, "h1_momentum_dir": "FLAT",
            "h1_momentum_pct": 0.0, "h1_lower_highs": False,
            "h1_higher_lows": False, "funding_rate_zscore": 0.0,
            "cvd_trend": 0,           # int: +1 buying, -1 selling, 0 flat/unknown
            "governor":           proxy,
            "full_regime_status": proxy,
        }


# Backtrader strategy
# ─────────────────────────────────────────────────────────────────────────────
class MLStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper supporting both PerformanceWeighted and
    InstitutionalCouncil aggregators.
    """

    params = (
        ("stop_loss_pct",          0.004),
        ("take_profit_pct",        0.008),   # 2×SL by default
        ("trailing_stop_pct",      0.015),
        ("risk_per_trade",         0.015),
        ("max_position_pct",       0.95),
        ("use_atr_sizing",         True),
        ("atr_period",             14),
        ("atr_multiplier",         1.8),
        ("lookback",               300),   # 200-bar EMA warmup + 100 signal buffer
        ("aggregator_preset",      "balanced"),
        ("aggregator_type",        "performance"),  # "performance" | "council"
        ("use_trailing_stop",      True),
        ("exit_on_opposite_signal", True),
        # AI validation
        ("use_ai_validation",              True),
        ("ai_sr_threshold",                0.015),
        ("ai_pattern_confidence",          0.50),
        ("ai_enable_adaptive",             True),
        ("ai_strong_signal_bypass",        0.85),
        ("ai_circuit_breaker_threshold",   0.70),
        ("use_macro_governor",             True),    # BacktestGovernor provides 4H regime data
        ("use_gatekeeper",                 True),    # BacktestGovernor provides 4H regime data
    )

    def __init__(self):
        self.asset_key = getattr(self.__class__, "asset_key", "BTC").upper()

        with open("config/config.json") as f:
            config = json.load(f)
        self.config = config

        mean_rev_model_path = f"models/mean_reversion_{self.asset_key.lower()}.pkl"
        trend_model_path    = f"models/trend_following_{self.asset_key.lower()}.pkl"
        ema_model_path      = f"models/ema_strategy_{self.asset_key.lower()}.pkl"

        mr_config  = config["strategy_configs"]["mean_reversion"][self.asset_key]
        mr_config["asset"] = self.asset_key   # Fix #15: MR scorecard label
        tf_config  = config["strategy_configs"]["trend_following"][self.asset_key]
        ema_config = config["strategy_configs"]["exponential_moving_averages"][self.asset_key]

        # ── Per-asset risk params — override global defaults from config ────────
        # The backtest strategy params are global defaults. For GOLD, USOIL, etc.
        # the ATR multiplier, trailing stop, and TP ratio differ significantly
        # from FX pairs and must be read from config to match live behaviour.
        _asset_risk = config.get("assets", {}).get(self.asset_key, {}).get("risk", {})
        self._atr_multiplier    = _asset_risk.get("atr_multiplier",    self.params.atr_multiplier)
        self._trailing_stop_pct = _asset_risk.get("trailing_stop_pct", self.params.trailing_stop_pct)
        self._tp_ratio          = _asset_risk.get("tp_ratio",          2.0)   # TP = stop × tp_ratio
        logger.info(
            f"[RISK] {self.asset_key}  ATR×{self._atr_multiplier}  "
            f"trail={self._trailing_stop_pct:.1%}  TP-ratio={self._tp_ratio}×"
        )

        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.trailing_stop_price = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry  = None

        self.mean_reversion  = MeanReversionStrategy(mr_config)
        self.trend_following = TrendFollowingStrategy(tf_config)
        self.ema_strategy    = EMAStrategy(ema_config)

        self.ai_validator = None
        if self.params.use_ai_validation:
            self.ai_validator = self._initialize_ai_layer()

        mr_loaded  = self.mean_reversion.load_model(mean_rev_model_path)
        tf_loaded  = self.trend_following.load_model(trend_model_path)
        ema_loaded = self.ema_strategy.load_model(ema_model_path)

        if not (mr_loaded and tf_loaded and ema_loaded):
            raise RuntimeError("Failed to load one or more strategy models")

        preset_name      = self.params.aggregator_preset
        confidence_config = get_aggregator_preset(self.asset_key, preset_name)

        # ── Aggregator selection ─────────────────────────────────────────────
        agg_type = self.params.aggregator_type.lower()
        if agg_type == "council":
            self.aggregator = InstitutionalCouncilAggregator(
                mean_reversion_strategy=self.mean_reversion,
                trend_following_strategy=self.trend_following,
                ema_strategy=self.ema_strategy,
                asset_type=self.asset_key,
                config=confidence_config,
                ai_validator=self.ai_validator if self.params.use_ai_validation else None,
                enable_detailed_logging=False,
                use_macro_governor=self.params.use_macro_governor,
                use_gatekeeper=self.params.use_gatekeeper,
            )
            logger.info(f"[AGGREGATOR] InstitutionalCouncilAggregator selected")
        else:
            self.aggregator = PerformanceWeightedAggregator(
                mean_reversion_strategy=self.mean_reversion,
                trend_following_strategy=self.trend_following,
                ema_strategy=self.ema_strategy,
                asset_type=self.asset_key,
                config=confidence_config,
                ai_validator=self.ai_validator if self.params.use_ai_validation else None,
                enable_ai_circuit_breaker=True,
                enable_detailed_logging=False,
                strong_signal_bypass_threshold=self.params.ai_strong_signal_bypass,
                use_macro_governor=self.params.use_macro_governor,
                use_gatekeeper=self.params.use_gatekeeper,
            )
            logger.info(f"[AGGREGATOR] PerformanceWeightedAggregator selected")

        self.order        = None
        self.trade_count  = 0
        self.signal_log   = []
        self.next_call_count = 0
        self.entry_price  = None
        self.stop_loss    = None
        self.take_profit  = None

        self.ai_stats = {
            "total_signals": 0, "ai_approved": 0, "ai_rejected": 0,
            "rejected_no_sr": 0, "rejected_no_pattern": 0,
        }

        # ── BacktestGovernor — 4H regime context ─────────────────────────────
        # Derive the 4H CSV filename from the same config-driven symbol used
        # for the 1H file, then load it into a BacktestGovernor instance.
        h1_filename = DATA_FILE_MAP.get(self.asset_key, f"{self.asset_key}_1h.csv")
        h4_filename = h1_filename.replace("_1h.csv", "_4h.csv")
        h4_path     = Path("data/raw") / h4_filename
        if h4_path.exists():
            self.governor = BacktestGovernor(str(h4_path), self.asset_key)
        else:
            logger.warning(
                f"[GOVERNOR] ⚠️  4H file not found: {h4_path}  "
                f"— governor disabled for {self.asset_key}"
            )
            self.governor = None

        logger.info("=" * 70)
        logger.info(f" Strategy: {self.asset_key} | Aggregator: {agg_type.upper()} | Preset: {preset_name}")
        logger.info(f" Governor: {'✅ 4H data loaded' if self.governor else '⚠️  disabled (no 4H file)'}")
        logger.info("=" * 70)

    def _initialize_ai_layer(self):
        try:
            models_dir   = Path("models/ai")
            model_path   = models_dir / "sniper_dual_timeframe_v1.weights.h5"
            mapping_path = models_dir / "sniper_dual_timeframe_v1_mapping.pkl"
            config_path  = models_dir / "sniper_dual_timeframe_v1_config.pkl"

            if not model_path.exists():
                logger.warning("[AI] Model not found — backtesting WITHOUT AI validation")
                return None

            with open(mapping_path, "rb") as f: pattern_map = pickle.load(f)
            with open(config_path,  "rb") as f: ai_config   = pickle.load(f)

            analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
            sniper  = OHLCSniper(input_shape=(15, 4), num_classes=ai_config["num_classes"])
            sniper.load_model(str(model_path))

            validator = HybridSignalValidator(
                analyst=analyst,
                sniper=sniper,
                pattern_id_map=pattern_map,
                sr_threshold_pct=self.params.ai_sr_threshold,
                pattern_confidence_min=self.params.ai_pattern_confidence,
                use_ai_validation=True,
                enable_adaptive_thresholds=self.params.ai_enable_adaptive,
                strong_signal_bypass_threshold=self.params.ai_strong_signal_bypass,
                circuit_breaker_threshold=self.params.ai_circuit_breaker_threshold,
                enable_detailed_logging=False,
            )
            logger.info("[AI] ✓ Validation layer initialized")
            return validator

        except Exception as e:
            logger.error(f"[AI] Failed to initialize: {e}")
            return None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                logger.info(f"✅ BUY  @ ${order.executed.price:.5f}  size={order.executed.size:.6f}")
            elif order.issell():
                self.entry_price = order.executed.price
                logger.info(f"✅ SELL @ ${order.executed.price:.5f}  size={order.executed.size:.6f}")

            if not self.position:
                self.entry_price  = None
                self.stop_loss    = None
                self.take_profit  = None
                self.trailing_stop_price       = None
                self.highest_price_since_entry = None
                self.lowest_price_since_entry  = None

            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl_pct = (trade.pnl / trade.value) * 100 if trade.value else 0
            logger.info(f"💰 TRADE CLOSED  PnL=${trade.pnl:.2f} ({pnl_pct:+.2f}%)  net=${trade.pnlcomm:.2f}")
            self.trade_count += 1

    def next(self):
        self.next_call_count += 1

        if self.order:
            return

        if len(self.data) < self.params.lookback:
            return

        try:
            current_price = self.data.close[0]

            # ── Exit logic for open positions ────────────────────────────────
            if self.position:
                is_long  = self.position.size > 0
                is_short = self.position.size < 0

                # Update trailing stop
                if self.params.use_trailing_stop:
                    self._update_trailing_stop(current_price, is_long, is_short)

                # Check stop-loss
                if self.stop_loss:
                    if (is_long and current_price <= self.stop_loss) or \
                       (is_short and current_price >= self.stop_loss):
                        self.order = self.close()
                        logger.info(f"🛑 STOP-LOSS @ ${current_price:.5f}")
                        return

                # Check take-profit
                if self.take_profit:
                    if (is_long and current_price >= self.take_profit) or \
                       (is_short and current_price <= self.take_profit):
                        self.order = self.close()
                        logger.info(f"🎯 TAKE-PROFIT @ ${current_price:.5f}")
                        return

                # Check trailing stop
                if self.params.use_trailing_stop and self.trailing_stop_price:
                    if (is_long and current_price <= self.trailing_stop_price) or \
                       (is_short and current_price >= self.trailing_stop_price):
                        self.order = self.close()
                        logger.info(f"📉 TRAILING-STOP @ ${current_price:.5f}")
                        return

            # ── Prepare OHLCV slice ──────────────────────────────────────────
            lb = self.params.lookback
            # bt.num2date converts backtrader's internal float date to a
            # timezone-naive Python datetime — required by _align_4h_to_1h.
            timestamps = [
                bt.num2date(d) for d in self.data.datetime.get(size=lb)
            ]
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open":   list(self.data.open.get(size=lb)),
                "high":   list(self.data.high.get(size=lb)),
                "low":    list(self.data.low.get(size=lb)),
                "close":  list(self.data.close.get(size=lb)),
                "volume": list(self.data.volume.get(size=lb)),
            })

            if len(df) < self.params.lookback:
                return

            # ── Get point-in-time 4H regime context (no lookahead) ──────────
            current_dt = self.data.datetime.datetime(0)
            if self.governor and self.params.use_macro_governor:
                governor_data = self.governor.get_regime_at(current_dt)
            else:
                governor_data = None

            signal, details = self.aggregator.get_aggregated_signal(
                df,
                current_regime  = governor_data.get("regime", "NEUTRAL") if governor_data else "NEUTRAL",
                is_bull_market  = governor_data.get("is_bull", True)      if governor_data else True,
                governor_data   = governor_data,
            )

            # Periodic log
            if self.next_call_count % 10 == 0:
                self.signal_log.append({
                    "date": self.data.datetime.date(0),
                    "price": current_price,
                    "signal": signal,
                    "details": details,
                })
                regime  = details.get("regime", "?")
                buy_s   = details.get("buy_score", 0)
                sell_s  = details.get("sell_score", 0)
                quality = details.get("signal_quality", 0)
                logger.info(
                    f"📍 {self.data.datetime.date(0)} | ${current_price:.5f} | {regime} | "
                    f"B/S:{buy_s:.2f}/{sell_s:.2f} | Q:{quality:.2f} | sig={signal:+d}"
                )

            # ── Entry logic ──────────────────────────────────────────────────
            if not self.position:
                if signal in (1, -1):
                    size = self._calculate_position_size()
                    if size > 0:
                        atr_value     = self.atr[0]
                        stop_distance = atr_value * self._atr_multiplier

                        if signal == 1:
                            self.order      = self.buy(size=size)
                            self.stop_loss   = current_price - stop_distance
                            self.take_profit = current_price + stop_distance * self._tp_ratio
                            self.highest_price_since_entry = current_price
                            self.lowest_price_since_entry  = current_price
                            logger.info(
                                f"🟢 BUY  @ ${current_price:.5f}  "
                                f"SL=${self.stop_loss:.5f}  TP=${self.take_profit:.5f}  "
                                f"size={size:.6f} | {details.get('reasoning','')}"
                            )
                        else:
                            self.order      = self.sell(size=size)
                            self.stop_loss   = current_price + stop_distance
                            self.take_profit = current_price - stop_distance * self._tp_ratio
                            self.highest_price_since_entry = current_price
                            self.lowest_price_since_entry  = current_price
                            logger.info(
                                f"🔴 SELL @ ${current_price:.5f}  "
                                f"SL=${self.stop_loss:.5f}  TP=${self.take_profit:.5f}  "
                                f"size={size:.6f} | {details.get('reasoning','')}"
                            )

            else:
                # Exit on confirmed opposite signal only
                if self.params.exit_on_opposite_signal:
                    is_long  = self.position.size > 0
                    is_short = self.position.size < 0
                    if (is_long and signal == -1) or (is_short and signal == 1):
                        self.order = self.close()
                        logger.info(
                            f"🔵 OPPOSITE-SIGNAL EXIT @ ${current_price:.5f} | "
                            f"{details.get('reasoning','')}"
                        )

        except Exception as e:
            logger.error(f"❌ Error in next(): {e}", exc_info=True)

    def _calculate_position_size(self) -> float:
        current_price = self.data.close[0]
        equity        = self.broker.getvalue()
        cash          = self.broker.getcash()
        atr_value     = self.atr[0]
        stop_distance = atr_value * self._atr_multiplier
        stop_pct      = stop_distance / current_price if current_price > 0 else self.params.stop_loss_pct

        risk_amount    = equity * self.params.risk_per_trade
        position_value = risk_amount / stop_pct if stop_pct > 0 else 0
        size           = position_value / current_price if current_price > 0 else 0

        max_size = (cash * self.params.max_position_pct) / current_price
        return max(min(size, max_size), 0)

    def _update_trailing_stop(self, price: float, is_long: bool, is_short: bool):
        """Ratchet trailing stop for both longs and shorts."""
        if is_long:
            if self.highest_price_since_entry is None:
                self.highest_price_since_entry = price
            else:
                self.highest_price_since_entry = max(self.highest_price_since_entry, price)
            new_trail = self.highest_price_since_entry * (1 - self._trailing_stop_pct)
            if self.trailing_stop_price is None or new_trail > self.trailing_stop_price:
                self.trailing_stop_price = new_trail

        elif is_short:
            if self.lowest_price_since_entry is None:
                self.lowest_price_since_entry = price
            else:
                self.lowest_price_since_entry = min(self.lowest_price_since_entry, price)
            new_trail = self.lowest_price_since_entry * (1 + self._trailing_stop_pct)
            if self.trailing_stop_price is None or new_trail < self.trailing_stop_price:
                self.trailing_stop_price = new_trail

    def stop(self):
        logger.info("=" * 70)
        logger.info(f"📊 {self.asset_key} | {self.params.aggregator_type.upper()} | {self.params.aggregator_preset}")
        logger.info(f"   Bars processed: {self.next_call_count} | Trades: {self.trade_count}")

        if self.signal_log:
            counts   = {-1: 0, 0: 0, 1: 0}
            reasons  = {}
            for log in self.signal_log:
                counts[log["signal"]] = counts.get(log["signal"], 0) + 1
                r = log["details"].get("reasoning", "unknown")
                reasons[r] = reasons.get(r, 0) + 1
            total = len(self.signal_log)
            logger.info(f"   Signals → BUY:{counts[1]} HOLD:{counts[0]} SELL:{counts[-1]}")
            top = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]
            for reason, cnt in top:
                logger.info(f"   Reasoning: {reason} ({cnt/total*100:.0f}%)")

        try:
            stats = self.aggregator.get_statistics()
            logger.info(f"   Signal rate: {stats.get('signal_rate',0):.1f}%  "
                        f"Buy: {stats.get('buy_rate',0):.1f}%  Sell: {stats.get('sell_rate',0):.1f}%")
        except Exception:
            pass
        logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Core run function
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    asset_key: str,
    aggregator_type: str = "performance",
    aggregator_preset: str = "balanced",
    use_ai: bool = True,
    use_macro_gov: bool = True,
    use_gatekeeper: bool = True,
    initial_capital: float = None,
    lookback: int = 300,           # bars fed to strategies per step (≥200 for EMA200 warmup)
) -> dict:
    """
    Run a single backtest. Returns a results dict for comparison tables.
    """
    asset_key = asset_key.upper()
    logger.info("=" * 70)
    logger.info(f"🚀 BACKTEST: {asset_key} | {aggregator_type.upper()} | preset={aggregator_preset}")
    logger.info("=" * 70)

    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("❌ config/config.json not found")
        sys.exit(1)

    filename = DATA_FILE_MAP.get(asset_key)
    if not filename:
        logger.error(f"❌ No data file mapping for {asset_key}. Known: {list(DATA_FILE_MAP.keys())}")
        sys.exit(1)

    data_path = f"data/raw/{filename}"
    try:
        df = load_ohlcv_csv(data_path)
        logger.info(f"✅ Loaded {len(df)} bars from {filename}")
        logger.info(f"   Date range: {df.index[0]} → {df.index[-1]}")
        logger.info(f"   Price range: {df['close'].min():.5f} → {df['close'].max():.5f}")
        if len(df) < 200:
            logger.warning(f"⚠️  Only {len(df)} bars — results may not be statistically meaningful")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"❌ Could not load data: {e}")
        sys.exit(1)

    cerebro = bt.Cerebro()

    bt_data = bt.feeds.PandasData(
        dataname=df,
        open="open", high="high", low="low", close="close",
        volume="volume", openinterest=-1,
    )
    cerebro.adddata(bt_data)

    MLStrategy.asset_key = asset_key
    cerebro.addstrategy(
        MLStrategy,
        aggregator_type=aggregator_type,
        aggregator_preset=aggregator_preset,
        use_ai_validation=use_ai,
        use_macro_governor=use_macro_gov,
        use_gatekeeper=use_gatekeeper,
        lookback=lookback,
    )

    cap = initial_capital or config["backtesting"]["initial_capital"]
    cerebro.broker.setcash(cap)
    cerebro.broker.setcommission(commission=config["backtesting"]["commission_pct"])

    asset_cfg    = config.get("assets", {}).get(asset_key, {})
    slippage_pct = asset_cfg.get("backtest_slippage_pct",
                                 config["backtesting"].get("slippage_pct", 0.0005))
    cerebro.broker.set_slippage_perc(slippage_pct)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe",   timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    logger.info(f"💵 Starting capital: ${cap:,.2f}")
    results  = cerebro.run()
    strat    = results[0]
    final_val = cerebro.broker.getvalue()

    total_return = (final_val - cap) / cap * 100
    sharpe_raw   = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    drawdown_raw = strat.analyzers.drawdown.get_analysis()
    trades_raw   = strat.analyzers.trades.get_analysis()

    try:
        max_dd = drawdown_raw.max.drawdown
    except (KeyError, AttributeError):
        max_dd = 0

    try:
        closed   = trades_raw.total.closed
        won      = trades_raw.won.total
        lost     = trades_raw.lost.total
        win_rate = (won / closed * 100) if closed > 0 else 0
        avg_pnl  = trades_raw.pnl.net.average
    except (KeyError, AttributeError):
        closed = won = lost = 0
        win_rate = 0.0
        avg_pnl  = 0.0

    # Print summary
    logger.info("=" * 70)
    logger.info(f"📊 RESULTS  {asset_key} | {aggregator_type.upper()} | {aggregator_preset}")
    logger.info(f"   Capital: ${cap:,.2f} → ${final_val:,.2f}  ({total_return:+.2f}%)")
    logger.info(f"   Sharpe: {sharpe_raw:.2f}" if sharpe_raw else "   Sharpe: N/A")
    logger.info(f"   Max Drawdown: {max_dd:.2f}%")
    logger.info(f"   Trades: {closed}  Win rate: {win_rate:.1f}%  Avg PnL: ${avg_pnl:.2f}")
    if closed == 0:
        logger.warning("⚠️  NO TRADES — try --preset aggressive or --no-ai")
    logger.info("=" * 70)

    return {
        "asset":          asset_key,
        "aggregator":     aggregator_type,
        "preset":         aggregator_preset,
        "initial":        cap,
        "final":          final_val,
        "return_pct":     total_return,
        "sharpe":         sharpe_raw,
        "max_drawdown":   max_dd,
        "trades":         closed,
        "win_rate":       win_rate,
        "avg_pnl":        avg_pnl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison runner — Performance vs Council, side-by-side table
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(
    assets: list,
    preset: str = "balanced",
    use_ai: bool = True,
    use_macro_gov: bool = True,
    use_gatekeeper: bool = True,
    initial_capital: float = None,
    lookback: int = 300,
):
    """
    Run Performance then Council on each asset and print a comparison table.
    """
    all_results = []

    for asset in assets:
        for agg in ("performance", "council"):
            r = run_backtest(
                asset_key=asset,
                aggregator_type=agg,
                aggregator_preset=preset,
                use_ai=use_ai,
                use_macro_gov=use_macro_gov,
                use_gatekeeper=use_gatekeeper,
                initial_capital=initial_capital,
                lookback=lookback,
            )
            all_results.append(r)

    # Build comparison table — uses logger.info so output goes to both
    # console and the per-run log file set up in _setup_run_log().
    logger.info("")
    logger.info("=" * 100)
    logger.info("  AGGREGATOR COMPARISON TABLE")
    logger.info("=" * 100)
    header = f"{'Asset':<8} {'Aggregator':<12} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'Trades':>7} {'WinRate%':>9} {'AvgPnL$':>9}"
    logger.info(header)
    logger.info("-" * 100)

    prev_asset = None
    for r in all_results:
        if r["asset"] != prev_asset and prev_asset is not None:
            logger.info("")   # blank line between assets
        prev_asset = r["asset"]

        sharpe_str = f"{r['sharpe']:>7.2f}" if r["sharpe"] else "    N/A"
        logger.info(
            f"{r['asset']:<8} {r['aggregator']:<12} "
            f"{r['return_pct']:>8.2f} {sharpe_str} "
            f"{r['max_drawdown']:>7.2f} {r['trades']:>7} "
            f"{r['win_rate']:>9.1f} {r['avg_pnl']:>9.2f}"
        )

    logger.info("=" * 100)

    # Winner summary
    logger.info("")
    logger.info("  WINNER BY ASSET")
    logger.info("-" * 50)
    for i in range(0, len(all_results), 2):
        perf = all_results[i]
        coun = all_results[i + 1]
        # Compare by return, break ties with win rate
        if perf["return_pct"] > coun["return_pct"]:
            winner = "PERFORMANCE"
            margin = perf["return_pct"] - coun["return_pct"]
        elif coun["return_pct"] > perf["return_pct"]:
            winner = "COUNCIL    "
            margin = coun["return_pct"] - perf["return_pct"]
        else:
            winner = "TIE        "
            margin = 0
        logger.info(f"  {perf['asset']:<8} → {winner}  (+{margin:.2f}% return)")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest Council vs Performance aggregator across assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single asset, performance aggregator (default)
  python backtest.py --asset EURUSD

  # Council aggregator
  python backtest.py --asset EURUSD --aggregator council

  # Compare both aggregators on one asset
  python backtest.py --asset GBPAUD --compare-both

  # Compare both on multiple assets
  python backtest.py --assets EURUSD GBPAUD USTEC --compare-both

  # No AI filter, aggressive preset
  python backtest.py --asset BTC --preset aggressive --no-ai

  # Full sweep of all assets
  python backtest.py --assets BTC GOLD EURUSD GBPAUD GBPUSD USOIL USTEC USDJPY --compare-both
        """,
    )
    parser.add_argument(
        "--asset", type=str, default=None,
        choices=SUPPORTED_ASSETS,
        help="Single asset to backtest",
    )
    parser.add_argument(
        "--assets", type=str, nargs="+", default=None,
        choices=SUPPORTED_ASSETS,
        metavar="ASSET",
        help="One or more assets (for --compare-both sweep)",
    )
    parser.add_argument(
        "--aggregator", type=str, default="performance",
        choices=["performance", "council"],
        help="Aggregator to use (ignored when --compare-both is set)",
    )
    parser.add_argument(
        "--preset", type=str, default="balanced",
        choices=["conservative", "balanced", "aggressive", "scalper"],
        help="Signal threshold preset",
    )
    parser.add_argument(
        "--compare-both", action="store_true",
        help="Run BOTH aggregators and print a side-by-side comparison table",
    )
    parser.add_argument("--no-ai",          action="store_true", help="Disable AI validation layer")
    # Governor and Gatekeeper are OFF by default in backtest (no live MTF governor is running).
    # Pass --with-gov / --with-gatekeeper to re-enable them (e.g. to replicate live behaviour).
    parser.add_argument("--no-gov",         action="store_true", help="[legacy] Disable Macro Governor (already off by default)")
    parser.add_argument("--with-gov",       action="store_true", help="Enable Macro Governor (requires live MTF data)")
    parser.add_argument("--no-gatekeeper",  action="store_true", help="[legacy] Disable Gatekeeper (already off by default)")
    parser.add_argument("--with-gatekeeper",action="store_true", help="Enable Gatekeeper (requires live MTF data)")
    parser.add_argument("--capital",        type=float, default=None, help="Override starting capital ($)")
    parser.add_argument("--lookback",       type=int,   default=300,
                        help="Bars of OHLCV history fed to strategies per step "
                             "(≥200 required for EMA200 warmup; default: 300)")
    parser.add_argument("--diagnose",       action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    if args.diagnose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Set up per-run log file ───────────────────────────────────────────────
    run_tag  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _setup_run_log(run_tag)
    logger.info(f"📝 Logging this run to: {log_path}")

    if args.diagnose:
        logger.info("🔍 DIAGNOSTIC MODE ENABLED")

    # Resolve asset list
    if args.assets:
        asset_list = [a.upper() for a in args.assets]
    elif args.asset:
        asset_list = [args.asset.upper()]
    else:
        asset_list = ["BTC"]   # default

    # Governor and gatekeeper default to ON — BacktestGovernor feeds real 4H data.
    # Pass --no-gov / --no-gatekeeper to disable (e.g. for a quick raw signal test).
    use_gov        = not args.no_gov
    use_gatekeeper = not args.no_gatekeeper

    if args.compare_both:
        run_comparison(
            assets=asset_list,
            preset=args.preset,
            use_ai=not args.no_ai,
            use_macro_gov=use_gov,
            use_gatekeeper=use_gatekeeper,
            initial_capital=args.capital,
            lookback=args.lookback,
        )
    else:
        for asset in asset_list:
            run_backtest(
                asset_key=asset,
                aggregator_type=args.aggregator,
                aggregator_preset=args.preset,
                use_ai=not args.no_ai,
                use_macro_gov=use_gov,
                use_gatekeeper=use_gatekeeper,
                initial_capital=args.capital,
                lookback=args.lookback,
            )

    logger.info(f"💾 Full run log saved → {log_path}")
