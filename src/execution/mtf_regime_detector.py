"""
Multi-Timeframe Regime Detector - REFACTORED FOR STRICT OPTION 2
================================================================
Enforces strict trend hierarchy using Dual-Structure Logic Gates.
Focuses on 1H and 4H timeframes, guided by explicit EMA relationships.
"""

import pandas as pd
import numpy as np
import logging
import talib as ta
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path


from enum import Enum

logger = logging.getLogger(__name__)

# Centralized Indicator Constants
FAST_EMA = 20
SLOW_EMA = 50
BASELINE_EMA = 200

class TradeType(Enum):
    """Supported institutional trade types."""
    TREND = "TREND"
    SCALP = "SCALP"
    V_SHAPE = "V_SHAPE"
    NEUTRAL = "NEUTRAL"


@dataclass
class GovernorStatus:
    """Simplified Governor status for the Constitution Gate."""
    is_bullish: bool
    is_bearish: bool
    reasoning: str
    ema_200: Optional[float] = None
    trade_type: TradeType = TradeType.TREND


@dataclass
class RegimeStatus:
    """Simplified aggregated regime status with granular details."""
    asset: str
    score: float
    is_bullish: bool
    is_bearish: bool
    reasoning: str
    timestamp: datetime
    consensus_regime: str 
    # Macro EMAs for VTM anchoring
    ema_1d_200: Optional[float] = None
    ema_4h_200: Optional[float] = None
    ema_4h_50: Optional[float] = None
    trade_type: TradeType = TradeType.TREND
    
    # Granular Timeframe Data
    timeframe_data: Dict[str, Dict] = field(default_factory=dict)
    df_4h: Optional[pd.DataFrame] = None


class MultiTimeFrameRegimeDetector:
    """
    Analyzes market regime across 1H and 4H timeframes using Dual-Structure Logic Gates.
    """

    def __init__(self, data_manager, asset_type: str = "BTC"):
        """Initialize detector"""
        self.data_manager = data_manager
        self.asset_type = asset_type.upper()

        # Governor thresholds (simplified, only relevant for _analyze_governor for Constitution Gate)
        self.governor_thresholds = {
            "min_required_bars_1d": 100, # Lowered from 220 to allow faster startup
            "ema_slope_positive": 0.0005, # Positive slope threshold for 200 EMA
        }

        # Cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

        logger.info(f"[MTF REGIME] Initialized for {asset_type} (Strict Option 2)")
        logger.info(f"  EMAs: FAST={FAST_EMA}, SLOW={SLOW_EMA}, BASELINE={BASELINE_EMA}")
        logger.info(f"  Logic Gates: Constitution (4H {BASELINE_EMA} EMA), General (4H {SLOW_EMA} EMA), Captain (1H {SLOW_EMA} EMA)")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate essential indicators for trend following (EMA, ADX, RSI).
        """
        if df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        
        # EMA Calculations
        df_copy[f"ema_{FAST_EMA}"] = ta.EMA(df_copy["close"], timeperiod=FAST_EMA)
        df_copy[f"ema_{SLOW_EMA}"] = ta.EMA(df_copy["close"], timeperiod=SLOW_EMA)
        
        # ✅ TASK 17: EMA-200 Burn-in (Strict 400-bar minimum)
        if len(df_copy) < 400:
            # We enforce strict burn-in to ensure indicator stability.
            logger.warning(f"[REGIME] ⚠️ CRITICAL: Insufficient bars for EMA-200 burn-in ({len(df_copy)}/400).")
            return pd.DataFrame()

        df_copy[f"ema_{BASELINE_EMA}"] = ta.EMA(df_copy["close"], timeperiod=BASELINE_EMA)
        
        # Trend strength and momentum
        df_copy["adx"] = ta.ADX(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=14)
        df_copy["rsi"] = ta.RSI(df_copy["close"], timeperiod=14)
        
        # Support for directional indicators (Trend)
        plus_di = ta.PLUS_DI(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=14)
        minus_di = ta.MINUS_DI(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=14)
        
        df_copy["trend_dir"] = np.where(plus_di > minus_di, "UP", "DOWN")
        
        return df_copy

    def _fetch_data_from_csv(
        self, symbol: str, timeframe_str: str, exchange: str
    ) -> pd.DataFrame:
        """
        Fetches data from the 'data/raw/' local directory.
        Falls back to API if CSV doesn't exist or is stale.
        """
        # Determine CSV file path dynamically based on symbol
        # Example: BTCUSDT_1h.csv, USTECm_1h.csv
        csv_file = Path(f"data/raw/{symbol}_{timeframe_str}.csv")

        # Try to read from CSV
        if csv_file.exists():
            try:
                logger.info(f"[CSV] Reading {self.asset_type} {timeframe_str} from {csv_file}")
                # ✅ FIX: Explicitly load and then set index to handle different CSV structures
                df = pd.read_csv(csv_file)
                
                # Find timestamp column
                date_col = None
                for col in ['date', 'timestamp', 'time', 'datetime']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
                    df = df.dropna(subset=[date_col])
                    df.set_index(date_col, inplace=True)
                else:
                    logger.warning(f"[CSV] No timestamp column found in {csv_file}")
                    return self._fetch_data(symbol, timeframe_str, exchange)

                # Check data freshness (should be within last 4 hours)
                if df.empty:
                    return self._fetch_data(symbol, timeframe_str, exchange)
                    
                latest_date = df.index[-1]
                now = pd.Timestamp.now(tz='UTC')
                hours_old = (now - latest_date).total_seconds() / 3600

                if hours_old > 4 or pd.isna(hours_old):
                    logger.warning(f"[CSV] Data is {hours_old:.1f} hours old (Stale) - FALLING BACK TO API")
                    return self._fetch_data(symbol, timeframe_str, exchange)

                logger.info(f"[CSV] ✓ Loaded {len(df)} bars from CSV")
                logger.info(f"[CSV]   Range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
                logger.info(f"[CSV]   Age: {hours_old:.1f} hours old")

                # ✨ B.1: Drop incomplete current-period candle (Consistency fix)
                # Reason: Prevents regime "jitter" and repainting.
                if not df.empty:
                    _now = datetime.now(timezone.utc)
                    if timeframe_str == "1h":
                        _floor = _now.replace(minute=0, second=0, microsecond=0)
                    elif timeframe_str == "4h":
                        _h = (_now.hour // 4) * 4
                        _floor = _now.replace(hour=_h, minute=0, second=0, microsecond=0)
                    elif timeframe_str == "1d":
                        _floor = _now.replace(hour=0, minute=0, second=0, microsecond=0)
                    else:
                        _floor = _now
                    
                    if df.index[-1] >= _floor:
                        df = df.iloc[:-1]

                return df

            except Exception as e:
                logger.warning(f"[CSV] Failed to read {csv_file}: {e}")
                logger.info(f"[CSV] Falling back to API fetch")

        else:
            logger.info(f"[CSV] File not found: {csv_file}")
            logger.info(f"[CSV] Falling back to API fetch")

        # Fallback: Use original API fetch method
        return self._fetch_data(symbol, timeframe_str, exchange)

    def _fetch_data(
        self, symbol: str, timeframe_str: str, exchange: str
    ) -> pd.DataFrame:
        """
        Fetch historical data for the specified timeframe from API.
        """
        end_time = datetime.now(timezone.utc)

        # Determine lookback based on timeframe string
        if timeframe_str == "1h":
            lookback_days = 60  # ~1440 bars
        elif timeframe_str == "4h":
            lookback_days = 120 # ~720 bars
        elif timeframe_str == "1d":
            lookback_days = 500 # ~500 bars
        else:
            logger.error(f"Unsupported timeframe_str: {timeframe_str}")
            return pd.DataFrame() # Return empty for unsupported

        start_time = end_time - timedelta(days=lookback_days)

        if exchange == "binance":
            df = self.data_manager.fetch_binance_data(
                symbol=symbol,
                interval=timeframe_str,
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:  # mt5
            df = self.data_manager.fetch_mt5_data(
                symbol=symbol,
                timeframe=timeframe_str.upper().replace('H', 'H'), # e.g., '1h' -> 'H1'
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        
        df = self.data_manager.clean_data(df)

        # ✨ B.1: Drop incomplete current-period candle (Consistency fix)
        # Reason: Prevents regime "jitter" and repainting. Ensures we only
        # analyze CONFIRMED market structures, identical to the signal logic.
        if not df.empty:
            _now = datetime.now(timezone.utc)
            if timeframe_str == "1h":
                _floor = _now.replace(minute=0, second=0, microsecond=0)
            elif timeframe_str == "4h":
                # MT5/Binance 4H bars start at 0, 4, 8, 12, 16, 20 UTC
                _h = (_now.hour // 4) * 4
                _floor = _now.replace(hour=_h, minute=0, second=0, microsecond=0)
            elif timeframe_str == "1d":
                _floor = _now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                _floor = _now # fallback
                
            if df.index[-1] >= _floor:
                df = df.iloc[:-1]
                
        return df


    def _analyze_governor(self, symbol: str, exchange: str) -> GovernorStatus:
        """
        Constitution Gate: Analyzes the Daily 200 EMA for macro trend.
        """
        try:
            df_daily = self._fetch_data_from_csv(symbol, "1d", exchange)

            # ✅ TASK 17: EMA-200 Burn-in (Strict 400-bar minimum)
            if len(df_daily) < 400:
                logger.warning(
                    f"[CONSTITUTION] CRITICAL: Insufficient daily data: {len(df_daily)} bars (need 400+). "
                    "Cannot establish reliable macro trend. Defaulting to neutral."
                )
                return GovernorStatus(
                    is_bullish=False, is_bearish=False, reasoning="Insufficient 1D data (<400 bars)", ema_200=None
                )

            df_daily = self._calculate_indicators(df_daily)
            latest = df_daily.iloc[-1]
            current_price = latest["close"]
            ema_200 = latest[f"ema_{BASELINE_EMA}"]

            if pd.isna(ema_200):
                 logger.warning("[CONSTITUTION] 1D 200 EMA is NaN. Defaulting to neutral.")
                 return GovernorStatus(
                    is_bullish=False, is_bearish=False, reasoning="1D 200 EMA is NaN", ema_200=None
                 )

            # Check slope of 200 EMA (use available data if less than 20)
            slope_lookback = min(20, len(df_daily) - 1)
            if slope_lookback < 5: 
                ema_slope = 0.0 # Not enough data for meaningful slope
            else:
                ema_200_series = df_daily[f"ema_{BASELINE_EMA}"]
                ema_slope = (ema_200_series.iloc[-1] - ema_200_series.iloc[-slope_lookback]) / ema_200_series.iloc[-slope_lookback]


            is_bullish = (current_price > ema_200) and (ema_slope > self.governor_thresholds["ema_slope_positive"])
            is_bearish = (current_price < ema_200) and (ema_slope < -self.governor_thresholds["ema_slope_positive"])

            trade_type = TradeType.NEUTRAL
            if is_bullish or is_bearish:
                trade_type = TradeType.TREND

            reasoning = (
                f"Price {'above' if is_bullish else 'below' if is_bearish else 'near'} 200 EMA "
                f"with {'positive' if ema_slope > 0 else 'negative' if ema_slope < 0 else 'flat'} slope."
            )

            return GovernorStatus(
                is_bullish=is_bullish, is_bearish=is_bearish, reasoning=reasoning, ema_200=ema_200, trade_type=trade_type
            )

        except Exception as e:
            logger.error(f"[CONSTITUTION] Error analyzing 1D Governor: {e}", exc_info=True)
            return GovernorStatus(
                is_bullish=False, is_bearish=False, reasoning=f"Error: {str(e)}", ema_200=None
            )

    def get_aggregated_regime_score(
        self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, governor_status: GovernorStatus, asset_type: str
    ) -> RegimeStatus:
        """
        Implements the INSTITUTIONAL 5-TIER REGIME MODEL.
        Anchored to 1D 200 EMA (Constitution Gate).

        Tiers:
        1. BULLISH: Price > 1D 200 & Price > 4H 50 & Price > 1H 50
        2. SLIGHTLY_BULLISH: Price > 1D 200 BUT Price < (4H 50 OR 1H 50)
        3. NEUTRAL: Price near 1D 200 EMA or mixed signals
        4. SLIGHTLY_BEARISH: Price < 1D 200 BUT Price > (4H 50 OR 1H 50)
        5. BEARISH: Price < 1D 200 & Price < 4H 50 & Price < 1H 50

        STRICT LOCK: Price > 1D 200 blocks all BEARISH states.
                    Price < 1D 200 blocks all BULLISH states.
        """
        score = 0.0
        reasons = []

        # Ensure dataframes are not empty
        if df_1h.empty or df_4h.empty:
            reasons.append("Insufficient 1H or 4H data.")
            return RegimeStatus(
                asset=asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=", ".join(reasons), timestamp=datetime.now(timezone.utc), consensus_regime="NEUTRAL"
            )

        # Calculate indicators for both timeframes
        df_1h_with_ema = self._calculate_indicators(df_1h)
        df_4h_with_ema = self._calculate_indicators(df_4h)

        if df_1h_with_ema.empty or df_4h_with_ema.empty:
            reasons.append("Failed to calculate EMAs due to insufficient data.")
            return RegimeStatus(
                asset=asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=", ".join(reasons), timestamp=datetime.now(timezone.utc), consensus_regime="NEUTRAL"
            )

        latest_1h = df_1h_with_ema.iloc[-1]
        latest_4h = df_4h_with_ema.iloc[-1]

        price_1h = latest_1h["close"]
        price_4h = latest_4h["close"]
        ema_4h_baseline = latest_4h[f"ema_{BASELINE_EMA}"]
        ema_4h_slow = latest_4h[f"ema_{SLOW_EMA}"]
        ema_1h_slow = latest_1h[f"ema_{SLOW_EMA}"]

        # ✅ TASK 21: Rolling Quantile Thresholds (Phase 3)
        # Reason: Fixed thresholds are blind to vol cycles.
        all_dists = (abs(df_4h_with_ema["close"] - df_4h_with_ema[f"ema_{BASELINE_EMA}"]) / df_4h_with_ema[f"ema_{BASELINE_EMA}"]).tail(100)
        
        # Bullish needs more extension (0.65), Bearish slightly less (0.35)
        thresh_bull = all_dists.quantile(0.65)
        thresh_bear = all_dists.quantile(0.35)
        
        # Clamp to realistic institutional bounds [0.05% to 0.40%]
        thresh_bull = max(0.0005, min(0.0040, thresh_bull))
        thresh_bear = max(0.0005, min(0.0040, thresh_bear))

        ema_dist_4h = (price_4h - ema_4h_baseline) / ema_4h_baseline

        # Macro Trend from Governor (1D 200 EMA)
        macro_bullish = governor_status.is_bullish
        macro_bearish = governor_status.is_bearish

        # Intermediate & Short-term alignment
        is_extended_bull = ema_dist_4h > thresh_bull
        is_extended_bear = ema_dist_4h < -thresh_bear
        
        above_4h_200 = price_4h > ema_4h_baseline
        above_4h_50 = price_4h > ema_4h_slow
        above_1h_50 = price_1h > ema_1h_slow

        # 5-TIER LOGIC (Regime-Adaptive)
        if macro_bullish:
            # BLOCK BEARISH STATES
            if above_4h_50 and above_1h_50 and is_extended_bull:
                consensus_regime = "BULLISH"
                score = 1.0
                reasons.append(f"Macro BULLISH: Strong extension ({ema_dist_4h:.4%} > {thresh_bull:.4%})")
            elif above_4h_200:
                consensus_regime = "SLIGHTLY_BULLISH"
                score = 0.5
                reasons.append("Macro BULLISH: Above 4H 200 (Not yet extended)")
            else:
                # 1D says bullish but 4H hasn't confirmed yet — transitional bullish zone
                if not above_4h_50 and not above_1h_50:
                    consensus_regime = "NEUTRAL"
                    score = 0.0
                    reasons.append("Macro BULLISH but 4H/1H Correction: Transitional NEUTRAL zone.")
                else:
                    consensus_regime = "SLIGHTLY_BULLISH"
                    score = 0.5
                    reasons.append("Macro BULLISH: Below 4H 200 but 1D structure intact — transitional bullish zone.")
        
        elif macro_bearish:
            # BLOCK BULLISH STATES
            if not above_4h_50 and not above_1h_50 and is_extended_bear:
                consensus_regime = "BEARISH"
                score = -1.0
                reasons.append(f"Macro BEARISH: Strong extension ({ema_dist_4h:.4%} < -{thresh_bear:.4%})")
            elif not above_4h_200:
                consensus_regime = "SLIGHTLY_BEARISH"
                score = -0.5
                reasons.append("Macro BEARISH: Below 4H 200 (Not yet extended)")
            else:
                # 1D says bearish but 4H hasn't confirmed yet — transitional bearish zone
                if above_4h_50 and above_1h_50:
                    consensus_regime = "NEUTRAL"
                    score = 0.0
                    reasons.append("Macro BEARISH but 4H/1H Recovery: Transitional NEUTRAL zone.")
                else:
                    consensus_regime = "SLIGHTLY_BEARISH"
                    score = -0.5
                    reasons.append("Macro BEARISH: Above 4H 200 but 1D structure broken — transitional bearish zone.")
        
        else:
            consensus_regime = "NEUTRAL"
            score = 0.0
            reasons.append("Constitution: 1D 200 EMA is neutral/unclear.")

        is_bullish = consensus_regime in ["BULLISH", "SLIGHTLY_BULLISH"]
        is_bearish = consensus_regime in ["BEARISH", "SLIGHTLY_BEARISH"]

        trade_type = TradeType.NEUTRAL
        if is_bullish or is_bearish:
            trade_type = TradeType.TREND

        final_reasoning = f"Institutional {consensus_regime} regime. " + ", ".join(reasons)

        # ── 1H Session Momentum ─────────────────────────────────────────────────
        # Captures INTRADAY direction (last 6 closes on 1H), separate from the
        # structural above_1h_50 check.  Used by the AI validator to soft-pass
        # SELL signals when 1H is actively declining even inside a bullish 4H
        # regime, and to BLOCK BUY signals when 1H is rolling over.
        # Values: "UP" (rising), "DOWN" (falling), "FLAT" (< 0.1% change)
        try:
            _h1_closes = df_1h_with_ema["close"].iloc[-6:]
            _h1_raw_mom = (_h1_closes.iloc[-1] - _h1_closes.iloc[0]) / max(_h1_closes.iloc[0], 1e-9)
            h1_momentum_pct = float(_h1_raw_mom)
            if _h1_raw_mom > 0.001:
                h1_momentum_dir = "UP"
            elif _h1_raw_mom < -0.001:
                h1_momentum_dir = "DOWN"
            else:
                h1_momentum_dir = "FLAT"
            # Also compute whether 1H is making lower highs over the last 4 bars
            # (more reliable bearish filter than just close slope)
            _h1_highs = df_1h_with_ema["high"].iloc[-4:]
            h1_lower_highs = bool(_h1_highs.iloc[-1] < _h1_highs.iloc[-2] < _h1_highs.iloc[-3])
            _h1_lows = df_1h_with_ema["low"].iloc[-4:]
            h1_higher_lows = bool(_h1_lows.iloc[-1] > _h1_lows.iloc[-2] > _h1_lows.iloc[-3])
        except Exception:
            h1_momentum_pct = 0.0
            h1_momentum_dir = "FLAT"
            h1_lower_highs = False
            h1_higher_lows = False

        # ✨ NEW: Populate granular timeframe data for database/dashboard
        timeframe_data = {
            "1h": {
                "regime": consensus_regime if above_1h_50 == is_bullish else "NEUTRAL",
                "confidence": abs(score),
                "adx": float(latest_1h["adx"]) if "adx" in latest_1h else None,
                "rsi": float(latest_1h["rsi"]) if "rsi" in latest_1h else None,
                "trend_direction": latest_1h["trend_dir"] if "trend_dir" in latest_1h else "N/A",
                "momentum_dir": h1_momentum_dir,
                "momentum_pct": round(h1_momentum_pct * 100, 3),
                "lower_highs": h1_lower_highs,
                "higher_lows": h1_higher_lows,
            },
            "4h": {
                "regime": consensus_regime if above_4h_200 == is_bullish else "NEUTRAL",
                "confidence": abs(score),
                "adx": float(latest_4h["adx"]) if "adx" in latest_4h else None,
                "rsi": float(latest_4h["rsi"]) if "rsi" in latest_4h else None,
                "trend_direction": latest_4h["trend_dir"] if "trend_dir" in latest_4h else "N/A"
            },
            "1d": {
                "regime": "BULLISH" if macro_bullish else "BEARISH" if macro_bearish else "NEUTRAL",
                "confidence": 1.0 if (macro_bullish or macro_bearish) else 0.0,
                "adx": None, # Will be filled if we calculate 1D indicators
                "rsi": None,
                "trend_direction": "UP" if macro_bullish else "DOWN" if macro_bearish else "SIDEWAYS"
            }
        }

        return RegimeStatus(
            asset=asset_type,
            score=score,
            is_bullish=is_bullish,
            is_bearish=is_bearish,
            reasoning=final_reasoning,
            timestamp=datetime.now(timezone.utc),
            consensus_regime=consensus_regime,
            ema_1d_200=governor_status.ema_200,
            ema_4h_200=ema_4h_baseline,
            ema_4h_50=ema_4h_slow,
            trade_type=trade_type,
            timeframe_data=timeframe_data,
            df_4h=df_4h_with_ema # ✨ Pass 4H data with features
        )

    def analyze_regime(
        self,
        symbol: str,
        exchange: str = "binance",
        force_refresh: bool = False
    ) -> RegimeStatus:
        """
        Analyzes market regime across 1H, 4H, and 1D timeframes using Dual-Structure Logic Gates.
        Returns a simplified RegimeStatus object.
        """
        cache_key = f"{symbol}_{exchange}"
        if not force_refresh and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_duration:
                logger.debug(f"[MTF] Using cached result for {symbol}")
                return cached_result

        logger.info(f"\n{'='*70}")
        logger.info(f"[MTF REGIME] Analyzing {self.asset_type} ({symbol}) - Strict Option 2")
        logger.info(f"{'='*70}")

        # --- STEP 1: Constitution Gate (Daily 200 EMA) ---
        governor_status = self._analyze_governor(symbol, exchange)
        logger.info(f"[CONSTITUTION] Status: {'Bullish' if governor_status.is_bullish else 'Bearish' if governor_status.is_bearish else 'Neutral'}. Reason: {governor_status.reasoning}")

        # --- STEP 2: Fetch 1H and 4H data ---
        df_1h = pd.DataFrame()
        df_4h = pd.DataFrame()
        
        try:
            df_1h = self._fetch_data_from_csv(symbol, "1h", exchange)
            if df_1h.empty or len(df_1h) < SLOW_EMA + 1:
                raise ValueError("Insufficient 1H data for EMA calculation.")
        except Exception as e:
            logger.error(f"[MTF] Failed to get 1H data: {e}", exc_info=True)
            return RegimeStatus(asset=self.asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=f"Insufficient 1H data: {str(e)}", timestamp=datetime.now(timezone.utc), consensus_regime="NEUTRAL")

        try:
            df_4h = self._fetch_data_from_csv(symbol, "4h", exchange)
            # ✅ TASK 17: EMA-200 Burn-in (Strict 400-bar minimum)
            if df_4h.empty or len(df_4h) < 400:
                raise ValueError(f"CRITICAL: Insufficient 4H data for EMA-200 burn-in ({len(df_4h)}/400 bars)")
        except Exception as e:
            logger.error(f"[MTF] Failed to get 4H data: {e}", exc_info=True)
            return RegimeStatus(asset=self.asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=f"Insufficient 4H data: {str(e)}", timestamp=datetime.now(timezone.utc), consensus_regime="NEUTRAL")

        # --- STEP 3: Get Aggregated Regime Score ---
        regime_status = self.get_aggregated_regime_score(df_1h, df_4h, governor_status, self.asset_type)

        logger.info(f"\n[FINAL REGIME] Score: {regime_status.score:.2f}")
        logger.info(f"  Bullish: {regime_status.is_bullish}, Bearish: {regime_status.is_bearish}")
        logger.info(f"  Reasoning: {regime_status.reasoning}")
        logger.info(f"{'='*70}\n")

        self.cache[cache_key] = (regime_status, datetime.now(timezone.utc))
        return regime_status
