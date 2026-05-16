"""
VolumeOrderFlowStrategy — Orthogonal signal from money flow analysis.
Detects institutional accumulation/distribution independent of price trend.
Does NOT use: MA crossovers, MACD, ADX.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class VolumeOrderFlowStrategy(BaseStrategy):
    """
    Volume and Order Flow Strategy
    Signals based on OBV trend, MFI momentum, CMF net flow, volume surges,
    and price-OBV divergence. Completely orthogonal to EMA and TrendFollowing.
    """

    def __init__(self, config: dict):
        super().__init__(config, "VolumeFlow")
        self.mfi_period = config.get("mfi_period", 14)
        self.mfi_overbought = config.get("mfi_overbought", 80)
        self.mfi_oversold = config.get("mfi_oversold", 20)
        self.cmf_period = config.get("cmf_period", 20)
        self.obv_slope_period = config.get("obv_slope_period", 10)
        self.volume_ma_period = config.get("volume_ma_period", 20)
        self.volume_surge_threshold = config.get("volume_surge_threshold", 1.5)
        self.min_score_threshold = config.get("min_conditions", 3)
        logger.info(
            f"[{self.name}] Initialized: MFI={self.mfi_period}, "
            f"CMF={self.cmf_period}, OBV slope={self.obv_slope_period}"
        )

    def get_warmup_period(self) -> int:
        return max(self.mfi_period, self.cmf_period, self.volume_ma_period) + 50

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"].values.astype("float64")
        high = df["high"].values.astype("float64")
        low = df["low"].values.astype("float64")
        volume = (
            df["volume"].values.astype("float64")
            if "volume" in df.columns
            else np.ones(len(df))
        )
        volume = np.where(volume <= 0, 1.0, volume)

        # --- OBV ---
        obv_raw = ta.OBV(close, volume)
        df["obv"] = pd.Series(obv_raw).ffill().fillna(0).values
        df["obv_slope"] = pd.Series(df["obv"]).diff(self.obv_slope_period).values
        df["obv_ma"] = pd.Series(
            ta.SMA(df["obv"].values, timeperiod=self.obv_slope_period)
        ).ffill().fillna(0).values
        df["obv_above_ma"] = (df["obv"] > df["obv_ma"]).astype(int)

        # --- MFI ---
        try:
            df["mfi"] = pd.Series(
                ta.MFI(high, low, close, volume, timeperiod=self.mfi_period)
            ).fillna(50).values
        except Exception:
            df["mfi"] = 50.0
        df["mfi_slope"] = pd.Series(df["mfi"]).diff(3).values

        # --- CMF ---
        try:
            hl_range = np.where(high - low == 0, 1e-10, high - low)
            clv = ((close - low) - (high - close)) / hl_range
            mfv = clv * volume
            cmf_num = pd.Series(mfv).rolling(self.cmf_period).sum()
            cmf_den = pd.Series(volume).rolling(self.cmf_period).sum().replace(0, 1e-10)
            df["cmf"] = (cmf_num / cmf_den).fillna(0).values
        except Exception:
            df["cmf"] = 0.0

        # --- Volume ---
        df["volume_ma"] = pd.Series(
            ta.SMA(volume, timeperiod=self.volume_ma_period)
        ).ffill().fillna(1.0).values
        df["volume_ratio"] = np.where(df["volume_ma"] > 0, volume / df["volume_ma"], 1.0)
        df["volume_surge"] = (df["volume_ratio"] > self.volume_surge_threshold).astype(int)

        # --- Price-OBV Divergence ---
        price_chg = pd.Series(close).diff(self.obv_slope_period)
        obv_chg = pd.Series(df["obv"]).diff(self.obv_slope_period)
        df["bullish_divergence"] = ((price_chg < 0) & (obv_chg > 0)).astype(int).values
        df["bearish_divergence"] = ((price_chg > 0) & (obv_chg < 0)).astype(int).values

        # --- ATR percentile ---
        try:
            atr_raw = ta.ATR(high, low, close, timeperiod=14)
            df["atr"] = pd.Series(atr_raw).fillna(close * 0.01).values
        except Exception:
            df["atr"] = close * 0.01
        df["atr_percentile"] = pd.Series(df["atr"]).rolling(50).rank(pct=True).fillna(0.5).values

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def _score(self, row) -> tuple:
        """Shared scoring logic."""
        bull = 0.0
        bear = 0.0

        # 1. OBV Trend (0-1.5 pts)
        if row["obv_slope"] > 0:
            bull += 1.5 if row["obv_above_ma"] == 1 else 0.75
        elif row["obv_slope"] < 0:
            bear += 1.5 if row["obv_above_ma"] == 0 else 0.75

        # 2. MFI State (0-2 pts)
        mfi = row["mfi"]
        ms = row["mfi_slope"]
        if mfi < self.mfi_oversold and ms > 0:
            bull += 2.0
        elif mfi < 45 and ms > 0:
            bull += 1.0
        elif mfi > self.mfi_overbought and ms < 0:
            bear += 2.0
        elif mfi > 55 and ms < 0:
            bear += 1.0

        # 3. CMF Net Flow (0-1.5 pts)
        cmf = row["cmf"]
        if cmf > 0.10:
            bull += 1.5
        elif cmf > 0.05:
            bull += 0.75
        elif cmf < -0.10:
            bear += 1.5
        elif cmf < -0.05:
            bear += 0.75

        # 4. Volume Surge confirms direction (0-1 pt)
        if row["volume_surge"] == 1:
            if bull > bear:
                bull += 1.0
            elif bear > bull:
                bear += 1.0

        # 5. Price-OBV Divergence — leading indicator (0-1.5 pts)
        if row["bullish_divergence"] == 1:
            bull += 1.5
        if row["bearish_divergence"] == 1:
            bear += 1.5

        # 6. Extreme volatility dampener
        if row["atr_percentile"] > 0.85:
            bull *= 0.85
            bear *= 0.85

        return bull, bear

    def generate_signal(self, df: pd.DataFrame, df_4h: pd.DataFrame = None) -> tuple:
        if len(df) < self.get_warmup_period():
            return 0, 0.0
        try:
            features_df = self.generate_features(df.tail(150))
            if features_df.empty:
                return 0, 0.0
            latest = features_df.iloc[-1]
            bull, bear = self._score(latest)

            signal = 0
            if bull >= self.min_score_threshold and bull > bear:
                signal = 1
            elif bear >= self.min_score_threshold and bear > bull:
                signal = -1

            if signal == 0:
                return 0, 0.0

            active = bull if signal == 1 else bear
            confidence = min(1.0, active / 8.5)
            if signal == 1 and latest["bullish_divergence"] == 1:
                confidence = min(1.0, confidence + 0.10)
            elif signal == -1 and latest["bearish_divergence"] == 1:
                confidence = min(1.0, confidence + 0.10)

            logger.debug(
                f"[{self.name}] bull={bull:.2f} bear={bear:.2f} "
                f"MFI={latest['mfi']:.1f} CMF={latest['cmf']:.3f} "
                f"signal={signal} conf={confidence:.2f}"
            )
            return signal, confidence
        except Exception as e:
            logger.error(f"[{self.name}] generate_signal error: {e}")
            return 0, 0.0

    def generate_labels(self, df: pd.DataFrame, df_4h: pd.DataFrame = None) -> pd.Series:
        if len(df) < self.get_warmup_period():
            return pd.Series(0, index=df.index)
        try:
            features_df = self.generate_features(df)
            labels = pd.Series(0, index=df.index)
            close = df["close"].values
            lookforward = 5
            for i in range(len(features_df) - lookforward):
                row = features_df.iloc[i]
                bull, bear = self._score(row)
                future_closes = close[i + 1: i + 1 + lookforward]
                if len(future_closes) == 0:
                    continue
                future_return = (np.mean(future_closes) - close[i]) / close[i]
                if bull >= self.min_score_threshold and bull > bear and future_return > 0.003:
                    labels.iloc[i] = 1
                elif bear >= self.min_score_threshold and bear > bull and future_return < -0.003:
                    labels.iloc[i] = -1
            return labels
        except Exception as e:
            logger.error(f"[{self.name}] generate_labels error: {e}")
            return pd.Series(0, index=df.index)
