"""
Dynamic Thresholds Engine
Replaces static magic numbers with market-derived baselines.
Every threshold adapts to the asset's own recent behaviour.
"""
import numpy as np
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DynamicThresholds:
    """
    Converts any metric into a Z-score against its rolling distribution.
    Usage:
        is_extreme, z_score, threshold = self.thresholds.check(
            asset="BTC", metric="ema50_distance",
            value=current_distance, z_threshold=2.5,
            fallback=3.5  # Static fallback if insufficient data
        )
    """

    def __init__(self, lookback: int = 100, min_samples: int = 20,
                 cache_path: str = "data/dynamic_thresholds_cache.json"):
        self._cache = {}  # {(asset, metric): [values]}
        self.lookback = lookback
        self.min_samples = min_samples
        self._cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        """
        Reload rolling history from disk on startup.
        Keys are serialised as "asset||metric" strings since JSON
        doesn't support tuple keys. Silently starts fresh if file
        is missing or corrupt — never raises.
        """
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r") as f:
                    raw = json.load(f)
                self._cache = {
                    tuple(k.split("||", 1)): v
                    for k, v in raw.items()
                    if "||" in k
                }
                logger.debug(
                    "[DT] Loaded %d metric histories from %s",
                    len(self._cache), self._cache_path,
                )
        except Exception as e:
            logger.warning("[DT] Could not load threshold cache: %s — starting fresh", e)
            self._cache = {}

    def save_cache(self) -> None:
        """
        Persist rolling history to disk using atomic write.
        Call from bot shutdown so the next session resumes from where
        this one left off instead of rebuilding from scratch.
        """
        try:
            Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
            serialisable = {
                f"{k[0]}||{k[1]}": v
                for k, v in self._cache.items()
            }
            tmp = Path(self._cache_path).with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(serialisable, f)
            os.replace(tmp, self._cache_path)
            logger.debug(
                "[DT] Saved %d metric histories to %s",
                len(self._cache), self._cache_path,
            )
        except Exception as e:
            logger.warning("[DT] Could not save threshold cache: %s", e)

    def check(self, asset: str, metric: str, value: float,
              z_threshold: float = 2.0, fallback: float = None) -> tuple:
        """
        Returns (is_extreme: bool, z_score: float, dynamic_threshold: float)
        Falls back to static threshold if insufficient data.
        """
        key = (asset, metric)

        # Update rolling window
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append(value)
        if len(self._cache[key]) > self.lookback:
            self._cache[key] = self._cache[key][-self.lookback:]

        values = self._cache[key]

        # Not enough data — use static fallback
        if len(values) < self.min_samples:
            if fallback is not None:
                return value > fallback, 0.0, fallback
            return False, 0.0, value

        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return False, 0.0, mean

        z_score = (value - mean) / std
        dynamic_threshold = mean + (z_threshold * std)

        return abs(z_score) > z_threshold, z_score, dynamic_threshold

    def get_percentile(self, asset: str, metric: str, value: float,
                       percentile: float = 0.90) -> tuple:
        """
        Returns (exceeds_percentile: bool, percentile_value: float)
        """
        key = (asset, metric)
        values = self._cache.get(key, [])
        if len(values) < self.min_samples:
            return False, value
        pct_val = np.percentile(values, percentile * 100)
        return value > pct_val, pct_val
