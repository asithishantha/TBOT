# src/data/historical_updater.py

import os
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HistoricalDataUpdater:
    """
    Automatically updates CSV files with latest candle data across multiple timeframes.
    Supports 1H (Sniper), 4H (Analyst), and 1D (Governor) data.
    Prevents duplicates and maintains data continuity.
    """

    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        # ✅ HARMONIZED: Use data/raw/ directory for all historical data
        self.historical_dir = Path("data/raw")
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        # Define timeframe configurations
        self.timeframes = {
            '1h': {
                'binance_interval': '1h',
                'mt5_timeframe': 'H1',
                'suffix': '1h',
                'lookback_days': 90,  # 3 months for 1H
                'min_candles': 100,
            },
            '4h': {
                'binance_interval': '4h',
                'mt5_timeframe': 'H4',
                'suffix': '4h',
                'lookback_days': 365,  # 1 year for 4H
                'min_candles': 250,
            },
            '1d': {
                'binance_interval': '1d',
                'mt5_timeframe': 'D1',
                'suffix': '1d',
                'lookback_days': 730,  # 2 years for 1D (Governor needs 200+ EMA)
                'min_candles': 250,
            },
        }
        
        logger.info("[MTF UPDATER] Initialized for 1H + 4H + 1D timeframes")

    def _get_filename(self, asset_name: str, timeframe: str) -> str:
        """
        Generate filename for asset and timeframe
        
        Args:
            asset_name: "BTC", "GOLD", etc.
            timeframe: "1h", "4h", "1d"
        
        Returns:
            CSV filename
        """
        filename_map = {
            "BTC": f"BTCUSDT_{timeframe}.csv",
            "GOLD": f"XAUUSDm_{timeframe}.csv",
            "XAU": f"XAUUSDm_{timeframe}.csv",
        }
        return filename_map.get(asset_name.upper(), f"{asset_name}_{timeframe}.csv")

    def update_asset_timeframe(
        self, 
        asset_name: str, 
        timeframe: str,
        force_full_refresh: bool = False
    ) -> bool:
        """
        Update historical data for a specific asset and timeframe.
        
        Args:
            asset_name: Asset name (e.g., "BTC", "GOLD")
            timeframe: Timeframe key ("1h", "4h", "1d")
            force_full_refresh: If True, re-download all data
        
        Returns:
            True if successful
        """
        try:
            # ✅ Check Market Hours
            from src.utils.market_hours import MarketHours
            # BTC/Crypto is 24/7, ignore the institutional weekend gate for data updates
            if "BTC" in asset_name.upper():
                market_open = True
            else:
                market_open = MarketHours.should_trade(asset_name)

            if not market_open:
                logger.debug(f"[UPDATE] Skipping {asset_name} {timeframe.upper()} - Market is CLOSED")
                return True

            # Get timeframe config
            tf_config = self.timeframes.get(timeframe)
            if not tf_config:
                logger.error(f"[UPDATE] Unknown timeframe: {timeframe}")
                return False
            
            # Get asset config
            asset_cfg = self.config["assets"].get(asset_name)
            if not asset_cfg:
                logger.error(f"[UPDATE] Unknown asset: {asset_name}")
                return False
            
            symbol = asset_cfg["symbol"]
            exchange = asset_cfg.get("exchange", "binance")
            
            # Generate filename
            csv_filename = self._get_filename(asset_name, timeframe)
            csv_path = self.historical_dir / csv_filename
            
            logger.info(f"\n{'='*70}")
            logger.info(f"[UPDATE] {asset_name} {timeframe.upper()} Data")
            logger.info(f"{'='*70}")
            logger.info(f"File:     {csv_filename}")
            logger.info(f"Exchange: {exchange}")
            logger.info(f"Symbol:   {symbol}")
            
            # Determine date range
            end_time = datetime.now(timezone.utc)
            existing_df = None
            start_time = None
            
            if csv_path.exists() and not force_full_refresh:
                # Load existing data
                try:
                    existing_df = pd.read_csv(csv_path)
                    
                    # Find timestamp column
                    timestamp_col = self._find_timestamp_column(existing_df)
                    
                    if timestamp_col:
                        existing_df[timestamp_col] = pd.to_datetime(
                            existing_df[timestamp_col], utc=True, errors='coerce'
                        )
                        
                        last_date = existing_df[timestamp_col].max()
                        
                        if pd.notna(last_date):
                            logger.info(f"Last Date: {last_date}")
                            
                            # Start from 1 period after last date
                            if timeframe == '1h':
                                start_time = last_date + timedelta(hours=1)
                            elif timeframe == '4h':
                                start_time = last_date + timedelta(hours=4)
                            else:  # 1d
                                start_time = last_date + timedelta(days=1)
                            
                            # ✅ FIX: Ensure start_time is not in the future
                            if start_time > end_time:
                                logger.info(f"[UPDATE] Last date {last_date} is current. No update needed.")
                                return True
                        else:
                            logger.warning(f"[UPDATE] No valid dates in existing file")
                            existing_df = None
                    else:
                        logger.warning(f"[UPDATE] No timestamp column found")
                        existing_df = None
                        
                except Exception as e:
                    logger.error(f"[UPDATE] Error reading existing file: {e}")
                    existing_df = None
            
            # If no existing data or forced refresh, use lookback
            if start_time is None:
                lookback_days = tf_config['lookback_days']
                start_time = end_time - timedelta(days=lookback_days)
                logger.info(f"Status:   {'Full refresh' if force_full_refresh else 'Creating new file'}")
                logger.info(f"Lookback: {lookback_days} days")
            else:
                logger.info(f"Status:   Incremental update")
            
            logger.info(f"Range:    {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
            
            # Fetch new data based on exchange
            if exchange == "binance":
                new_df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=tf_config['binance_interval'],
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:  # MT5
                new_df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=tf_config['mt5_timeframe'],
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            
            # Check if new data was fetched
            if new_df is None or len(new_df) == 0:
                logger.info(f"[UPDATE] No new data available")
                return True  # Not an error, just no updates needed
            
            logger.info(f"Fetched:  {len(new_df)} new candles")
            
            # Normalize datetime column to 'date'
            new_df = self._normalize_datetime(new_df)
            
            # Merge with existing data if available
            if existing_df is not None:
                # Normalize existing data
                timestamp_col = self._find_timestamp_column(existing_df)
                if timestamp_col and timestamp_col != 'date':
                    existing_df.rename(columns={timestamp_col: 'date'}, inplace=True)
                
                # Combine and deduplicate
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                
                new_rows = len(combined_df) - len(existing_df)
                logger.info(f"Added:    {new_rows} new candles")
                logger.info(f"Total:    {len(combined_df)} candles")
            else:
                combined_df = new_df
                logger.info(f"Created:  {len(combined_df)} candles")
            
            # Validate minimum candles
            min_candles = tf_config['min_candles']
            if len(combined_df) < min_candles:
                logger.warning(
                    f"[UPDATE] Only {len(combined_df)} candles (need {min_candles}+ for {timeframe})"
                )
            
            # Save to CSV
            columns_to_save = ['date', 'open', 'high', 'low', 'close', 'volume']
            columns_to_save = [col for col in columns_to_save if col in combined_df.columns]
            
            combined_df[columns_to_save].to_csv(csv_path, index=False)
            
            logger.info(f"✅ Saved to: {csv_path}")
            logger.info(f"Date Range: {combined_df['date'].min()} → {combined_df['date'].max()}")
            logger.info(f"Latest Price: ${combined_df['close'].iloc[-1]:,.2f}")
            logger.info(f"{'='*70}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"[UPDATE] Failed to update {asset_name} {timeframe}: {e}", exc_info=True)
            return False

    def update_asset_history(self, asset_name: str, force_full_refresh: bool = False) -> Dict[str, bool]:
        """
        Update all timeframes for a specific asset.
        
        Args:
            asset_name: Asset name (e.g., "BTC", "GOLD")
            force_full_refresh: If True, re-download all data
        
        Returns:
            Dict mapping timeframe to success status
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"# UPDATING {asset_name} - ALL TIMEFRAMES")
        logger.info(f"{'#'*70}")
        
        results = {}
        
        for timeframe in ['1h', '4h', '1d']:
            success = self.update_asset_timeframe(
                asset_name=asset_name,
                timeframe=timeframe,
                force_full_refresh=force_full_refresh
            )
            results[timeframe] = success
        
        # Summary
        successful = sum(1 for s in results.values() if s)
        logger.info(f"\n[SUMMARY] {asset_name}: {successful}/3 timeframes updated")
        
        return results

    def update_all_enabled_assets(self, force_full_refresh: bool = False) -> Dict[str, Dict[str, bool]]:
        """
        Update historical data for all enabled assets across all timeframes.
        
        Args:
            force_full_refresh: If True, re-download all data
        
        Returns:
            Nested dict: {asset: {timeframe: success}}
        """
        enabled = [
            name for name, cfg in self.config["assets"].items()
            if cfg.get("enabled", False)
        ]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"MULTI-TIMEFRAME HISTORICAL DATA UPDATE")
        logger.info(f"{'='*70}")
        logger.info(f"Assets:     {', '.join(enabled)}")
        logger.info(f"Timeframes: 1H, 4H, 1D")
        logger.info(f"Mode:       {'Full Refresh' if force_full_refresh else 'Incremental'}")
        logger.info(f"{'='*70}\n")
        
        all_results = {}
        
        for asset_name in enabled:
            results = self.update_asset_history(
                asset_name=asset_name,
                force_full_refresh=force_full_refresh
            )
            all_results[asset_name] = results
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info(f"UPDATE COMPLETE - FINAL SUMMARY")
        logger.info(f"{'='*70}")
        
        for asset_name, results in all_results.items():
            successful = sum(1 for s in results.values() if s)
            status = "✅" if successful == 3 else "⚠️"
            logger.info(f"{status} {asset_name}: {successful}/3 timeframes")
            
            for tf, success in results.items():
                status_icon = "✅" if success else "❌"
                logger.info(f"   {status_icon} {tf.upper()}")
        
        logger.info(f"{'='*70}\n")
        
        return all_results

    def verify_data_integrity(self, asset_name: str) -> Dict[str, Dict]:
        """
        Check for gaps and issues in all timeframes for an asset.
        
        Args:
            asset_name: Asset name
        
        Returns:
            Dict mapping timeframe to integrity report
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA INTEGRITY CHECK - {asset_name}")
        logger.info(f"{'='*70}\n")
        
        reports = {}
        
        for timeframe in ['1h', '4h', '1d']:
            csv_filename = self._get_filename(asset_name, timeframe)
            csv_path = self.historical_dir / csv_filename
            
            if not csv_path.exists():
                logger.warning(f"[{timeframe.upper()}] File does not exist: {csv_filename}")
                reports[timeframe] = {'exists': False}
                continue
            
            try:
                df = pd.read_csv(csv_path)
                timestamp_col = self._find_timestamp_column(df)
                
                if not timestamp_col:
                    logger.error(f"[{timeframe.upper()}] No timestamp column")
                    reports[timeframe] = {'exists': True, 'error': 'no_timestamp'}
                    continue
                
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values(timestamp_col)
                
                # Check for gaps
                df['time_diff'] = df[timestamp_col].diff()
                
                # Define expected intervals
                expected_intervals = {
                    '1h': pd.Timedelta(hours=1),
                    '4h': pd.Timedelta(hours=4),
                    '1d': pd.Timedelta(days=1),
                }
                
                expected_interval = expected_intervals[timeframe]
                tolerance = expected_interval * 2  # 2x tolerance
                
                gaps = df[df['time_diff'] > tolerance]
                
                # Generate report
                report = {
                    'exists': True,
                    'total_candles': len(df),
                    'date_range': {
                        'start': str(df[timestamp_col].min()),
                        'end': str(df[timestamp_col].max()),
                    },
                    'latest_candle': str(df[timestamp_col].max()),
                    'gaps_found': len(gaps),
                }
                
                reports[timeframe] = report
                
                # Log results
                logger.info(f"[{timeframe.upper()}] {csv_filename}")
                logger.info(f"  Total Candles: {len(df)}")
                logger.info(f"  Date Range:    {df[timestamp_col].min()} → {df[timestamp_col].max()}")
                logger.info(f"  Latest Candle: {df[timestamp_col].max()}")
                
                if len(gaps) > 0:
                    logger.warning(f"  ⚠️  Gaps Found:   {len(gaps)}")
                    for idx, row in gaps.head(5).iterrows():
                        logger.warning(f"     Gap at {row[timestamp_col]}: {row['time_diff']}")
                else:
                    logger.info(f"  ✅ No Gaps")
                
                logger.info("")
                
            except Exception as e:
                logger.error(f"[{timeframe.upper()}] Verification failed: {e}")
                reports[timeframe] = {'exists': True, 'error': str(e)}
        
        return reports

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in dataframe"""
        for col in ['date', 'timestamp', 'time', 'datetime']:
            if col in df.columns:
                return col
        return None

    def _normalize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dataframe to have 'date' column with datetime
        
        Args:
            df: Input dataframe
        
        Returns:
            Normalized dataframe with 'date' column
        """
        # Handle DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            
        # Find and rename timestamp column
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
            
        else:
            raise ValueError(
                f"No time information found in data: "
                f"index={type(df.index)}, cols={df.columns.tolist()}"
            )
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        return df

    def get_file_info(self, asset_name: str, timeframe: str) -> Optional[Dict]:
        """
        Get information about a specific data file.
        
        Args:
            asset_name: Asset name
            timeframe: Timeframe ("1h", "4h", "1d")
        
        Returns:
            Dict with file information or None if file doesn't exist
        """
        csv_filename = self._get_filename(asset_name, timeframe)
        csv_path = self.historical_dir / csv_filename
        
        if not csv_path.exists():
            return None
        
        try:
            df = pd.read_csv(csv_path)
            timestamp_col = self._find_timestamp_column(df)
            
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                
                return {
                    'filename': csv_filename,
                    'path': str(csv_path),
                    'size_kb': csv_path.stat().st_size / 1024,
                    'total_candles': len(df),
                    'date_start': str(df[timestamp_col].min()),
                    'date_end': str(df[timestamp_col].max()),
                    'latest_price': float(df['close'].iloc[-1]),
                }
            
        except Exception as e:
            logger.error(f"Error reading file info: {e}")
        
        return None