"""
Historical Data Service for SolPool Insight.

This module provides a service for storing and retrieving historical pool data.
It supports:
- Time-series storage of pool data
- Trend analysis
- Comparison of metrics over time
- Disk-based storage with SQLite
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import threading
import time

# Configure logging
logger = logging.getLogger(__name__)

# Singleton service instance
_instance = None
_lock = threading.Lock()


class HistoricalDataService:
    """
    Service for storing and retrieving historical pool data.
    
    Features:
    - Time-series storage of pool data
    - Trend analysis
    - Comparison of metrics over time
    - Disk-based storage with SQLite
    """
    
    def __init__(self, db_path: str = "data/historical_pools.db"):
        """
        Initialize the historical data service.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the database
        self.db_path = db_path
        self._init_database()
        
        # Stats tracking
        self.stats = {
            "total_snapshots": 0,
            "last_snapshot": None,
            "pools_tracked": 0,
            "metrics_tracked": 0
        }
        
        logger.info(f"Initialized historical data service with database at {db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create the pools table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS pools (
                pool_id TEXT,
                timestamp TEXT,
                data TEXT,
                PRIMARY KEY (pool_id, timestamp)
            )
            ''')
            
            # Create the metrics table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                pool_id TEXT,
                timestamp TEXT,
                metric TEXT,
                value REAL,
                PRIMARY KEY (pool_id, timestamp, metric)
            )
            ''')
            
            # Create the snapshots table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS snapshots (
                timestamp TEXT PRIMARY KEY,
                pool_count INTEGER,
                pools TEXT
            )
            ''')
            
            # Create indexes for faster querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pools_id ON pools (pool_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pools_timestamp ON pools (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_id ON metrics (pool_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (metric)')
            
            # Commit the changes
            conn.commit()
            
            # Close the connection
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def store_pool_snapshot(self, pools: List[Dict[str, Any]]) -> bool:
        """
        Store a snapshot of pool data.
        
        Args:
            pools: List of pool data to store
            
        Returns:
            True if successful, False otherwise
        """
        if not pools:
            logger.warning("No pools provided for snapshot storage")
            return False
        
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the current timestamp
            timestamp = datetime.now().isoformat()
            
            # Process each pool
            for pool in pools:
                # Ensure pool has required fields
                if 'id' not in pool and 'poolId' not in pool:
                    logger.warning("Pool missing identifier, skipping")
                    continue
                
                # Get the pool ID
                pool_id = pool.get('id', pool.get('poolId', ''))
                
                # Store the full pool data
                pool_data = json.dumps(pool)
                cursor.execute(
                    'INSERT OR REPLACE INTO pools (pool_id, timestamp, data) VALUES (?, ?, ?)',
                    (pool_id, timestamp, pool_data)
                )
                
                # Extract important metrics for time-series analysis
                metrics = pool.get('metrics', {})
                
                # Store key metrics separately for easier querying
                metric_data = [
                    (pool_id, timestamp, 'liquidity', metrics.get('tvl', 0)),
                    (pool_id, timestamp, 'volume_24h', metrics.get('volumeUsd', 0)),
                    (pool_id, timestamp, 'apr_24h', metrics.get('apy24h', metrics.get('apr24h', 0))),
                    (pool_id, timestamp, 'apr_7d', metrics.get('apy7d', metrics.get('apr7d', 0))),
                    (pool_id, timestamp, 'apr_30d', metrics.get('apy30d', metrics.get('apr30d', 0))),
                    (pool_id, timestamp, 'token1_price', pool.get('token1_price', 0)),
                    (pool_id, timestamp, 'token2_price', pool.get('token2_price', 0))
                ]
                
                # Insert metric data
                cursor.executemany(
                    'INSERT OR REPLACE INTO metrics (pool_id, timestamp, metric, value) VALUES (?, ?, ?, ?)',
                    metric_data
                )
            
            # Store summary snapshot information
            pool_ids = json.dumps([p.get('id', p.get('poolId', '')) for p in pools])
            cursor.execute(
                'INSERT OR REPLACE INTO snapshots (timestamp, pool_count, pools) VALUES (?, ?, ?)',
                (timestamp, len(pools), pool_ids)
            )
            
            # Commit the changes
            conn.commit()
            
            # Close the connection
            conn.close()
            
            # Update stats
            self.stats["total_snapshots"] += 1
            self.stats["last_snapshot"] = timestamp
            self.stats["pools_tracked"] = len(self._get_unique_pool_ids())
            self.stats["metrics_tracked"] = self._count_metrics_stored()
            
            logger.info(f"Stored snapshot of {len(pools)} pools at {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error storing pool snapshot: {str(e)}")
            return False
    
    def get_pool_history(self, pool_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific pool.
        
        Args:
            pool_id: Pool ID to get history for
            days: Number of days of history to retrieve
            
        Returns:
            List of historical pool data
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Calculate the start date
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Query for pool data
            cursor.execute(
                'SELECT timestamp, data FROM pools WHERE pool_id = ? AND timestamp >= ? ORDER BY timestamp ASC',
                (pool_id, start_date)
            )
            
            # Get the results
            rows = cursor.fetchall()
            
            # Close the connection
            conn.close()
            
            # Process the results
            history = []
            for row in rows:
                timestamp = row['timestamp']
                data = json.loads(row['data'])
                
                # Add the timestamp to the data
                data['snapshot_timestamp'] = timestamp
                
                history.append(data)
            
            return history
        except Exception as e:
            logger.error(f"Error getting pool history: {str(e)}")
            return []
    
    def get_metric_history(self, pool_id: str, metric: str, days: int = 30) -> List[Tuple[str, float]]:
        """
        Get historical data for a specific metric.
        
        Args:
            pool_id: Pool ID to get history for
            metric: Metric to get history for
            days: Number of days of history to retrieve
            
        Returns:
            List of (timestamp, value) tuples
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the start date
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Query for metric data
            cursor.execute(
                'SELECT timestamp, value FROM metrics WHERE pool_id = ? AND metric = ? AND timestamp >= ? ORDER BY timestamp ASC',
                (pool_id, metric, start_date)
            )
            
            # Get the results
            rows = cursor.fetchall()
            
            # Close the connection
            conn.close()
            
            return rows
        except Exception as e:
            logger.error(f"Error getting metric history: {str(e)}")
            return []
    
    def get_metric_trends(self, pool_ids: List[str], metric: str, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Get trend analysis for a specific metric across multiple pools.
        
        Args:
            pool_ids: List of pool IDs to analyze
            metric: Metric to analyze
            days: Number of days of history to analyze
            
        Returns:
            Dictionary mapping pool IDs to trend statistics
        """
        trends = {}
        
        for pool_id in pool_ids:
            # Get the metric history
            history = self.get_metric_history(pool_id, metric, days)
            
            if not history:
                trends[pool_id] = {
                    "trend": "unknown",
                    "change": 0,
                    "volatility": 0,
                    "data_points": 0
                }
                continue
            
            # Calculate trend statistics
            values = [value for _, value in history]
            timestamps = [ts for ts, _ in history]
            
            if len(values) < 2:
                # Not enough data points for trend analysis
                trends[pool_id] = {
                    "trend": "unknown",
                    "change": 0,
                    "volatility": 0,
                    "data_points": len(values)
                }
                continue
            
            # Calculate overall change
            start_value = values[0]
            end_value = values[-1]
            
            if start_value == 0:
                change_pct = 0
            else:
                change_pct = ((end_value - start_value) / start_value) * 100
            
            # Determine trend direction
            if change_pct > 5:
                trend = "increasing"
            elif change_pct < -5:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Calculate volatility (standard deviation of changes)
            changes = []
            for i in range(1, len(values)):
                if values[i-1] == 0:
                    continue
                
                change = ((values[i] - values[i-1]) / values[i-1]) * 100
                changes.append(change)
            
            volatility = 0
            if changes:
                volatility = sum(abs(c) for c in changes) / len(changes)
            
            # Store the trend statistics
            trends[pool_id] = {
                "trend": trend,
                "change": change_pct,
                "volatility": volatility,
                "data_points": len(values),
                "first_timestamp": timestamps[0],
                "last_timestamp": timestamps[-1],
                "first_value": values[0],
                "last_value": values[-1]
            }
        
        return trends
    
    def get_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent snapshots.
        
        Args:
            limit: Maximum number of snapshots to retrieve
            
        Returns:
            List of snapshot metadata
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query for snapshots
            cursor.execute(
                'SELECT timestamp, pool_count FROM snapshots ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            )
            
            # Get the results
            rows = cursor.fetchall()
            
            # Close the connection
            conn.close()
            
            # Process the results
            snapshots = []
            for row in rows:
                snapshots.append(dict(row))
            
            return snapshots
        except Exception as e:
            logger.error(f"Error getting snapshots: {str(e)}")
            return []
    
    def purge_old_data(self, days_to_keep: int = 90) -> bool:
        """
        Purge old data from the database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Delete old data from pools table
            cursor.execute(
                'DELETE FROM pools WHERE timestamp < ?',
                (cutoff_date,)
            )
            pools_deleted = cursor.rowcount
            
            # Delete old data from metrics table
            cursor.execute(
                'DELETE FROM metrics WHERE timestamp < ?',
                (cutoff_date,)
            )
            metrics_deleted = cursor.rowcount
            
            # Delete old data from snapshots table
            cursor.execute(
                'DELETE FROM snapshots WHERE timestamp < ?',
                (cutoff_date,)
            )
            snapshots_deleted = cursor.rowcount
            
            # Commit the changes
            conn.commit()
            
            # Vacuum the database to reclaim space
            cursor.execute('VACUUM')
            
            # Close the connection
            conn.close()
            
            logger.info(f"Purged old data: {pools_deleted} pools, {metrics_deleted} metrics, {snapshots_deleted} snapshots")
            return True
        except Exception as e:
            logger.error(f"Error purging old data: {str(e)}")
            return False
    
    def _get_unique_pool_ids(self) -> List[str]:
        """Get a list of unique pool IDs in the database."""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for unique pool IDs
            cursor.execute('SELECT DISTINCT pool_id FROM pools')
            
            # Get the results
            rows = cursor.fetchall()
            
            # Close the connection
            conn.close()
            
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Error getting unique pool IDs: {str(e)}")
            return []
    
    def _count_metrics_stored(self) -> int:
        """Count the total number of metrics stored."""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for metric count
            cursor.execute('SELECT COUNT(*) FROM metrics')
            
            # Get the result
            count = cursor.fetchone()[0]
            
            # Close the connection
            conn.close()
            
            return count
        except Exception as e:
            logger.error(f"Error counting metrics: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        # Update pools tracked count
        self.stats["pools_tracked"] = len(self._get_unique_pool_ids())
        self.stats["metrics_tracked"] = self._count_metrics_stored()
        
        return self.stats


def get_historical_service(db_path: str = "data/historical_pools.db") -> HistoricalDataService:
    """
    Get the singleton historical data service instance.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        HistoricalDataService instance
    """
    global _instance, _lock
    
    with _lock:
        if _instance is None:
            _instance = HistoricalDataService(db_path)
    
    return _instance