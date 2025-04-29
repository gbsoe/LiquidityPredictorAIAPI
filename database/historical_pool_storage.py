"""
Historical Pool Data Storage System

This module handles the storage and retrieval of time-series pool data
for analytics and prediction purposes.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, 
    Float, Boolean, DateTime, Text, JSON, ForeignKey, select, desc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Configure logging
logger = logging.getLogger('historical_storage')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# SQLAlchemy setup
Base = declarative_base()

class PoolSnapshot(Base):
    """Model representing a single snapshot of a pool at a point in time"""
    __tablename__ = 'pool_snapshots'
    
    id = Column(Integer, primary_key=True)
    pool_id = Column(String(64), index=True, nullable=False)
    dex_name = Column(String(32), index=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    tvl = Column(Float)
    apy = Column(Float)
    volume_24h = Column(Float)
    fee = Column(Float)
    token1 = Column(String(32), index=True)
    token2 = Column(String(32), index=True)
    token1_price = Column(Float)
    token2_price = Column(Float)
    price_ratio = Column(Float)
    extra_data = Column(JSON)
    data_source = Column(String(32))
    
    def __repr__(self):
        return f"<PoolSnapshot(pool_id='{self.pool_id}', timestamp='{self.timestamp}')>"


class HistoricalPoolStorage:
    """
    Manages storage and retrieval of historical pool data with time-series support
    for analytics and prediction.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the storage system.
        
        Args:
            db_url: Database connection URL (uses SQLite by default if None)
        """
        # Default to SQLite if no URL provided
        if db_url is None:
            db_path = os.path.join(os.getcwd(), 'historical_pools.db')
            db_url = f"sqlite:///{db_path}"
        
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logger.info(f"Initialized historical pool storage with DB: {db_url}")
    
    def store_pool_snapshot(self, pool_data: Dict[str, Any]) -> bool:
        """
        Store a single pool snapshot.
        
        Args:
            pool_data: Pool data including metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract required fields, with defaults for missing data
            pool_id = pool_data.get('poolId')
            if not pool_id:
                logger.warning(f"Pool data missing required field 'poolId'")
                return False
            
            timestamp = datetime.now()
            if 'timestamp' in pool_data:
                # If timestamp was provided, use it instead
                timestamp_data = pool_data['timestamp']
                if isinstance(timestamp_data, str):
                    timestamp = datetime.fromisoformat(timestamp_data)
                elif isinstance(timestamp_data, datetime):
                    timestamp = timestamp_data
            
            # Extract metrics data
            metrics = pool_data.get('metrics', {})
            
            # Parse token data from name if tokens array is empty
            token1 = None
            token2 = None
            name = pool_data.get('name', '')
            if name and '-' in name:
                parts = name.split('-')
                if len(parts) >= 2:
                    token1 = parts[0].strip()
                    token2_parts = parts[1].split(' ')
                    token2 = token2_parts[0].strip()
            
            # Create snapshot record
            snapshot = PoolSnapshot(
                pool_id=pool_id,
                dex_name=pool_data.get('source'),
                timestamp=timestamp,
                tvl=metrics.get('tvl'),
                apy=metrics.get('apy24h') or metrics.get('apy') or metrics.get('apr24h') or metrics.get('apr'),
                volume_24h=metrics.get('volumeUsd') or metrics.get('volume24h') or metrics.get('volume'),
                fee=metrics.get('fee'),
                token1=token1,
                token2=token2,
                token1_price=metrics.get('token1Price'),
                token2_price=metrics.get('token2Price'),
                price_ratio=metrics.get('priceRatio'),
                extra_data=pool_data.get('extraData') or metrics.get('extraData'),
                data_source=pool_data.get('data_source', 'API')
            )
            
            # Save to database
            session = self.Session()
            session.add(snapshot)
            session.commit()
            logger.debug(f"Stored snapshot for pool {pool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing pool snapshot: {str(e)}")
            return False
    
    def store_multiple_snapshots(self, pools: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Store multiple pool snapshots in bulk.
        
        Args:
            pools: List of pool data dictionaries
            
        Returns:
            Tuple of (success_count, total_count)
        """
        success_count = 0
        total_count = len(pools)
        
        # Process in batches for better performance
        batch_size = 100
        
        try:
            session = self.Session()
            
            for i in range(0, total_count, batch_size):
                batch = pools[i:i+batch_size]
                snapshots = []
                
                for pool_data in batch:
                    try:
                        # Extract required fields, with defaults for missing data
                        pool_id = pool_data.get('poolId')
                        if not pool_id:
                            continue
                        
                        timestamp = datetime.now()
                        if 'timestamp' in pool_data:
                            # If timestamp was provided, use it instead
                            timestamp_data = pool_data['timestamp']
                            if isinstance(timestamp_data, str):
                                timestamp = datetime.fromisoformat(timestamp_data)
                            elif isinstance(timestamp_data, datetime):
                                timestamp = timestamp_data
                        
                        # Extract metrics data
                        metrics = pool_data.get('metrics', {})
                        
                        # Parse token data from name if tokens array is empty
                        token1 = None
                        token2 = None
                        name = pool_data.get('name', '')
                        if name and '-' in name:
                            parts = name.split('-')
                            if len(parts) >= 2:
                                token1 = parts[0].strip()
                                token2_parts = parts[1].split(' ')
                                token2 = token2_parts[0].strip()
                        
                        # Create snapshot record
                        snapshot = PoolSnapshot(
                            pool_id=pool_id,
                            dex_name=pool_data.get('source'),
                            timestamp=timestamp,
                            tvl=metrics.get('tvl'),
                            apy=metrics.get('apy24h') or metrics.get('apy') or metrics.get('apr24h') or metrics.get('apr'),
                            volume_24h=metrics.get('volumeUsd') or metrics.get('volume24h') or metrics.get('volume'),
                            fee=metrics.get('fee'),
                            token1=token1,
                            token2=token2,
                            token1_price=metrics.get('token1Price'),
                            token2_price=metrics.get('token2Price'),
                            price_ratio=metrics.get('priceRatio'),
                            extra_data=pool_data.get('extraData') or metrics.get('extraData'),
                            data_source=pool_data.get('data_source', 'API')
                        )
                        
                        snapshots.append(snapshot)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing pool: {str(e)}")
                
                # Bulk insert snapshots
                if snapshots:
                    session.bulk_save_objects(snapshots)
                    session.commit()
                    logger.info(f"Stored batch of {len(snapshots)} snapshots")
            
            return success_count, total_count
            
        except Exception as e:
            logger.error(f"Error in bulk snapshot storage: {str(e)}")
            session.rollback()
            return success_count, total_count
    
    def get_pool_history(self, pool_id: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical data for a specific pool.
        
        Args:
            pool_id: The pool ID
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with time-series data
        """
        try:
            session = self.Session()
            
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Query snapshots
            snapshots = session.query(PoolSnapshot).filter(
                PoolSnapshot.pool_id == pool_id,
                PoolSnapshot.timestamp >= start_date
            ).order_by(PoolSnapshot.timestamp).all()
            
            # Convert to DataFrame
            if not snapshots:
                return pd.DataFrame()
            
            # Extract data from snapshots
            data = []
            for snap in snapshots:
                row = {
                    'timestamp': snap.timestamp,
                    'tvl': snap.tvl,
                    'apy': snap.apy,
                    'volume_24h': snap.volume_24h,
                    'fee': snap.fee,
                    'token1': snap.token1,
                    'token2': snap.token2,
                    'token1_price': snap.token1_price,
                    'token2_price': snap.token2_price,
                    'price_ratio': snap.price_ratio,
                    'dex': snap.dex_name
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving pool history: {str(e)}")
            return pd.DataFrame()
    
    def get_all_pools(self) -> List[str]:
        """
        Get a list of all unique pool IDs in the database.
        
        Returns:
            List of pool IDs
        """
        try:
            session = self.Session()
            
            # Query unique pool IDs
            result = session.query(PoolSnapshot.pool_id).distinct().all()
            
            # Extract IDs from result
            pool_ids = [r[0] for r in result]
            
            return pool_ids
            
        except Exception as e:
            logger.error(f"Error retrieving all pools: {str(e)}")
            return []
    
    def get_pools_by_token(self, token: str) -> List[str]:
        """
        Get pools containing a specific token.
        
        Args:
            token: Token symbol
            
        Returns:
            List of pool IDs
        """
        try:
            session = self.Session()
            
            # Query pools with the token
            result = session.query(PoolSnapshot.pool_id).filter(
                sa.or_(
                    PoolSnapshot.token1 == token,
                    PoolSnapshot.token2 == token
                )
            ).distinct().all()
            
            # Extract IDs from result
            pool_ids = [r[0] for r in result]
            
            return pool_ids
            
        except Exception as e:
            logger.error(f"Error retrieving pools by token: {str(e)}")
            return []
    
    def get_latest_snapshot(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent snapshot for a specific pool.
        
        Args:
            pool_id: The pool ID
            
        Returns:
            Dict with snapshot data or None if not found
        """
        try:
            session = self.Session()
            
            # Query the latest snapshot
            snapshot = session.query(PoolSnapshot).filter(
                PoolSnapshot.pool_id == pool_id
            ).order_by(desc(PoolSnapshot.timestamp)).first()
            
            if not snapshot:
                return None
            
            # Convert to dict
            result = {
                'pool_id': snapshot.pool_id,
                'dex': snapshot.dex_name,
                'timestamp': snapshot.timestamp,
                'tvl': snapshot.tvl,
                'apy': snapshot.apy,
                'volume_24h': snapshot.volume_24h,
                'fee': snapshot.fee,
                'token1': snapshot.token1,
                'token2': snapshot.token2,
                'token1_price': snapshot.token1_price,
                'token2_price': snapshot.token2_price,
                'price_ratio': snapshot.price_ratio,
                'extra_data': snapshot.extra_data,
                'data_source': snapshot.data_source
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving latest snapshot: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.
        
        Returns:
            Dict with statistics
        """
        try:
            session = self.Session()
            
            # Count total snapshots
            total_snapshots = session.query(sa.func.count(PoolSnapshot.id)).scalar() or 0
            
            # Count unique pools
            unique_pools = session.query(sa.func.count(sa.distinct(PoolSnapshot.pool_id))).scalar() or 0
            
            # Count unique tokens
            unique_tokens_q1 = session.query(sa.func.count(sa.distinct(PoolSnapshot.token1))).scalar() or 0
            unique_tokens_q2 = session.query(sa.func.count(sa.distinct(PoolSnapshot.token2))).scalar() or 0
            # This is an approximation since we're not deduplicating across token1 and token2
            unique_tokens = unique_tokens_q1 + unique_tokens_q2
            
            # Get earliest and latest timestamps
            earliest = session.query(sa.func.min(PoolSnapshot.timestamp)).scalar()
            latest = session.query(sa.func.max(PoolSnapshot.timestamp)).scalar()
            
            # Get timespan in days
            timespan_days = 0
            if earliest and latest:
                delta = latest - earliest
                timespan_days = delta.days + delta.seconds / 86400
            
            # Get DEX distribution
            dex_counts = {}
            dex_results = session.query(
                PoolSnapshot.dex_name,
                sa.func.count(PoolSnapshot.id)
            ).group_by(PoolSnapshot.dex_name).all()
            
            for dex, count in dex_results:
                if dex:
                    dex_counts[dex] = count
            
            # Compile stats
            stats = {
                'total_snapshots': total_snapshots,
                'unique_pools': unique_pools,
                'unique_tokens_approx': unique_tokens,
                'earliest_timestamp': earliest.isoformat() if earliest else None,
                'latest_timestamp': latest.isoformat() if latest else None,
                'timespan_days': timespan_days,
                'dex_distribution': dex_counts
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving stats: {str(e)}")
            return {
                'error': str(e),
                'total_snapshots': 0,
                'unique_pools': 0
            }
    
    def prune_old_data(self, days_to_keep: int = 30) -> int:
        """
        Remove data older than specified days to manage database size.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            session = self.Session()
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old snapshots
            result = session.query(PoolSnapshot).filter(
                PoolSnapshot.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Pruned {result} snapshots older than {days_to_keep} days")
            
            return result
            
        except Exception as e:
            logger.error(f"Error pruning old data: {str(e)}")
            session.rollback()
            return 0

# Create a global instance
storage = None

def get_storage() -> HistoricalPoolStorage:
    """Get or create the global storage instance"""
    global storage
    if storage is None:
        # Use environment variable for DB URL if available
        db_url = os.getenv('DATABASE_URL')
        storage = HistoricalPoolStorage(db_url)
    return storage

if __name__ == "__main__":
    # Simple test
    storage = get_storage()
    stats = storage.get_stats()
    print(f"Storage stats: {stats}")