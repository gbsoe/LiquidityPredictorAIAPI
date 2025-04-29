"""
Database handler for SolPool Insight application.
Manages interactions with the database (PostgreSQL or SQLite fallback).
"""

import os
import json
import sys
import sqlite3
from datetime import datetime
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get the database URL from environment variables
# Load from environment directly (more reliable than os.environ.get)
DATABASE_URL = os.getenv('DATABASE_URL')

# SQLite fallback path
SQLITE_DB_PATH = 'liquidity_pools.db'

# Initialize SQLAlchemy base class regardless of database connection
Base = declarative_base()

# Initialize SQLAlchemy engine and metadata
engine = None
metadata = None
Session = None

try:
    # Attempt to connect to PostgreSQL - Replit automatically provides DATABASE_URL
    if DATABASE_URL:
        print(f"Attempting to connect to PostgreSQL database...")
        # Make the URL easier to print without exposing credentials
        display_url = DATABASE_URL.split('@')[0] + '@' + '@'.join(DATABASE_URL.split('@')[1:]) if '@' in DATABASE_URL else 'Invalid URL format'
        print(f"Database URL: {display_url}")
        
        # Create the SQLAlchemy engine with the URL
        engine = create_engine(DATABASE_URL)
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        print("Successfully connected to PostgreSQL database")
    else:
        # DATABASE_URL not found, try alternative ways to get it
        print("DATABASE_URL not found in direct environment - trying alternative methods")
        
        # Try to get from os.environ directly
        if 'DATABASE_URL' in os.environ:
            DATABASE_URL = os.environ['DATABASE_URL']
            print("Found DATABASE_URL in os.environ")
            if DATABASE_URL:  # Check if it's not empty
                engine = create_engine(DATABASE_URL)
                with engine.connect() as conn:
                    conn.execute(sa.text("SELECT 1"))
                print("Successfully connected to PostgreSQL database")
            else:
                raise ValueError("DATABASE_URL is empty in environment variables")
        else:
            raise ValueError("DATABASE_URL not found in environment variables")
except Exception as e:
    print(f"PostgreSQL connection error: {str(e)}")
    print(f"Falling back to SQLite database at {SQLITE_DB_PATH}")
    
    # Use SQLite as fallback
    sqlite_url = f"sqlite:///{SQLITE_DB_PATH}"
    engine = create_engine(sqlite_url)
    print(f"Using SQLite database: {sqlite_url}")

# Create metadata and session regardless of which database we're using
metadata = MetaData()
Session = sessionmaker(bind=engine)

# Define the LiquidityPool model
class LiquidityPool(Base):
    """SQLAlchemy model for liquidity pool data"""
    __tablename__ = 'liquidity_pools'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    dex = Column(String, nullable=False)
    category = Column(String, nullable=False)
    token1_symbol = Column(String, nullable=False)
    token2_symbol = Column(String, nullable=False)
    token1_address = Column(String, nullable=False)
    token2_address = Column(String, nullable=False)
    liquidity = Column(Float, nullable=False)
    volume_24h = Column(Float, nullable=False)
    apr = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    version = Column(String, nullable=False)
    apr_change_24h = Column(Float, nullable=False)
    apr_change_7d = Column(Float, nullable=False)
    tvl_change_24h = Column(Float, nullable=False)
    tvl_change_7d = Column(Float, nullable=False)
    prediction_score = Column(Float, nullable=False)
    apr_change_30d = Column(Float, nullable=False, default=0.0)
    tvl_change_30d = Column(Float, nullable=False, default=0.0)
    created_at = Column(String, nullable=True)
    updated_at = Column(String, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "dex": self.dex,
            "category": self.category,
            "token1_symbol": self.token1_symbol,
            "token2_symbol": self.token2_symbol,
            "token1_address": self.token1_address,
            "token2_address": self.token2_address,
            "liquidity": self.liquidity,
            "volume_24h": self.volume_24h,
            "apr": self.apr,
            "fee": self.fee,
            "version": self.version,
            "apr_change_24h": self.apr_change_24h,
            "apr_change_7d": self.apr_change_7d,
            "tvl_change_24h": self.tvl_change_24h,
            "tvl_change_7d": self.tvl_change_7d,
            "prediction_score": self.prediction_score,
            "apr_change_30d": self.apr_change_30d,
            "tvl_change_30d": self.tvl_change_30d,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

# Function to initialize database schema
def init_db():
    """Initialize database schema if it doesn't exist"""
    if engine:
        try:
            Base.metadata.create_all(engine)
            print("Database schema created successfully.")
            return True
        except Exception as e:
            print(f"Error creating database schema: {e}")
            print("Will attempt to use existing schema if available")
            return False
    else:
        print("Cannot initialize database: engine not available.")
        return False

# Function to store data in database
def store_pools(pool_data, replace=True):
    """
    Store pool data in the database
    
    Args:
        pool_data: List of dictionaries containing pool data
        replace: If True, replace existing entries; if False, skip duplicates
    
    Returns:
        Number of pools stored
    """
    if not engine:
        print("Database connection not available")
        return 0
    
    # Initialize database schema if needed
    schema_ready = init_db()
    
    if not schema_ready:
        # Since we couldn't create the schema, write to JSON backup
        backup_to_json(pool_data)
        print("Unable to store in database, data backed up to JSON instead")
        return 0
    
    # Create a session
    session = Session()
    
    count = 0
    try:
        for pool in pool_data:
            try:
                # Ensure all required fields are present
                required_fields = [
                    "id", "name", "dex", "category", "token1_symbol", "token2_symbol",
                    "token1_address", "token2_address", "liquidity", "volume_24h", 
                    "apr", "fee", "version", "apr_change_24h", "apr_change_7d",
                    "tvl_change_24h", "tvl_change_7d", "prediction_score"
                ]
                
                # Check for missing fields
                missing_fields = [field for field in required_fields if field not in pool]
                if missing_fields:
                    print(f"Skipping pool {pool.get('id', 'unknown')}: Missing fields: {missing_fields}")
                    continue
                
                # Check if this pool already exists
                existing = session.query(LiquidityPool).filter_by(id=pool["id"]).first()
                
                if existing and replace:
                    # Update existing entry
                    for key, value in pool.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    count += 1
                elif not existing:
                    # Create new entry
                    new_pool = LiquidityPool(**pool)
                    session.add(new_pool)
                    count += 1
            except Exception as e:
                print(f"Error processing pool {pool.get('id', 'unknown')}: {e}")
                # Continue with the next pool
                continue
                
        # Commit changes
        session.commit()
        print(f"Successfully stored {count} pools in database")
        return count
    except Exception as e:
        session.rollback()
        print(f"Error storing pools in database: {e}")
        # Still backup to JSON even if database fails
        backup_to_json(pool_data)
        return 0
    finally:
        session.close()

# Function to retrieve data from database
def get_pools(limit=50):
    """
    Retrieve pool data from database
    
    Args:
        limit: Maximum number of pools to retrieve (default 50, 0 for all)
    
    Returns:
        List of dictionaries containing pool data
    """
    # Convert None to None type explicitly to satisfy type checking
    if limit is not None and not isinstance(limit, int):
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = None
    if not engine:
        print("Database connection not available")
        # Try to fallback to JSON file
        return load_from_json()
    
    # Create a session
    session = Session()
    
    try:
        query = session.query(LiquidityPool)
        
        if limit:
            query = query.limit(limit)
            
        pools = [pool.to_dict() for pool in query.all()]
        print(f"Retrieved {len(pools)} pools from database")
        
        # If we got no pools from database, try JSON file as backup
        if not pools:
            print("No pools found in database, trying JSON file...")
            pools = load_from_json()
            if pools:
                # We found pools in JSON, try to save them back to database
                try:
                    store_pools(pools)
                except Exception as e:
                    print(f"Failed to save JSON data to database: {e}")
        
        return pools
    except Exception as e:
        print(f"Error retrieving pools from database: {e}")
        # Fallback to JSON file
        return load_from_json()
    finally:
        session.close()

# Function to convert DataFrame to pool data
def dataframe_to_pools(df):
    """
    Convert pandas DataFrame to list of pool dictionaries
    
    Args:
        df: pandas DataFrame
    
    Returns:
        List of dictionaries
    """
    return df.to_dict(orient='records')

# Function to convert pool data to DataFrame
def pools_to_dataframe(pools):
    """
    Convert list of pool dictionaries to pandas DataFrame
    
    Args:
        pools: List of dictionaries containing pool data
    
    Returns:
        pandas DataFrame
    """
    return pd.DataFrame(pools)

# Function to backup data to JSON file
def backup_to_json(pools, filename='extracted_pools.json'):
    """
    Save pool data to JSON file as backup
    
    Args:
        pools: List of dictionaries containing pool data
        filename: Output JSON filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(pools, f, indent=2)
        print(f"Successfully backed up {len(pools)} pools to {filename}")
    except Exception as e:
        print(f"Error backing up data to {filename}: {e}")
        
# Define the PoolHistoricalData model for tracking historical metrics
class PoolHistoricalData(Base):
    """SQLAlchemy model for historical pool metrics data"""
    __tablename__ = 'pool_price_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pool_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    price_ratio = Column(Float, nullable=False, default=0)
    liquidity = Column(Float, nullable=False, default=0)
    volume_24h = Column(Float, nullable=False, default=0)
    apr_24h = Column(Float, nullable=False, default=0)
    apr_7d = Column(Float, nullable=False, default=0)
    apr_30d = Column(Float, nullable=False, default=0)
    token1_price = Column(Float, nullable=False, default=0)
    token2_price = Column(Float, nullable=False, default=0)
    
    def __repr__(self):
        return f"<PoolHistoricalData(pool_id='{self.pool_id}', timestamp='{self.timestamp}')>"
        
# Define Watchlist model for organizing and categorizing pools
class Watchlist(Base):
    """SQLAlchemy model for watchlists"""
    __tablename__ = 'watchlists'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<Watchlist(name='{self.name}')>"
        
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

# Define WatchlistPool model for pools in watchlists
class WatchlistPool(Base):
    """SQLAlchemy model for pools in watchlists"""
    __tablename__ = 'watchlist_pools'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    watchlist_id = Column(Integer, nullable=False)
    pool_id = Column(String, nullable=False)
    added_at = Column(DateTime, nullable=False, default=datetime.now)
    notes = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<WatchlistPool(watchlist_id={self.watchlist_id}, pool_id='{self.pool_id}')>"
        
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "watchlist_id": self.watchlist_id,
            "pool_id": self.pool_id,
            "added_at": self.added_at,
            "notes": self.notes
        }

# Function to store historical pool data for better predictions
def store_historical_pool_data(historical_records):
    """
    Store historical pool data for analysis and prediction.
    
    This function stores time-series data for each pool, allowing the system
    to build up historical performance data for better predictive analytics.
    
    Args:
        historical_records: List of dictionaries with historical data points
        
    Returns:
        Number of records stored
    """
    if not engine:
        print("Database connection not available")
        return 0
    
    # Initialize database schema if needed
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        print(f"Error ensuring schema for historical data: {e}")
        return 0
    
    # Create a session
    session = Session()
    
    count = 0
    try:
        for record in historical_records:
            try:
                # Create a new historical data entry
                # Parse the timestamp string to a datetime object
                timestamp_str = record.get("timestamp", "")
                try:
                    # If it's already a datetime object, use it directly
                    if hasattr(timestamp_str, 'isoformat'):
                        timestamp = timestamp_str
                    else:
                        # Try to parse as ISO format
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except Exception as e:
                    print(f"Error parsing timestamp {timestamp_str}: {e}")
                    # Fallback to current time
                    timestamp = datetime.now()
                
                new_record = PoolHistoricalData(
                    pool_id=record.get("pool_id", ""),
                    timestamp=timestamp,
                    price_ratio=record.get("price_ratio", 0),
                    liquidity=record.get("liquidity", 0),
                    volume_24h=record.get("volume_24h", 0),
                    apr_24h=record.get("apr_24h", 0),
                    apr_7d=record.get("apr_7d", 0),
                    apr_30d=record.get("apr_30d", 0),
                    token1_price=record.get("token1_price", 0),
                    token2_price=record.get("token2_price", 0)
                )
                
                session.add(new_record)
                count += 1
                
            except Exception as e:
                print(f"Error processing historical record for pool {record.get('pool_id', 'unknown')}: {e}")
                continue
                
        # Commit changes
        session.commit()
        print(f"Successfully stored {count} historical pool records")
        return count
        
    except Exception as e:
        session.rollback()
        print(f"Error storing historical pool data: {e}")
        return 0
    finally:
        session.close()

# Function to load data from JSON file
def load_from_json(filename='extracted_pools.json'):
    """
    Load pool data from JSON file
    
    Args:
        filename: Input JSON filename
    
    Returns:
        List of dictionaries containing pool data
    """
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return []
            
        with open(filename, 'r') as f:
            content = f.read()
            if not content.strip():
                print(f"Empty file: {filename}")
                return []
                
            pools = json.loads(content)
            print(f"Successfully loaded {len(pools)} pools from {filename}")
            return pools
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return []

# Function to query pools
def query_pools(dex=None, category=None, min_liquidity=None, max_liquidity=None, min_apr=None, max_apr=None, 
                search_term=None, sort_by=None, limit=None):
    """
    Query pools from database with filtering
    
    Args:
        dex: Filter by DEX
        category: Filter by category
        min_liquidity: Minimum liquidity
        max_liquidity: Maximum liquidity
        min_apr: Minimum APR
        max_apr: Maximum APR
        search_term: Search in name or token symbols
        sort_by: Column to sort by
        limit: Maximum number of results
        
    Returns:
        List of dictionaries containing filtered pool data
    """
    if not engine:
        print("Database connection not available")
        # Try to filter from JSON as fallback
        all_pools = load_from_json()
        return filter_pools_in_memory(all_pools, dex, category, min_liquidity, max_liquidity, 
                                     min_apr, max_apr, search_term, sort_by, limit)
    
    # Create a session
    session = Session()
    
    try:
        query = session.query(LiquidityPool)
        
        # Apply filters
        if dex:
            query = query.filter(LiquidityPool.dex == dex)
        
        if category:
            query = query.filter(LiquidityPool.category == category)
        
        if min_liquidity is not None:
            query = query.filter(LiquidityPool.liquidity >= min_liquidity)
        
        if max_liquidity is not None:
            query = query.filter(LiquidityPool.liquidity <= max_liquidity)
        
        if min_apr is not None:
            query = query.filter(LiquidityPool.apr >= min_apr)
        
        if max_apr is not None:
            query = query.filter(LiquidityPool.apr <= max_apr)
        
        if search_term:
            search_pattern = f"%{search_term}%"
            query = query.filter(
                sa.or_(
                    LiquidityPool.name.ilike(search_pattern),
                    LiquidityPool.token1_symbol.ilike(search_pattern),
                    LiquidityPool.token2_symbol.ilike(search_pattern)
                )
            )
        
        # Apply sorting
        if sort_by:
            direction = sa.desc if sort_by.startswith('-') else sa.asc
            sort_column = sort_by.lstrip('-')
            
            if hasattr(LiquidityPool, sort_column):
                query = query.order_by(direction(getattr(LiquidityPool, sort_column)))
            else:
                print(f"Invalid sort column: {sort_column}")
        
        # Apply limit
        if limit:
            query = query.limit(limit)
            
        pools = [pool.to_dict() for pool in query.all()]
        
        # If we got no results from the database, try the JSON file with the same filters
        if not pools:
            print("No pools found in database with given filters, trying JSON file...")
            all_pools = load_from_json()
            return filter_pools_in_memory(all_pools, dex, category, min_liquidity, max_liquidity, 
                                         min_apr, max_apr, search_term, sort_by, limit)
        
        return pools
    except Exception as e:
        print(f"Error querying pools from database: {e}")
        # Try to filter from JSON as fallback
        all_pools = load_from_json()
        return filter_pools_in_memory(all_pools, dex, category, min_liquidity, max_liquidity, 
                                     min_apr, max_apr, search_term, sort_by, limit)
    finally:
        session.close()

# Watchlist management functions
def create_watchlist(name, description=""):
    """
    Create a new watchlist
    
    Args:
        name: Name of the watchlist
        description: Optional description
        
    Returns:
        Dictionary with watchlist data if successful, None otherwise
    """
    if not engine:
        print("Database connection not available")
        return None
    
    # Initialize database schema if needed
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        print(f"Error ensuring watchlist schema: {e}")
        return None
    
    # Create a session
    session = Session()
    
    try:
        # Check if watchlist already exists
        existing = session.query(Watchlist).filter_by(name=name).first()
        if existing:
            print(f"Watchlist with name '{name}' already exists")
            # Convert to dictionary before returning
            result = existing.to_dict()
            return result
        
        # Create new watchlist
        watchlist = Watchlist(name=name, description=description)
        session.add(watchlist)
        session.commit()
        
        # Convert to dictionary before returning
        result = watchlist.to_dict()
        
        print(f"Successfully created watchlist: {name}")
        return result
    except Exception as e:
        session.rollback()
        print(f"Error creating watchlist: {e}")
        return None
    finally:
        session.close()

def get_watchlists():
    """
    Get all watchlists
    
    Returns:
        List of watchlist dictionaries
    """
    if not engine:
        print("Database connection not available")
        return []
    
    # Create a session
    session = Session()
    
    try:
        watchlists = [w.to_dict() for w in session.query(Watchlist).all()]
        return watchlists
    except Exception as e:
        print(f"Error retrieving watchlists: {e}")
        return []
    finally:
        session.close()

def get_watchlist(watchlist_id):
    """
    Get a specific watchlist by ID
    
    Args:
        watchlist_id: ID of the watchlist
        
    Returns:
        Watchlist dictionary if found, None otherwise
    """
    if not engine:
        print("Database connection not available")
        return None
    
    # Create a session
    session = Session()
    
    try:
        watchlist = session.query(Watchlist).filter_by(id=watchlist_id).first()
        if watchlist:
            return watchlist.to_dict()
        return None
    except Exception as e:
        print(f"Error retrieving watchlist: {e}")
        return None
    finally:
        session.close()

def add_pool_to_watchlist(watchlist_id, pool_id, notes=""):
    """
    Add a pool to a watchlist
    
    Args:
        watchlist_id: ID of the watchlist
        pool_id: ID of the pool
        notes: Optional notes about this pool
        
    Returns:
        True if successful, False otherwise
    """
    if not engine:
        print("Database connection not available")
        return False
    
    # Create a session
    session = Session()
    
    try:
        # Check if watchlist exists
        watchlist = session.query(Watchlist).filter_by(id=watchlist_id).first()
        if not watchlist:
            print(f"Watchlist with ID {watchlist_id} not found")
            return False
        
        # Check if pool is already in watchlist
        existing = session.query(WatchlistPool).filter_by(
            watchlist_id=watchlist_id, pool_id=pool_id).first()
            
        if existing:
            print(f"Pool {pool_id} is already in watchlist {watchlist_id}")
            return True
        
        # Add pool to watchlist
        watchlist_pool = WatchlistPool(
            watchlist_id=watchlist_id, pool_id=pool_id, notes=notes)
        session.add(watchlist_pool)
        session.commit()
        
        print(f"Successfully added pool {pool_id} to watchlist {watchlist_id}")
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding pool to watchlist: {e}")
        return False
    finally:
        session.close()

def remove_pool_from_watchlist(watchlist_id, pool_id):
    """
    Remove a pool from a watchlist
    
    Args:
        watchlist_id: ID of the watchlist
        pool_id: ID of the pool
        
    Returns:
        True if successful, False otherwise
    """
    if not engine:
        print("Database connection not available")
        return False
    
    # Create a session
    session = Session()
    
    try:
        # Find the watchlist pool entry
        watchlist_pool = session.query(WatchlistPool).filter_by(
            watchlist_id=watchlist_id, pool_id=pool_id).first()
            
        if not watchlist_pool:
            print(f"Pool {pool_id} not found in watchlist {watchlist_id}")
            return False
        
        # Remove the entry
        session.delete(watchlist_pool)
        session.commit()
        
        print(f"Successfully removed pool {pool_id} from watchlist {watchlist_id}")
        return True
    except Exception as e:
        session.rollback()
        print(f"Error removing pool from watchlist: {e}")
        return False
    finally:
        session.close()

def delete_watchlist(watchlist_id):
    """
    Delete a watchlist and all its pools
    
    Args:
        watchlist_id: ID of the watchlist
        
    Returns:
        True if successful, False otherwise
    """
    if not engine:
        print("Database connection not available")
        return False
    
    # Create a session
    session = Session()
    
    try:
        # Delete all pool entries for this watchlist
        session.query(WatchlistPool).filter_by(watchlist_id=watchlist_id).delete()
        
        # Delete the watchlist
        watchlist = session.query(Watchlist).filter_by(id=watchlist_id).first()
        if not watchlist:
            print(f"Watchlist with ID {watchlist_id} not found")
            return False
            
        session.delete(watchlist)
        session.commit()
        
        print(f"Successfully deleted watchlist {watchlist_id}")
        return True
    except Exception as e:
        session.rollback()
        print(f"Error deleting watchlist: {e}")
        return False
    finally:
        session.close()

def get_pools_in_watchlist(watchlist_id):
    """
    Get all pools in a watchlist
    
    Args:
        watchlist_id: ID of the watchlist
        
    Returns:
        List of pool IDs in the watchlist
    """
    if not engine:
        print("Database connection not available")
        return []
    
    # Create a session
    session = Session()
    
    try:
        pool_entries = session.query(WatchlistPool).filter_by(watchlist_id=watchlist_id).all()
        return [entry.pool_id for entry in pool_entries]
    except Exception as e:
        print(f"Error retrieving pools in watchlist: {e}")
        return []
    finally:
        session.close()

def get_watchlist_details(watchlist_id):
    """
    Get detailed information about a watchlist including all its pools
    
    Args:
        watchlist_id: ID of the watchlist
        
    Returns:
        Dictionary with watchlist details and pools
    """
    if not engine:
        print("Database connection not available")
        return None
    
    # Create a session
    session = Session()
    
    try:
        # Get the watchlist
        watchlist = session.query(Watchlist).filter_by(id=watchlist_id).first()
        if not watchlist:
            print(f"Watchlist with ID {watchlist_id} not found")
            return None
            
        # Get all pool entries for this watchlist
        pool_entries = session.query(WatchlistPool).filter_by(watchlist_id=watchlist_id).all()
        pool_ids = [entry.pool_id for entry in pool_entries]
        
        # Get the pool data for these IDs
        pool_data = []
        for pool_id in pool_ids:
            pool = session.query(LiquidityPool).filter_by(id=pool_id).first()
            if pool:
                pool_data.append(pool.to_dict())
        
        # Combine the data
        return {
            "watchlist": watchlist.to_dict(),
            "pools": pool_data
        }
    except Exception as e:
        print(f"Error retrieving watchlist details: {e}")
        return None
    finally:
        session.close()

def import_watchlist_from_json(json_data):
    """
    Import a watchlist from JSON data
    
    Args:
        json_data: JSON string or dictionary with watchlist data
        
    Returns:
        ID of the created watchlist if successful, None otherwise
    """
    if isinstance(json_data, str):
        try:
            watchlist_data = json.loads(json_data)
        except Exception as e:
            print(f"Error parsing JSON data: {e}")
            return None
    else:
        watchlist_data = json_data
    
    # Validate the data
    if not isinstance(watchlist_data, dict):
        print("Invalid watchlist data format: expected dictionary")
        return None
        
    if "name" not in watchlist_data:
        print("Invalid watchlist data: missing 'name' field")
        return None
        
    if "pools" not in watchlist_data or not isinstance(watchlist_data["pools"], list):
        print("Invalid watchlist data: missing or invalid 'pools' field")
        return None
    
    # Create the watchlist
    watchlist_dict = create_watchlist(
        name=watchlist_data["name"], 
        description=watchlist_data.get("description", "")
    )
    
    if not watchlist_dict:
        return None
    
    # Get the watchlist ID
    watchlist_id = watchlist_dict["id"]
    
    # Add pools to the watchlist
    for pool_id in watchlist_data["pools"]:
        add_pool_to_watchlist(
            watchlist_id=watchlist_id, 
            pool_id=pool_id,
            notes=watchlist_data.get("notes", {}).get(pool_id, "")
        )
    
    return watchlist_id

def export_watchlist_to_json(watchlist_id):
    """
    Export a watchlist to JSON format
    
    Args:
        watchlist_id: ID of the watchlist
        
    Returns:
        JSON string with watchlist data if successful, None otherwise
    """
    details = get_watchlist_details(watchlist_id)
    if not details:
        return None
    
    # Format the data for export
    export_data = {
        "name": details["watchlist"]["name"],
        "description": details["watchlist"]["description"],
        "created_at": details["watchlist"]["created_at"].isoformat() if hasattr(details["watchlist"]["created_at"], "isoformat") else details["watchlist"]["created_at"],
        "pools": [pool["id"] for pool in details["pools"]],
        "notes": {}
    }
    
    # Add notes if available
    session = Session()
    try:
        pool_entries = session.query(WatchlistPool).filter_by(watchlist_id=watchlist_id).all()
        for entry in pool_entries:
            if entry.notes:
                export_data["notes"][entry.pool_id] = entry.notes
    except Exception as e:
        print(f"Error retrieving pool notes: {e}")
    finally:
        session.close()
    
    return json.dumps(export_data, indent=2)

def backup_watchlists_to_file(filename="watchlists.json"):
    """
    Back up all watchlists to a JSON file
    
    Args:
        filename: Output JSON filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        watchlists = get_watchlists()
        
        export_data = []
        for watchlist in watchlists:
            details = get_watchlist_details(watchlist["id"])
            if details:
                watchlist_data = {
                    "name": details["watchlist"]["name"],
                    "description": details["watchlist"]["description"],
                    "created_at": details["watchlist"]["created_at"].isoformat() if hasattr(details["watchlist"]["created_at"], "isoformat") else details["watchlist"]["created_at"],
                    "pools": [pool["id"] for pool in details["pools"]],
                    "notes": {}
                }
                
                # Add notes if available
                session = Session()
                try:
                    pool_entries = session.query(WatchlistPool).filter_by(watchlist_id=watchlist["id"]).all()
                    for entry in pool_entries:
                        if entry.notes:
                            watchlist_data["notes"][entry.pool_id] = entry.notes
                except Exception as e:
                    print(f"Error retrieving pool notes: {e}")
                finally:
                    session.close()
                    
                export_data.append(watchlist_data)
        
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Successfully backed up {len(export_data)} watchlists to {filename}")
        return True
    except Exception as e:
        print(f"Error backing up watchlists: {e}")
        return False

def restore_watchlists_from_file(filename="watchlists.json", replace=False):
    """
    Restore watchlists from a JSON file
    
    Args:
        filename: Input JSON filename
        replace: If True, replace existing watchlists; if False, skip duplicates
        
    Returns:
        Number of watchlists restored
    """
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return 0
            
        with open(filename, "r") as f:
            content = f.read()
            if not content.strip():
                print(f"Empty file: {filename}")
                return 0
                
            watchlists_data = json.loads(content)
            
        count = 0
        for watchlist_data in watchlists_data:
            # Check if this watchlist already exists
            existing = None
            session = Session()
            try:
                existing = session.query(Watchlist).filter_by(name=watchlist_data["name"]).first()
            except Exception:
                pass
            finally:
                session.close()
            
            if existing and not replace:
                print(f"Skipping existing watchlist: {watchlist_data['name']}")
                continue
                
            if existing and replace:
                # Delete the existing watchlist
                delete_watchlist(existing.to_dict()["id"])
            
            # Import the watchlist
            if import_watchlist_from_json(watchlist_data):
                count += 1
        
        print(f"Successfully restored {count} watchlists from {filename}")
        return count
    except Exception as e:
        print(f"Error restoring watchlists: {e}")
        return 0

# Function to filter pools in memory (used as fallback when database is unavailable)
def filter_pools_in_memory(pools, dex=None, category=None, min_liquidity=None, max_liquidity=None, 
                          min_apr=None, max_apr=None, search_term=None, sort_by=None, limit=None):
    """
    Filter pools in memory (Python-based filtering instead of SQL)
    
    Args:
        pools: List of pool dictionaries
        dex: Filter by DEX
        category: Filter by category
        min_liquidity: Minimum liquidity
        max_liquidity: Maximum liquidity
        min_apr: Minimum APR
        max_apr: Maximum APR
        search_term: Search in name or token symbols
        sort_by: Column to sort by
        limit: Maximum number of results
        
    Returns:
        Filtered list of dictionaries
    """
    filtered_pools = pools.copy()
    
    # Apply filters
    if dex:
        filtered_pools = [p for p in filtered_pools if p.get('dex') == dex]
    
    if category:
        filtered_pools = [p for p in filtered_pools if p.get('category') == category]
    
    if min_liquidity is not None:
        filtered_pools = [p for p in filtered_pools if p.get('liquidity', 0) >= min_liquidity]
    
    if max_liquidity is not None:
        filtered_pools = [p for p in filtered_pools if p.get('liquidity', float('inf')) <= max_liquidity]
    
    if min_apr is not None:
        filtered_pools = [p for p in filtered_pools if p.get('apr', 0) >= min_apr]
    
    if max_apr is not None:
        filtered_pools = [p for p in filtered_pools if p.get('apr', float('inf')) <= max_apr]
    
    if search_term:
        search_term = search_term.lower()
        filtered_pools = [p for p in filtered_pools if 
                         search_term in p.get('name', '').lower() or
                         search_term in p.get('token1_symbol', '').lower() or
                         search_term in p.get('token2_symbol', '').lower()]
    
    # Apply sorting
    if sort_by:
        reverse = sort_by.startswith('-')
        sort_column = sort_by.lstrip('-')
        
        if sort_column in pools[0] if pools else []:
            filtered_pools = sorted(filtered_pools, 
                                   key=lambda p: p.get(sort_column, 0) if isinstance(p.get(sort_column, 0), (int, float)) 
                                   else str(p.get(sort_column, '')),
                                   reverse=reverse)
    
    # Apply limit
    if limit:
        filtered_pools = filtered_pools[:limit]
    
    return filtered_pools