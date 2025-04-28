"""
Database handler for SolPool Insight application.
Manages interactions with the database (PostgreSQL or SQLite fallback).
"""

import os
import json
import sys
import sqlite3
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table
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
    # Check if DATABASE_URL is available from check_database_status tool
    if DATABASE_URL:
        # Attempt to connect to PostgreSQL
        print(f"Attempting to connect to PostgreSQL database...")
        # Make the URL easier to print without exposing credentials
        display_url = DATABASE_URL.split('@')[0] + '@' + '@'.join(DATABASE_URL.split('@')[1:])
        print(f"Database URL: {display_url}")
        engine = create_engine(DATABASE_URL)
        # Test the connection
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        print("Successfully connected to PostgreSQL database")
    else:
        # Try one more time using a different method
        import subprocess
        try:
            # This might help in certain Replit environments
            result = subprocess.run(['echo', '$DATABASE_URL'], capture_output=True, text=True)
            potential_url = result.stdout.strip()
            if potential_url and not potential_url.startswith('$'):
                print(f"Found DATABASE_URL through subprocess")
                DATABASE_URL = potential_url
                engine = create_engine(DATABASE_URL)
                # Test the connection
                with engine.connect() as conn:
                    conn.execute(sa.text("SELECT 1"))
                print("Successfully connected to PostgreSQL database")
            else:
                raise ValueError("DATABASE_URL not found in environment variables")
        except Exception:
            raise ValueError("DATABASE_URL not found in environment variables")
except Exception as e:
    print(f"PostgreSQL connection error: {e}")
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
        limit: Maximum number of pools to retrieve (default 50, None for all)
    
    Returns:
        List of dictionaries containing pool data
    """
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