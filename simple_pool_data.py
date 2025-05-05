import os
import psycopg2
import pandas as pd

def get_pool_data():
    """
    Fetch a simple version of pool data directly from the database
    Returns a DataFrame with basic pool data
    """
    try:
        # Get database URL from environment
        db_url = os.environ.get('DATABASE_URL')
        
        if not db_url:
            print("DATABASE_URL not found in environment variables")
            return pd.DataFrame()
            
        # Connect to the database
        conn = psycopg2.connect(db_url)
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Execute a simple query to get basic pool data
        cursor.execute("""
            SELECT p.pool_id, p.name, p.dex, p.token1, p.token2, 
                   COALESCE(pm.liquidity, 0) as liquidity,
                   COALESCE(pm.volume, 0) as volume,
                   COALESCE(pm.apr, 0) as apr
            FROM pools p
            LEFT JOIN (
                SELECT pool_id, 
                       MAX(timestamp) as latest_timestamp
                FROM pool_metrics
                GROUP BY pool_id
            ) latest ON p.pool_id = latest.pool_id
            LEFT JOIN pool_metrics pm ON pm.pool_id = latest.pool_id 
                AND pm.timestamp = latest.latest_timestamp
            LIMIT 20;
        """)
        
        # Fetch all results
        rows = cursor.fetchall()
        
        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        
        print(f"Successfully fetched {len(df)} pools from database")
        return df
        
    except Exception as e:
        print(f"Error fetching pool data: {str(e)}")
        return pd.DataFrame()
        
def insert_sample_pool_data():
    """
    Insert some sample pool data into the database for testing
    """
    try:
        # Get database URL from environment
        db_url = os.environ.get('DATABASE_URL')
        
        if not db_url:
            print("DATABASE_URL not found in environment variables")
            return False
            
        # Connect to the database
        conn = psycopg2.connect(db_url)
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if we already have data
        cursor.execute("SELECT COUNT(*) FROM pools;")
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"Database already has {count} pools, skipping sample data insertion")
            cursor.close()
            conn.close()
            return True
        
        # Sample pool data
        sample_pools = [
            ("RAYDIUM_SOL_USDC", "SOL-USDC", "Raydium", "SOL", "USDC", "So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            ("RAYDIUM_SOL_USDT", "SOL-USDT", "Raydium", "SOL", "USDT", "So11111111111111111111111111111111111111112", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"),
            ("ORCA_SOL_USDC", "SOL-USDC", "Orca", "SOL", "USDC", "So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            ("RAYDIUM_BTC_USDC", "BTC-USDC", "Raydium", "BTC", "USDC", "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            ("ORCA_MSOL_SOL", "MSOL-SOL", "Orca", "MSOL", "SOL", "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So", "So11111111111111111111111111111111111111112")
        ]
        
        # Insert pool data
        cursor.executemany("""
            INSERT INTO pools (pool_id, name, dex, token1, token2, token1_address, token2_address)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (pool_id) DO NOTHING;
        """, sample_pools)
        
        # Sample pool metrics
        sample_metrics = [
            ("RAYDIUM_SOL_USDC", 5000000.0, 1200000.0, 24.5, 0.25),
            ("RAYDIUM_SOL_USDT", 4500000.0, 980000.0, 22.3, 0.25),
            ("ORCA_SOL_USDC", 4800000.0, 1150000.0, 23.8, 0.3),
            ("RAYDIUM_BTC_USDC", 3200000.0, 780000.0, 18.7, 0.25),
            ("ORCA_MSOL_SOL", 2800000.0, 580000.0, 16.5, 0.3)
        ]
        
        # Insert pool metrics
        cursor.executemany("""
            INSERT INTO pool_metrics (pool_id, liquidity, volume, apr, fee)
            VALUES (%s, %s, %s, %s, %s);
        """, sample_metrics)
        
        # Commit the transaction
        conn.commit()
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        print(f"Successfully inserted {len(sample_pools)} sample pools and metrics")
        return True
        
    except Exception as e:
        print(f"Error inserting sample pool data: {str(e)}")
        return False
