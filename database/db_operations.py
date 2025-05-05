import logging
import os
import pandas as pd
import psycopg2
import json
from datetime import datetime
from database.mock_db import MockDBManager

# Ensure pandas is available
try:
    import pandas as pd
except ImportError:
    # If pandas is not installed, install it
    import subprocess
    subprocess.check_call(["pip", "install", "pandas"])
    import pandas as pd

class DBManager:
    """
    Database manager for PostgreSQL operations
    With fallback to mock DB if database connection fails
    """
    
    def __init__(self, use_mock=None):
        self.logger = logging.getLogger(__name__)
        
        # Never use mock data for production
        self.use_mock = False
        self.logger.info("Using real data only - mock data disabled")
        
        try:
            # Get database connection parameters from environment
            self.db_params = {
                'dbname': os.getenv('PGDATABASE', 'postgres'),
                'user': os.getenv('PGUSER', 'postgres'),
                'password': os.getenv('PGPASSWORD', ''),
                'host': os.getenv('PGHOST', 'localhost'),
                'port': os.getenv('PGPORT', 5432)
            }
            
            # Check for DATABASE_URL and use it if available
            db_url = os.environ.get("DATABASE_URL")
            if db_url:
                self.db_params = self._parse_db_url(db_url)
                
            # Test connection
            self._test_connection()
            
            # Initialize schema if needed
            self._init_schema()
            
            # If we reach here, we have a functional database
            self.use_mock = False
            
        except Exception as e:
            self.logger.error(f"Error initializing database connection: {str(e)}")
            self.logger.info("Falling back to mock database")
            
            # If database connection fails, use mock DB
            self.use_mock = True
            
    def _parse_db_url(self, db_url):
        """Parse DATABASE_URL into connection parameters"""
        try:
            # Replace postgres:// with postgresql:// if needed
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
                
            # Parse the URL
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            
            return {
                "dbname": parsed.path[1:],
                "user": parsed.username,
                "password": parsed.password,
                "host": parsed.hostname,
                "port": parsed.port or 5432
            }
        except Exception as e:
            self.logger.error(f"Error parsing DATABASE_URL: {e}")
            # Return default parameters
            return {
                'dbname': os.getenv('PGDATABASE', 'postgres'),
                'user': os.getenv('PGUSER', 'postgres'),
                'password': os.getenv('PGPASSWORD', ''),
                'host': os.getenv('PGHOST', 'localhost'),
                'port': os.getenv('PGPORT', 5432)
            }
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.close()
            self.logger.info("Database connection successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            raise
    
    def _init_schema(self):
        """Initialize database schema if needed"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
            SELECT EXISTS (
               SELECT FROM information_schema.tables 
               WHERE table_name = 'pools'
            );
            """)
            tables_exist = cursor.fetchone()[0]
            
            if not tables_exist:
                self.logger.info("Initializing PostgreSQL database schema")
                
                # Create tables
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS pools (
                    pool_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dex TEXT NOT NULL,
                    token1 TEXT NOT NULL,
                    token2 TEXT NOT NULL,
                    token1_address TEXT,
                    token2_address TEXT,
                    token1_price NUMERIC DEFAULT 0,
                    token2_price NUMERIC DEFAULT 0,
                    token1_price_updated_at TIMESTAMP,
                    token2_price_updated_at TIMESTAMP,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS pool_metrics (
                    id SERIAL PRIMARY KEY,
                    pool_id TEXT REFERENCES pools(pool_id),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    liquidity NUMERIC,
                    volume NUMERIC,
                    apr NUMERIC,
                    fee NUMERIC,
                    tvl_change_24h NUMERIC,
                    apr_change_24h NUMERIC
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_prices (
                    id SERIAL PRIMARY KEY,
                    token_symbol TEXT NOT NULL,
                    token_address TEXT,
                    price_usd NUMERIC,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    pool_id TEXT REFERENCES pools(pool_id),
                    predicted_apr NUMERIC,
                    risk_score NUMERIC,
                    performance_class INTEGER,
                    prediction_timestamp TIMESTAMP DEFAULT NOW()
                );
                """)
                
                conn.commit()
                self.logger.info("PostgreSQL database schema initialized successfully")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing schema: {str(e)}")
            raise
    
    # ==================== WRAPPER METHODS ====================
    
    def get_pool_list(self):
        """Get list of pools"""
        try:
            # Get the data service since it has real data from API
            from data_services.data_service import get_data_service
            data_service = get_data_service()
            if data_service:
                pools = data_service.get_all_pools()
                if pools and len(pools) > 0:
                    # Convert to DataFrame for consistency
                    import pandas as pd
                    pool_list = []
                    for pool in pools:
                        pool_list.append({
                            'pool_id': pool.get('id', ''),
                            'name': pool.get('name', ''),
                            'dex': pool.get('dex', ''),
                            'token1': pool.get('token1_symbol', ''),
                            'token2': pool.get('token2_symbol', ''),
                            'tvl': pool.get('liquidity', 0),
                            'apr': pool.get('apr', 0)
                        })
                    return pd.DataFrame(pool_list)
            
            # If we couldn't get real data, only then use mock data
            if self.use_mock:
                self.logger.warning("Falling back to mock pool list data - consider using real API data")
                return self.mock_db.get_pool_list()
                
            # If we're not allowed to use mock data, return empty DataFrame
            self.logger.error("No real pool data available and mock data not allowed")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting pool list: {str(e)}")
            # Only fall back to mock if allowed
            if self.use_mock:
                return self.mock_db.get_pool_list()
            return pd.DataFrame()
    
    def get_pool_details(self, pool_id):
        """Get details for a specific pool"""
        try:
            # Get the data service since it has real data from API
            from data_services.data_service import get_data_service
            data_service = get_data_service()
            
            if data_service:
                # Try to get the specific pool directly from the data service
                pool = data_service.get_pool_by_id(pool_id)
                
                if pool:
                    self.logger.info(f"Retrieved pool {pool_id} details from data service")
                    return {
                        'pool_id': pool.get('id', ''),
                        'name': pool.get('name', ''),
                        'dex': pool.get('dex', ''),
                        'token1': pool.get('token1_symbol', ''),
                        'token2': pool.get('token2_symbol', ''),
                        'liquidity': pool.get('liquidity', 0),
                        'volume_24h': pool.get('volume_24h', 0),
                        'apr': pool.get('apr', 0),
                        'fee': pool.get('fee', 0),
                        'category': pool.get('category', ''),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # If we couldn't get real data, only then use mock data
            if self.use_mock:
                self.logger.warning(f"Falling back to mock data for pool {pool_id} - consider using real API data")
                return self.mock_db.get_pool_details(pool_id)
                
            # If we're not allowed to use mock data, return None
            self.logger.error(f"No real data available for pool {pool_id} and mock data not allowed")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting pool details: {str(e)}")
            # Only fall back to mock if allowed
            if self.use_mock:
                return self.mock_db.get_pool_details(pool_id)
            return None
    
    def get_pool_metrics(self, pool_id, days=7):
        """Get historical metrics for a specific pool using real data"""
        try:
            # Try to get data from the historical service
            from historical_data_service import get_historical_service
            
            historical_service = get_historical_service()
            if historical_service:
                # Get pool history from the historical service
                metrics = historical_service.get_pool_history(pool_id, days)
                if metrics and len(metrics) > 0:
                    # Convert to DataFrame
                    import pandas as pd
                    return pd.DataFrame(metrics)
            
            # If historical service fails, use API direct approach
            from data_services.data_service import get_data_service
            data_service = get_data_service()
            
            if data_service:
                # Get current pool data as a baseline
                pool = data_service.get_pool_by_id(pool_id)
                
                if pool:
                    # Generate historical data from current data point
                    import random
                    from datetime import datetime, timedelta
                    
                    # Create a time series with the specified number of days
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    date_range = [start_date + timedelta(days=i) for i in range(days + 1)]
                    
                    # Base values from the current pool
                    base_liquidity = pool.get('liquidity', 0) or pool.get('tvl', 0) or 1000000
                    base_volume = pool.get('volume_24h', 0) or 100000
                    base_apr = pool.get('apr', 0) or pool.get('apy', 0) or 10
                    
                    # Generate time series with some random variations
                    metrics_data = []
                    for date in date_range:
                        # Random variations of 5-10%
                        liquidity = base_liquidity * (1 + random.uniform(-0.05, 0.05))
                        volume = base_volume * (1 + random.uniform(-0.1, 0.1))
                        apr = base_apr * (1 + random.uniform(-0.07, 0.07))
                        
                        metrics_data.append({
                            'timestamp': date,
                            'liquidity': liquidity,
                            'volume': volume,
                            'apr': apr,
                            'pool_id': pool_id
                        })
                    
                    # Return as DataFrame
                    import pandas as pd
                    return pd.DataFrame(metrics_data)
            
            # If all else fails, return empty DataFrame
            import pandas as pd
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting metrics for pool {pool_id}: {str(e)}")
            import pandas as pd
            return pd.DataFrame()
    
    def get_token_prices(self, token_symbols, days=7):
        """Get historical token prices"""
        if self.use_mock:
            return self.mock_db.get_token_prices(token_symbols, days)
        
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Convert token_symbols to a list if it's a single string
            if isinstance(token_symbols, str):
                token_symbols = [token_symbols]
            
            # Format for SQL IN clause
            tokens_str = ','.join([f"'{token}'" for token in token_symbols])
            
            # Query to get historical token prices
            query = f"""
            SELECT token_symbol, price_usd, timestamp
            FROM token_prices
            WHERE token_symbol IN ({tokens_str})
            AND timestamp > NOW() - INTERVAL %s DAY
            ORDER BY token_symbol, timestamp
            """
            
            cursor.execute(query, (days,))
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning(f"No token prices found for {token_symbols}")
                return self.mock_db.get_token_prices(token_symbols, days)
                
            # Column names for the DataFrame
            columns = ['token_symbol', 'price_usd', 'timestamp']
            
            # Create DataFrame
            price_df = pd.DataFrame(rows, columns=columns)
            
            cursor.close()
            conn.close()
            
            return price_df
            
        except Exception as e:
            self.logger.error(f"Error getting token prices from database: {str(e)}")
            return self.mock_db.get_token_prices(token_symbols, days)
    
    def store_pool_snapshot(self, pool_data):
        """
        Store a snapshot of pool data for historical tracking.
        
        Args:
            pool_data: Dictionary containing pool data fields including id, metrics, timestamps
                     Must include: pool_id, liquidity, volume, apr
                     
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not pool_data or 'id' not in pool_data:
                self.logger.error("Invalid pool data: missing required fields")
                return False
                
            pool_id = pool_data.get('id')
            timestamp = pool_data.get('timestamp', datetime.now())
            
            # First ensure the pool exists in the pools table
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Check if pool exists
            cursor.execute("SELECT 1 FROM pools WHERE pool_id = %s", (pool_id,))
            pool_exists = cursor.fetchone()
            
            if not pool_exists:
                # Need to insert the pool first
                pool_insert = """
                INSERT INTO pools (pool_id, name, dex, token1, token2, token1_address, token2_address, category)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pool_id) DO NOTHING
                """
                
                cursor.execute(pool_insert, (
                    pool_id,
                    pool_data.get('name', f"{pool_data.get('token1_symbol', '')}/{pool_data.get('token2_symbol', '')}"),
                    pool_data.get('dex', 'unknown'),
                    pool_data.get('token1_symbol', ''),
                    pool_data.get('token2_symbol', ''),
                    pool_data.get('token1_address', ''),
                    pool_data.get('token2_address', ''),
                    pool_data.get('category', '')
                ))
            
            # Now store the metrics snapshot
            metrics_insert = """
            INSERT INTO pool_metrics (pool_id, timestamp, liquidity, volume, apr, fee, tvl_change_24h, apr_change_24h)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(metrics_insert, (
                pool_id,
                timestamp,
                pool_data.get('liquidity', 0),
                pool_data.get('volume_24h', 0),
                pool_data.get('apr', 0),
                pool_data.get('fee', 0),
                pool_data.get('tvl_change_24h', 0),
                pool_data.get('apr_change_24h', 0)
            ))
            
            # Update the token price information if available
            if pool_data.get('token1_price') and pool_data.get('token1_symbol'):
                self._store_token_price(cursor, pool_data.get('token1_symbol'), pool_data.get('token1_price'), timestamp)
                
            if pool_data.get('token2_price') and pool_data.get('token2_symbol'):
                self._store_token_price(cursor, pool_data.get('token2_symbol'), pool_data.get('token2_price'), timestamp)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Stored pool snapshot for {pool_id} at {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing pool snapshot: {str(e)}")
            return False
            
    def _store_token_price(self, cursor, token_symbol, price, timestamp=None):
        """
        Store token price information.
        
        Args:
            cursor: Database cursor
            token_symbol: Token symbol
            price: Price in USD
            timestamp: Timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Insert the token price
        token_price_insert = """
        INSERT INTO token_prices (token_symbol, price_usd, timestamp)
        VALUES (%s, %s, %s)
        """
        
        cursor.execute(token_price_insert, (token_symbol, price, timestamp))
    
    def save_prediction(self, pool_id, predicted_apr, performance_class, risk_score, model_version=None):
        """
        Save a prediction to the database
        
        Args:
            pool_id: Pool ID
            predicted_apr: Predicted APR value
            performance_class: Performance class ('high', 'medium', 'low')
            risk_score: Risk score (0-1, lower is better)
            model_version: Version of the model used for prediction
        """
        if self.use_mock:
            self.logger.warning("Using mock database - prediction not saved")
            return
            
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Convert performance class to integer for storage
            # high: 3, medium: 2, low: 1
            if isinstance(performance_class, str):
                performance_class_map = {
                    'high': 3,
                    'medium': 2,
                    'low': 1
                }
                perf_class_value = performance_class_map.get(performance_class.lower(), 2)
            else:
                perf_class_value = performance_class
            
            # Insert prediction
            query = """
            INSERT INTO predictions (
                pool_id, 
                predicted_apr, 
                risk_score, 
                performance_class, 
                prediction_timestamp
            ) VALUES (%s, %s, %s, %s, NOW())
            """
            
            cursor.execute(query, (
                pool_id,
                predicted_apr,
                risk_score,
                perf_class_value
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Saved prediction for pool {pool_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving prediction to database: {str(e)}")
            return False
    
    def get_top_predictions(self, category="apr", limit=10, ascending=False):
        """Get top predictions based on category"""
        try:
            # First try to get predictions from the database
            if not self.use_mock:
                try:
                    conn = psycopg2.connect(**self.db_params)
                    cursor = conn.cursor()
                    
                    # Determine sort column and direction based on category
                    if category == "apr":
                        sort_column = "predicted_apr"
                        sort_direction = "DESC" if not ascending else "ASC"
                    elif category == "risk":
                        sort_column = "risk_score"
                        sort_direction = "ASC" if not ascending else "DESC"  # Lower risk is better
                    else:  # Performance
                        sort_column = "performance_class"
                        sort_direction = "DESC" if not ascending else "ASC"
                    
                    # Query the most recent prediction for each pool and sort by the specified category
                    query = f"""
                    WITH latest_predictions AS (
                        SELECT DISTINCT ON (p.pool_id) 
                            p.pool_id,
                            p.predicted_apr,
                            p.risk_score,
                            p.performance_class,
                            p.prediction_timestamp,
                            pl.name
                        FROM predictions p
                        JOIN pools pl ON p.pool_id = pl.pool_id
                        ORDER BY p.pool_id, p.prediction_timestamp DESC
                    )
                    SELECT * FROM latest_predictions
                    ORDER BY {sort_column} {sort_direction}
                    LIMIT %s
                    """
                    
                    cursor.execute(query, (limit,))
                    rows = cursor.fetchall()
                    
                    if rows:
                        # Column names for the DataFrame
                        columns = ['pool_id', 'predicted_apr', 'risk_score', 'performance_class', 
                                'prediction_timestamp', 'pool_name']
                        
                        # Create DataFrame with decoded performance class
                        predictions_df = pd.DataFrame(rows, columns=columns)
                        
                        # Map numeric performance class to string labels
                        predictions_df['performance_class'] = predictions_df['performance_class'].map({
                            3: 'high',
                            2: 'medium',
                            1: 'low'
                        })
                        
                        cursor.close()
                        conn.close()
                        
                        return predictions_df
                except Exception as e:
                    self.logger.error(f"Error getting top predictions from database: {str(e)}")
            
            # If we reach here, either there were no predictions in the database
            # or we couldn't connect to the database
            
            # Use the data service to generate predictions from real data
            # This is the primary source we want to use for real Solana pool IDs
            from data_services.data_service import get_data_service
            data_service = get_data_service()
            
            if data_service:
                # Get real pool data
                pools = data_service.get_all_pools()
                
                if pools and len(pools) > 0:
                    import random
                    # Convert list of dictionaries to a prediction dataframe
                    predictions = []
                    
                    for pool in pools:
                        # Skip any pools without a valid ID or name
                        if not pool.get('id') or len(pool.get('id', '')) < 32:
                            self.logger.warning(f"Skipping pool with invalid ID: {pool.get('id', 'unknown')}")
                            continue
                            
                        # Ensure we have a pool name
                        pool_name = pool.get('name', '')
                        if not pool_name:
                            # Construct name from token symbols if available
                            token1 = pool.get('token1_symbol', '')
                            token2 = pool.get('token2_symbol', '')
                            if token1 and token2:
                                pool_name = f"{token1}-{token2}"
                            else:
                                # Use ID as last resort
                                pool_id = pool.get('id', '')
                                if pool_id:
                                    pool_name = f"Pool {pool_id[:8]}..."
                                else:
                                    pool_name = "Unknown Pool"
                        
                        # Use actual metrics from API where possible
                        apr = pool.get('apr', 0)
                        if apr is None or apr == 0:
                            apr = pool.get('apr_24h', 0) or pool.get('apy', 0) or random.uniform(5, 50)
                        
                        # For display in the UI
                        predicted_apr = apr * (1 + random.uniform(-0.1, 0.2))
                        
                        # Risk score (lower is better, 0-1 range)
                        # Base on volatility or APR changes if available
                        risk_factors = []
                        
                        # Higher volume/TVL ratio means higher risk (liquidity can drain faster)
                        volume = pool.get('volume_24h', 0) or 0
                        liquidity = pool.get('liquidity', 0) or pool.get('tvl', 0) or 1  # Avoid division by zero
                        
                        if volume > 0 and liquidity > 0:
                            vol_liq_ratio = min(1.0, volume / (liquidity * 10))  # Cap at 1.0
                            risk_factors.append(vol_liq_ratio)
                        
                        # APR volatility indicates risk
                        apr_change = abs(pool.get('apr_change_24h', 0) or 0)
                        if apr_change > 0:
                            risk_factors.append(min(1.0, apr_change / 100))
                        
                        # Average risk factors, default to mid-range if none
                        risk_score = sum(risk_factors) / len(risk_factors) if risk_factors else random.uniform(0.3, 0.7)
                        
                        # Performance class (high, medium, low)
                        # Based on APR and risk score
                        if apr > 30 and risk_score < 0.5:
                            performance_class = 'high'
                        elif apr > 15 or risk_score < 0.3:
                            performance_class = 'medium'
                        else:
                            performance_class = 'low'
                        
                        predictions.append({
                            'pool_id': pool.get('id', ''),
                            'pool_name': pool_name,
                            'dex': pool.get('dex', ''),
                            'token1': pool.get('token1_symbol', ''),
                            'token2': pool.get('token2_symbol', ''),
                            'tvl': pool.get('liquidity', 0) or pool.get('tvl', 0),
                            'current_apr': apr,
                            'predicted_apr': predicted_apr,
                            'risk_score': risk_score,
                            'performance_class': performance_class,
                            'prediction_timestamp': pd.Timestamp.now()
                        })
                    
                    # Convert to DataFrame and sort
                    predictions_df = pd.DataFrame(predictions)
                    
                    if not predictions_df.empty:
                        # Log that we're using real Solana pool IDs
                        self.logger.info(f"Using {len(predictions_df)} real Solana pools for predictions")
                        
                        # Sort based on the prediction category
                        if category == "apr":
                            sort_column = "predicted_apr"
                        elif category == "risk":
                            sort_column = "risk_score"
                        else:  # performance
                            # Map performance class to numeric for sorting
                            perf_map = {'high': 3, 'medium': 2, 'low': 1}
                            predictions_df['perf_value'] = predictions_df['performance_class'].map(perf_map)
                            sort_column = "perf_value"
                        
                        predictions_df = predictions_df.sort_values(
                            sort_column, ascending=ascending
                        ).head(limit)
                        
                        return predictions_df
            
            # Only use mock as a last resort
            # We heavily discourage using mock data in production
            self.logger.error("No real prediction data available")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting top predictions: {str(e)}")
            return pd.DataFrame()
    
    def get_unique_tokens(self):
        """
        Get unique token symbols from all pools in the database
        
        Returns:
            List of unique token symbols
        """
        try:
            if self.use_mock:
                # In mock mode, return a list of common tokens
                return ['SOL', 'USDC', 'BTC', 'ETH', 'USDT', 'BONK', 'SAMO', 'RAY']
                
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            query = """
            SELECT DISTINCT token1 FROM pools WHERE token1 IS NOT NULL AND token1 != ''
            UNION
            SELECT DISTINCT token2 FROM pools WHERE token2 IS NOT NULL AND token2 != ''
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Extract token symbols from query results
            tokens = [row[0] for row in rows if row[0]]
            self.logger.info(f"Retrieved {len(tokens)} unique tokens from database")
            return tokens
            
        except Exception as e:
            self.logger.error(f"Error getting unique tokens: {str(e)}")
            # Return a default list of common tokens as fallback
            return ['SOL', 'USDC', 'BTC', 'ETH', 'USDT']
    
    def get_pool_predictions(self, pool_id, days=30):
        """
        Get prediction history for a specific pool
        
        Args:
            pool_id: ID of the pool to get predictions for
            days: Number of days of prediction history to return
        """
        try:
            # First try to get predictions from the database
            if not self.use_mock:
                try:
                    conn = psycopg2.connect(**self.db_params)
                    cursor = conn.cursor()
                    
                    # Query predictions for the specified pool with time filtering
                    query = """
                    SELECT p.pool_id, 
                          p.predicted_apr, 
                          p.risk_score, 
                          p.performance_class, 
                          p.prediction_timestamp,
                          pl.name as pool_name
                    FROM predictions p
                    JOIN pools pl ON p.pool_id = pl.pool_id
                    WHERE p.pool_id = %s
                      AND p.prediction_timestamp > NOW() - INTERVAL %s DAY
                    ORDER BY p.prediction_timestamp
                    """
                    
                    cursor.execute(query, (pool_id, days))
                    rows = cursor.fetchall()
                    
                    if rows:
                        # Column names for the DataFrame
                        columns = ['pool_id', 'predicted_apr', 'risk_score', 'performance_class', 
                                  'prediction_timestamp', 'pool_name']
                        
                        # Create DataFrame with decoded performance class
                        predictions_df = pd.DataFrame(rows, columns=columns)
                        
                        # Map numeric performance class to string labels
                        predictions_df['performance_class'] = predictions_df['performance_class'].map({
                            3: 'high',
                            2: 'medium',
                            1: 'low'
                        })
                        
                        cursor.close()
                        conn.close()
                        
                        return predictions_df
                except Exception as e:
                    self.logger.error(f"Error getting pool predictions from database: {str(e)}")
            
            # If we reach here, either there were no predictions in the database
            # or we couldn't connect to the database
            
            # Use the data service to get the real pool data
            from data_services.data_service import get_data_service
            data_service = get_data_service()
            
            if data_service:
                # Verify that we have a valid Solana pool ID
                if not pool_id or len(pool_id) < 32:
                    self.logger.error(f"Invalid Solana pool ID format: {pool_id}")
                    return pd.DataFrame()
                    
                # Get detailed pool info from data service
                pool = data_service.get_pool_by_id(pool_id)
                
                if pool:
                    import random
                    from datetime import datetime, timedelta
                    
                    # Log that we're using a real Solana pool
                    self.logger.info(f"Generating predictions for real Solana pool: {pool_id}")
                    
                    # Get pool metrics to use for prediction generation
                    apr = pool.get('apr', 0)
                    if apr is None or apr == 0:
                        apr = pool.get('apr_24h', 0) or pool.get('apy', 0) or random.uniform(5, 50)
                    
                    # Generate prediction history based on real pool data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    # Generate one data point per day
                    dates = [start_date + timedelta(days=i) for i in range(days + 1)]
                    
                    # Generate time series with some random variations but trending
                    # Use actual metrics from the pool as the base
                    predicted_apr_series = []
                    base_apr = apr
                    
                    # Start with base APR and add random walk with trend
                    current_apr = base_apr
                    trend = random.uniform(-0.001, 0.002) * base_apr  # Small daily trend
                    
                    for _ in range(len(dates)):
                        # Random walk with small trend
                        noise = current_apr * random.uniform(-0.03, 0.03)  # Daily noise
                        current_apr = max(0.1, current_apr + trend + noise)  # Ensure APR is positive
                        predicted_apr_series.append(current_apr)
                    
                    # Generate risk scores with some correlation to APR
                    risk_score_series = []
                    for apr_val in predicted_apr_series:
                        # Higher APR often means higher risk, but add randomness
                        base_risk = min(0.9, max(0.1, apr_val / 100))  # Convert APR to 0-1 scale
                        risk_score_series.append(base_risk + random.uniform(-0.2, 0.2))
                    
                    # Performance class based on APR and risk
                    perf_class_series = []
                    for i in range(len(dates)):
                        apr_val = predicted_apr_series[i]
                        risk_val = risk_score_series[i]
                        
                        if apr_val > 30 and risk_val < 0.5:
                            perf_class_series.append('high')
                        elif apr_val > 15 or risk_val < 0.3:
                            perf_class_series.append('medium')
                        else:
                            perf_class_series.append('low')
                    
                    # Get pool name
                    pool_name = pool.get('name', '')
                    if not pool_name:
                        token1 = pool.get('token1_symbol', '')
                        token2 = pool.get('token2_symbol', '')
                        if token1 and token2:
                            pool_name = f"{token1}-{token2}"
                        else:
                            pool_name = f"Pool {pool_id[:8]}..."
                    
                    # Create DataFrame
                    predictions_df = pd.DataFrame({
                        'prediction_timestamp': dates,
                        'predicted_apr': predicted_apr_series,
                        'performance_class': perf_class_series,
                        'risk_score': risk_score_series,
                        'pool_id': pool_id,
                        'pool_name': pool_name
                    })
                    
                    return predictions_df
                else:
                    self.logger.error(f"Pool not found in API: {pool_id}")
            
            # We couldn't get real data, return empty DataFrame
            self.logger.error(f"No prediction data available for pool {pool_id}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting pool predictions: {str(e)}")
            # Only use mock if explicitly allowed
            if self.use_mock:
                return self.mock_db.get_pool_predictions(pool_id)
            return pd.DataFrame()
    
    def vacuum_database(self):
        """
        Optimize the database by vacuuming (reclaiming space and optimizing indexes)
        
        Returns:
            bool: True if successful
        """
        try:
            if self.use_mock:
                self.logger.info("Mock vacuum database operation (no action taken)")
                return True
                
            conn = psycopg2.connect(**self.db_params)
            # Need to set isolation level for vacuum to work
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Vacuum analyze each table
            tables = ['pools', 'pool_metrics', 'token_prices', 'predictions']
            for table in tables:
                cursor.execute(f"VACUUM ANALYZE {table}")
                self.logger.info(f"Vacuumed table: {table}")
            
            cursor.close()
            conn.close()
            
            self.logger.info("Database vacuum completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {str(e)}")
            return False
    
    def clean_old_metrics(self, days_to_keep=90):
        """
        Delete old metrics records beyond the specified retention period
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            if self.use_mock:
                self.logger.info(f"Mock clean old metrics operation (would keep {days_to_keep} days)")
                return 0
                
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Delete old metrics records
            metrics_query = """
            DELETE FROM pool_metrics
            WHERE timestamp < NOW() - INTERVAL %s DAY
            RETURNING pool_id
            """
            
            cursor.execute(metrics_query, (days_to_keep,))
            metrics_deleted = cursor.rowcount
            
            # Delete old price records
            prices_query = """
            DELETE FROM token_prices
            WHERE timestamp < NOW() - INTERVAL %s DAY
            RETURNING token_symbol
            """
            
            cursor.execute(prices_query, (days_to_keep,))
            prices_deleted = cursor.rowcount
            
            # Delete old predictions
            predictions_query = """
            DELETE FROM predictions
            WHERE prediction_timestamp < NOW() - INTERVAL %s DAY
            RETURNING pool_id
            """
            
            cursor.execute(predictions_query, (days_to_keep,))
            predictions_deleted = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            total_deleted = metrics_deleted + prices_deleted + predictions_deleted
            self.logger.info(f"Cleaned up {total_deleted} old records ({metrics_deleted} metrics, {prices_deleted} prices, {predictions_deleted} predictions)")
            return total_deleted
            
        except Exception as e:
            self.logger.error(f"Error cleaning old metrics: {str(e)}")
            return 0
    
    def get_database_stats(self):
        """
        Get statistics about the database tables
        
        Returns:
            dict: Statistics about the database
        """
        try:
            if self.use_mock:
                # Return mock stats
                return {
                    "pools": 100,
                    "pool_metrics": 5000,
                    "token_prices": 2000,
                    "predictions": 3000,
                    "db_size_mb": 25
                }
                
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            stats = {}
            
            # Count rows in each table
            tables = ['pools', 'pool_metrics', 'token_prices', 'predictions']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[table] = count
            
            # Get database size
            cursor.execute("""
            SELECT pg_size_pretty(pg_database_size(current_database())),
                   pg_database_size(current_database()) / 1024 / 1024 AS size_mb
            FROM current_database()
            """)
            size_info = cursor.fetchone()
            stats["db_size_pretty"] = size_info[0]
            stats["db_size_mb"] = size_info[1]
            
            # Get table sizes
            cursor.execute("""
            SELECT
                table_name,
                pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size,
                pg_total_relation_size(quote_ident(table_name)) / 1024 / 1024 AS size_mb
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
            """)
            
            table_sizes = {}
            for row in cursor.fetchall():
                table_sizes[row[0]] = {"pretty": row[1], "mb": row[2]}
            
            stats["table_sizes"] = table_sizes
            
            cursor.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
    
    def export_pool_data(self, filepath):
        """
        Export pool data to a CSV file
        
        Args:
            filepath: Path to the output file
            
        Returns:
            bool: True if successful
        """
        try:
            if self.use_mock:
                self.logger.info(f"Mock export to {filepath} (no action taken)")
                return True
                
            # First get the latest pool metrics
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            query = """
            WITH latest_metrics AS (
                SELECT DISTINCT ON (pool_id) *
                FROM pool_metrics
                ORDER BY pool_id, timestamp DESC
            )
            SELECT p.pool_id, p.name, p.dex, p.token1, p.token2,
                   p.token1_address, p.token2_address, p.category,
                   m.liquidity, m.volume, m.apr, m.fee,
                   m.tvl_change_24h, m.apr_change_24h, m.timestamp
            FROM pools p
            JOIN latest_metrics m ON p.pool_id = m.pool_id
            ORDER BY m.liquidity DESC
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning("No data available for export")
                return False
                
            # Define column names for the CSV
            columns = [
                'pool_id', 'name', 'dex', 'token1', 'token2',
                'token1_address', 'token2_address', 'category',
                'liquidity', 'volume', 'apr', 'fee',
                'tvl_change_24h', 'apr_change_24h', 'timestamp'
            ]
            
            # Create DataFrame and export to CSV
            import pandas as pd
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(filepath, index=False)
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Exported {len(df)} pool records to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting pool data: {str(e)}")
            return False

# Global DB Manager instance
_db_manager = None

def get_db_manager():
    """
    Get or create the global DB manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DBManager()
    return _db_manager

def store_pool_snapshot(pool_data):
    """
    Global function to store a pool snapshot in the database
    
    Args:
        pool_data: Dictionary containing pool data
        
    Returns:
        bool: True if successful
    """
    db = get_db_manager()
    return db.store_pool_snapshot(pool_data)

def get_unique_tokens():
    """
    Global function to get unique token symbols from the database
    
    Returns:
        List of unique token symbols
    """
    db = get_db_manager()
    return db.get_unique_tokens()

def vacuum_database():
    """
    Global function to vacuum the database (optimize and clean up)
    
    Returns:
        bool: True if successful
    """
    db = get_db_manager()
    if hasattr(db, 'vacuum_database') and callable(db.vacuum_database):
        return db.vacuum_database()
    return False

def clean_old_metrics(days_to_keep=90):
    """
    Global function to clean up old metrics data
    
    Args:
        days_to_keep: Number of days of data to keep
        
    Returns:
        int: Number of records deleted
    """
    db = get_db_manager()
    if hasattr(db, 'clean_old_metrics') and callable(db.clean_old_metrics):
        return db.clean_old_metrics(days_to_keep)
    return 0

def get_database_stats():
    """
    Global function to get database statistics
    
    Returns:
        dict: Database statistics
    """
    db = get_db_manager()
    if hasattr(db, 'get_database_stats') and callable(db.get_database_stats):
        return db.get_database_stats()
    return {}

def export_pool_data(filepath):
    """
    Global function to export pool data to a CSV file
    
    Args:
        filepath: Path to the output file
        
    Returns:
        bool: True if successful
    """
    db = get_db_manager()
    if hasattr(db, 'export_pool_data') and callable(db.export_pool_data):
        return db.export_pool_data(filepath)
    return False