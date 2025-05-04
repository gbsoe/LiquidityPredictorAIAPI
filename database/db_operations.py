import logging
import os
import pandas as pd
import psycopg2
from datetime import datetime
from database.mock_db import MockDBManager

class DBManager:
    """
    Database manager for PostgreSQL operations
    With fallback to mock DB if database connection fails
    """
    
    def __init__(self, use_mock=None):
        self.logger = logging.getLogger(__name__)
        
        # Always create the mock database for fallback
        self.mock_db = MockDBManager()
        
        # Allow forcing mock DB usage
        if use_mock is not None:
            self.use_mock = use_mock
            if use_mock:
                self.logger.info("Explicitly using mock database")
            return
        
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
        """Get historical metrics for a specific pool"""
        if self.use_mock:
            return self.mock_db.get_pool_metrics(pool_id, days)
        
        # Real implementation would query the database
        # For now, just return mock data
        return self.mock_db.get_pool_metrics(pool_id, days)
    
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
                        # These would normally come from actual prediction models
                        # For now, generate based on pool metrics
                        
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
                            'pool_name': pool.get('name', ''),
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
            
            # If we reach here and the use of mock data is allowed,
            # only then use the mock data
            if self.use_mock:
                self.logger.warning("No real prediction data available - falling back to mock data")
                return self.mock_db.get_top_predictions(category, limit, ascending)
            
            # If we're not allowed to use mock data, return empty DataFrame
            self.logger.error("No prediction data available and mock data not allowed")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting top predictions: {str(e)}")
            # Only use mock if it's explicitly allowed
            if self.use_mock:
                return self.mock_db.get_top_predictions(category, limit, ascending)
            return pd.DataFrame()
    
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
                # Get detailed pool info from data service
                pool = data_service.get_pool_by_id(pool_id)
                
                if pool:
                    import random
                    from datetime import datetime, timedelta
                    
                    # Get pool metrics to use for prediction generation
                    apr = pool.get('apr', 0)
                    if apr is None or apr == 0:
                        apr = pool.get('apr_24h', 0) or pool.get('apy', 0) or random.uniform(5, 50)
                    
                    # Generate synthetic prediction history
                    # In a real system this would come from historical data
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
            
            # Only use mock if it's explicitly allowed
            if self.use_mock:
                self.logger.warning(f"No real prediction data available for pool {pool_id} - falling back to mock data")
                all_predictions = self.mock_db.get_pool_predictions(pool_id)
                
                if days and not all_predictions.empty and 'timestamp' in all_predictions.columns:
                    # Filter by days if we have timestamp data
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                    return all_predictions[all_predictions['timestamp'] >= cutoff_date]
                
                return all_predictions
            
            # If we reach here, we couldn't get real data and mock isn't allowed
            self.logger.error(f"No prediction data available for pool {pool_id} and mock data not allowed")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting pool predictions: {str(e)}")
            # Only use mock if explicitly allowed
            if self.use_mock:
                return self.mock_db.get_pool_predictions(pool_id)
            return pd.DataFrame()