import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("filtering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_filtering")

@dataclass
class AdvancedFilter:
    """Advanced filter definition with value range and comparison logic"""
    field: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    values: Optional[List[Any]] = None
    exclude_values: Optional[List[Any]] = None
    custom_filter: Optional[Callable] = None
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply this filter to a DataFrame"""
        if data.empty:
            return data
            
        # Make a copy to avoid modifying the original
        filtered_data = data.copy()
        
        # Skip if field doesn't exist
        if self.field not in filtered_data.columns and not self.custom_filter:
            logger.warning(f"Field '{self.field}' not found in data, skipping filter")
            return filtered_data
        
        # Apply numeric range filters
        if self.min_value is not None:
            filtered_data = filtered_data[filtered_data[self.field] >= self.min_value]
            
        if self.max_value is not None:
            filtered_data = filtered_data[filtered_data[self.field] <= self.max_value]
        
        # Apply inclusion/exclusion filters
        if self.values is not None:
            filtered_data = filtered_data[filtered_data[self.field].isin(self.values)]
            
        if self.exclude_values is not None:
            filtered_data = filtered_data[~filtered_data[self.field].isin(self.exclude_values)]
        
        # Apply custom filter function if provided
        if self.custom_filter is not None:
            mask = filtered_data.apply(self.custom_filter, axis=1)
            filtered_data = filtered_data[mask]
        
        return filtered_data


class AdvancedFilteringSystem:
    """
    Advanced filtering system for liquidity pool data.
    Supports complex filtering operations including:
    
    1. Range-based filtering (min/max values)
    2. Multi-field filtering
    3. Category and token-based filtering
    4. Time-series based filtering (trends)
    5. Derived metrics filtering
    6. Custom filter functions
    """
    
    def __init__(self, pool_data: pd.DataFrame):
        """
        Initialize with pool data
        
        Args:
            pool_data: DataFrame containing pool data
        """
        self.original_data = pool_data.copy()
        self.filtered_data = pool_data.copy()
        self.applied_filters = []
        self.filter_history = []
        
    def reset_filters(self) -> None:
        """Reset all filters and restore original data"""
        self.filtered_data = self.original_data.copy()
        self.applied_filters = []
        logger.info("All filters reset")
        
    def add_filter(self, filter_obj: AdvancedFilter) -> 'AdvancedFilteringSystem':
        """
        Add a filter to the chain.
        Returns self for method chaining.
        """
        self.applied_filters.append(filter_obj)
        return self
        
    def apply_filters(self) -> pd.DataFrame:
        """Apply all added filters in sequence"""
        # Start with original data
        self.filtered_data = self.original_data.copy()
        
        # Track the impact of each filter for analysis
        filter_impacts = []
        
        # Apply each filter in sequence
        for i, filter_obj in enumerate(self.applied_filters):
            count_before = len(self.filtered_data)
            
            # Apply the filter
            self.filtered_data = filter_obj.apply(self.filtered_data)
            
            count_after = len(self.filtered_data)
            impact = {
                "filter_index": i,
                "filter_field": filter_obj.field,
                "records_before": count_before,
                "records_after": count_after,
                "records_filtered": count_before - count_after,
                "percentage_filtered": (count_before - count_after) / count_before * 100 if count_before > 0 else 0
            }
            filter_impacts.append(impact)
            
            logger.info(f"Applied filter on '{filter_obj.field}': {count_before - count_after} records filtered out")
        
        # Save filter history for analysis
        self.filter_history.append({
            "timestamp": datetime.now(),
            "filters_applied": len(self.applied_filters),
            "records_remaining": len(self.filtered_data),
            "filter_impacts": filter_impacts
        })
        
        return self.filtered_data
    
    def get_filtered_data(self) -> pd.DataFrame:
        """Get the current filtered data"""
        return self.filtered_data
        
    def get_filter_impact_analysis(self) -> pd.DataFrame:
        """
        Analyze the impact of each filter.
        Returns a DataFrame with metrics on each filter's impact.
        """
        if not self.filter_history:
            logger.warning("No filters have been applied yet")
            return pd.DataFrame()
            
        # Get the most recent filter run
        latest_run = self.filter_history[-1]
        
        # Convert impact data to DataFrame
        impact_df = pd.DataFrame(latest_run["filter_impacts"])
        
        return impact_df
    
    # Predefined filter factory methods for common use cases
    
    @staticmethod
    def liquidity_filter(min_value: Optional[float] = None, 
                         max_value: Optional[float] = None) -> AdvancedFilter:
        """Create a filter for liquidity range"""
        return AdvancedFilter(
            field="liquidity",
            min_value=min_value,
            max_value=max_value
        )
    
    @staticmethod
    def apr_filter(min_value: Optional[float] = None, 
                  max_value: Optional[float] = None) -> AdvancedFilter:
        """Create a filter for APR range"""
        return AdvancedFilter(
            field="apr",
            min_value=min_value,
            max_value=max_value
        )
    
    @staticmethod
    def volume_filter(min_value: Optional[float] = None, 
                     max_value: Optional[float] = None) -> AdvancedFilter:
        """Create a filter for volume range"""
        return AdvancedFilter(
            field="volume_24h",
            min_value=min_value,
            max_value=max_value
        )
    
    @staticmethod
    def dex_filter(dexes: List[str]) -> AdvancedFilter:
        """Create a filter for specific DEXes"""
        return AdvancedFilter(
            field="dex",
            values=dexes
        )
    
    @staticmethod
    def category_filter(categories: List[str]) -> AdvancedFilter:
        """Create a filter for specific categories"""
        return AdvancedFilter(
            field="category",
            values=categories
        )
    
    @staticmethod
    def token_filter(tokens: List[str]) -> AdvancedFilter:
        """
        Create a filter for pools containing specific tokens
        This requires a custom filter function since tokens can be in either position
        """
        def token_filter_func(row):
            row_tokens = [
                str(row.get('token1_symbol', '')).upper(), 
                str(row.get('token2_symbol', '')).upper()
            ]
            return any(token.upper() in row_tokens for token in tokens)
            
        return AdvancedFilter(
            field="token_filter",
            custom_filter=token_filter_func
        )
    
    @staticmethod
    def trend_filter(field: str, days: int, trend_type: str,
                    threshold: float) -> AdvancedFilter:
        """
        Create a filter based on trend analysis
        
        Args:
            field: Field to analyze trend for (e.g., 'apr', 'liquidity')
            days: Number of days to analyze
            trend_type: 'increasing', 'decreasing', or 'stable'
            threshold: Percentage change threshold
        """
        trend_field = f"{field}_change_{days}d"
        
        if trend_type == 'increasing':
            return AdvancedFilter(
                field=trend_field,
                min_value=threshold
            )
        elif trend_type == 'decreasing':
            return AdvancedFilter(
                field=trend_field,
                max_value=-threshold
            )
        else:  # stable
            return AdvancedFilter(
                field=trend_field,
                min_value=-threshold,
                max_value=threshold
            )
    
    @staticmethod
    def prediction_score_filter(min_score: float) -> AdvancedFilter:
        """Create a filter for prediction score threshold"""
        return AdvancedFilter(
            field="prediction_score",
            min_value=min_score
        )
    
    @staticmethod
    def volatility_filter(max_volatility: float) -> AdvancedFilter:
        """Create a filter for volatility threshold"""
        return AdvancedFilter(
            field="volatility",
            max_value=max_volatility
        )
    
    @staticmethod
    def volume_to_liquidity_ratio_filter(min_ratio: float) -> AdvancedFilter:
        """
        Create a filter for volume/liquidity ratio
        This is a derived metric that needs to be calculated
        """
        def ratio_filter_func(row):
            if row.get('liquidity', 0) <= 0:
                return False
            ratio = row.get('volume_24h', 0) / row.get('liquidity', 1)
            return ratio >= min_ratio
            
        return AdvancedFilter(
            field="volume_to_liquidity_ratio",
            custom_filter=ratio_filter_func
        )
    
    # Advanced filtering methods
    
    def calculate_derived_metrics(self) -> None:
        """
        Calculate additional derived metrics for filtering
        Adds new columns to the dataframe
        """
        if self.filtered_data.empty:
            return
            
        # Volume to Liquidity ratio (useful for finding efficient pools)
        self.filtered_data['volume_liquidity_ratio'] = (
            self.filtered_data['volume_24h'] / self.filtered_data['liquidity'].replace(0, np.nan)
        ).fillna(0)
        
        # APR to Volatility ratio (risk-adjusted return)
        if 'apr' in self.filtered_data.columns and 'volatility' in self.filtered_data.columns:
            non_zero_vol = self.filtered_data['volatility'].replace(0, np.nan)
            self.filtered_data['risk_adjusted_apr'] = (
                self.filtered_data['apr'] / non_zero_vol
            ).fillna(0)
            
        # Impermanent Loss Estimate 
        # (simplified - would need price correlation data for accuracy)
        if 'volatility' in self.filtered_data.columns:
            self.filtered_data['impermanent_loss_risk'] = self.filtered_data['volatility'] * 2
        
        logger.info("Calculated derived metrics for filtering")
    
    def rank_pools(self, 
                  metrics: List[str], 
                  weights: List[float]) -> pd.DataFrame:
        """
        Rank pools using a weighted scoring system.
        
        Args:
            metrics: List of metric names to include in ranking
            weights: Corresponding weights for each metric
            
        Returns:
            DataFrame with original data plus rank and score columns
        """
        if len(metrics) != len(weights):
            raise ValueError("Metrics and weights must have the same length")
            
        # First ensure we have filtered data
        if self.filtered_data.empty:
            return pd.DataFrame()
            
        # Create a copy to add ranking columns
        ranked_data = self.filtered_data.copy()
        
        # Normalize each metric to 0-1 scale for fair weighting
        normalized_data = pd.DataFrame()
        
        for metric in metrics:
            if metric not in ranked_data.columns:
                logger.warning(f"Metric '{metric}' not found in data, skipping")
                continue
                
            # For some metrics higher is better, for others lower is better
            higher_is_better = metric not in ['volatility', 'impermanent_loss_risk']
            
            values = ranked_data[metric]
            min_val = values.min()
            max_val = values.max()
            
            if max_val == min_val:  # Avoid division by zero
                normalized_data[f"{metric}_normalized"] = 1 if higher_is_better else 0
            else:
                if higher_is_better:
                    normalized_data[f"{metric}_normalized"] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[f"{metric}_normalized"] = 1 - (values - min_val) / (max_val - min_val)
        
        # Calculate weighted score
        ranked_data['score'] = 0
        
        for i, metric in enumerate(metrics):
            if f"{metric}_normalized" in normalized_data.columns:
                ranked_data['score'] += normalized_data[f"{metric}_normalized"] * weights[i]
        
        # Normalize final score to 0-100 for readability
        min_score = ranked_data['score'].min()
        max_score = ranked_data['score'].max()
        
        if max_score > min_score:
            ranked_data['score'] = 100 * (ranked_data['score'] - min_score) / (max_score - min_score)
        
        # Add rank column
        ranked_data['rank'] = ranked_data['score'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        ranked_data = ranked_data.sort_values('rank')
        
        return ranked_data
    
    def find_similar_pools(self, 
                          pool_id: str, 
                          metrics: List[str] = None,
                          top_n: int = 5) -> pd.DataFrame:
        """
        Find pools that are similar to a reference pool.
        
        Args:
            pool_id: ID of the reference pool
            metrics: Metrics to use for similarity calculation (default: all numeric)
            top_n: Number of similar pools to return
            
        Returns:
            DataFrame with similar pools
        """
        if pool_id not in self.original_data['id'].values:
            logger.error(f"Pool ID '{pool_id}' not found in data")
            return pd.DataFrame()
            
        # Get the reference pool
        reference_pool = self.original_data[self.original_data['id'] == pool_id].iloc[0]
        
        # If metrics not specified, use all numeric columns
        if metrics is None:
            metrics = self.original_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID-like columns and calculated columns
            metrics = [m for m in metrics if m not in ['id', 'rank', 'score']]
        
        # Calculate similarity (Euclidean distance)
        distances = []
        
        for _, pool in self.filtered_data.iterrows():
            distance = 0
            
            for metric in metrics:
                if metric in pool and metric in reference_pool:
                    # Normalize by the range of the metric to make distances comparable
                    metric_range = self.original_data[metric].max() - self.original_data[metric].min()
                    if metric_range > 0:  # Avoid division by zero
                        norm_distance = ((pool[metric] - reference_pool[metric]) / metric_range) ** 2
                        distance += norm_distance
            
            distance = np.sqrt(distance)
            distances.append((pool['id'], distance))
        
        # Convert to DataFrame for easier handling
        distance_df = pd.DataFrame(distances, columns=['id', 'distance'])
        
        # Merge with filtered data
        similar_pools = pd.merge(self.filtered_data, distance_df, on='id')
        
        # Sort by distance (most similar first) and take top N
        similar_pools = similar_pools.sort_values('distance').head(top_n)
        
        return similar_pools
    
    def get_pool_clusters(self, 
                         n_clusters: int = 5, 
                         metrics: List[str] = None) -> pd.DataFrame:
        """
        Cluster pools based on key metrics using K-means.
        
        Args:
            n_clusters: Number of clusters
            metrics: List of metrics to use for clustering
            
        Returns:
            DataFrame with original data plus cluster labels
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn package needed for clustering is not installed")
            return self.filtered_data
        
        if self.filtered_data.empty:
            return pd.DataFrame()
            
        # If metrics not specified, use common pool metrics
        if metrics is None:
            metrics = ['liquidity', 'volume_24h', 'apr']
            metrics = [m for m in metrics if m in self.filtered_data.columns]
        
        # Select only numeric columns that exist
        data_for_clustering = self.filtered_data[metrics].copy()
        
        # Drop rows with NaN values
        data_for_clustering = data_for_clustering.dropna()
        
        if len(data_for_clustering) < n_clusters:
            logger.warning("Not enough data points for clustering")
            return self.filtered_data
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to the data
        clustered_data = self.filtered_data.copy()
        clustered_data.loc[data_for_clustering.index, 'cluster'] = clusters
        
        # Calculate cluster properties
        cluster_stats = []
        
        for i in range(n_clusters):
            cluster_data = clustered_data[clustered_data['cluster'] == i]
            stats = {
                'cluster': i,
                'size': len(cluster_data),
                'mean_liquidity': cluster_data['liquidity'].mean() if 'liquidity' in cluster_data else None,
                'mean_volume': cluster_data['volume_24h'].mean() if 'volume_24h' in cluster_data else None,
                'mean_apr': cluster_data['apr'].mean() if 'apr' in cluster_data else None,
            }
            cluster_stats.append(stats)
        
        # Add cluster description
        cluster_desc = pd.DataFrame(cluster_stats)
        
        # Create labels for clusters
        descriptions = []
        
        for _, row in cluster_desc.iterrows():
            if pd.notna(row['mean_apr']) and pd.notna(row['mean_liquidity']):
                if row['mean_apr'] > cluster_desc['mean_apr'].median():
                    if row['mean_liquidity'] > cluster_desc['mean_liquidity'].median():
                        desc = "High APR, High Liquidity"
                    else:
                        desc = "High APR, Low Liquidity"
                else:
                    if row['mean_liquidity'] > cluster_desc['mean_liquidity'].median():
                        desc = "Low APR, High Liquidity"
                    else:
                        desc = "Low APR, Low Liquidity"
            else:
                desc = f"Cluster {row['cluster']}"
                
            descriptions.append((row['cluster'], desc))
        
        # Create a mapping dictionary
        desc_mapping = dict(descriptions)
        
        # Map descriptions to data
        clustered_data['cluster_description'] = clustered_data['cluster'].map(desc_mapping)
        
        return clustered_data


# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'id': [f'pool_{i}' for i in range(1, 11)],
        'name': [f'Pool {i}' for i in range(1, 11)],
        'dex': ['Raydium', 'Orca', 'Jupiter', 'Raydium', 'Meteora', 'Raydium', 'Orca', 'Saber', 'Raydium', 'Jupiter'],
        'category': ['Major', 'Major', 'DeFi', 'Meme', 'Meme', 'Stablecoin', 'Gaming', 'Stablecoin', 'DeFi', 'Major'],
        'token1_symbol': ['SOL', 'ETH', 'JUP', 'BONK', 'SAMO', 'USDC', 'AURORY', 'USDC', 'RAY', 'BTC'],
        'token2_symbol': ['USDC', 'USDC', 'USDC', 'USDC', 'USDC', 'USDT', 'USDC', 'USDT', 'USDC', 'USDC'],
        'liquidity': [24532890.45, 18245789.12, 5678234.89, 5432167.89, 3456789.01, 54321987.65, 2345678.90, 32145678.90, 4321987.65, 45321987.65],
        'volume_24h': [8763021.32, 6542891.45, 1987654.32, 1987654.32, 876543.21, 8765432.10, 987654.32, 4567890.12, 1543219.87, 12345678.90],
        'apr': [12.87, 11.23, 15.42, 25.67, 22.45, 5.67, 22.45, 4.89, 18.76, 9.87],
        'volatility': [0.05, 0.04, 0.08, 0.12, 0.11, 0.01, 0.09, 0.01, 0.07, 0.03],
        'prediction_score': [85, 72, 68, 94, 88, 60, 83, 55, 75, 65],
        'apr_change_7d': [1.2, -0.8, 0.5, 4.2, 2.8, 0.1, 2.1, 0.08, 1.2, 0.3],
        'apr_change_30d': [-2.1, -1.5, -3.8, 32.7, 15.2, 1.2, 14.5, 0.9, 8.9, 2.5]
    }
    
    # Convert to DataFrame
    pool_df = pd.DataFrame(data)
    
    # Initialize the advanced filtering system
    filter_system = AdvancedFilteringSystem(pool_df)
    
    # Calculate derived metrics
    filter_system.calculate_derived_metrics()
    
    # Apply a chain of filters
    filtered_df = (filter_system
        .add_filter(AdvancedFilteringSystem.dex_filter(['Raydium', 'Orca']))
        .add_filter(AdvancedFilteringSystem.liquidity_filter(min_value=5000000))
        .add_filter(AdvancedFilteringSystem.apr_filter(min_value=10))
        .apply_filters())
    
    print(f"Filtered data has {len(filtered_df)} pools")
    
    # Analyze filter impact
    impact = filter_system.get_filter_impact_analysis()
    print("Filter impact analysis:")
    print(impact)
    
    # Rank pools by weighted scoring
    ranked_df = filter_system.rank_pools(
        metrics=['apr', 'liquidity', 'volume_24h', 'volatility'],
        weights=[0.4, 0.3, 0.2, 0.1]
    )
    
    print("\nTop 3 pools by weighted score:")
    print(ranked_df[['name', 'rank', 'score']].head(3))
    
    # Find similar pools to a reference pool
    similar_df = filter_system.find_similar_pools('pool_1', top_n=3)
    
    print("\nPools similar to pool_1:")
    print(similar_df[['name', 'distance']].head(3))
    
    # Reset filters
    filter_system.reset_filters()
    
    # Apply a different filter chain
    new_filtered_df = (filter_system
        .add_filter(AdvancedFilteringSystem.category_filter(['Meme', 'DeFi']))
        .add_filter(AdvancedFilteringSystem.trend_filter('apr', 7, 'increasing', 0.5))
        .apply_filters())
    
    print(f"\nMeme and DeFi pools with increasing APR: {len(new_filtered_df)}")
    
    # Cluster the pools
    clustered_df = filter_system.get_pool_clusters(n_clusters=3)
    
    print("\nCluster analysis:")
    for cluster, group in clustered_df.groupby('cluster'):
        print(f"Cluster {cluster} ({group['cluster_description'].iloc[0]}): {len(group)} pools")
        print(f"  Average APR: {group['apr'].mean():.2f}%")
        print(f"  Average Liquidity: ${group['liquidity'].mean():,.2f}")
        print(f"  Average Volume: ${group['volume_24h'].mean():,.2f}")