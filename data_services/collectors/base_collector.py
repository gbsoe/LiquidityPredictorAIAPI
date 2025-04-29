"""
Base data collector class for SolPool Insight.

This module defines the base class for all data collectors, including
common functionality and interfaces.
"""

import time
import json
import logging
import threading
import requests
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger('data_collectors')

class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    Implements common functionality like rate limiting, error handling,
    and result standardization.
    """
    
    def __init__(self, 
                name: str, 
                priority: int = 5,
                rate_limit_calls: int = 10,
                rate_limit_period: int = 60,
                max_retries: int = 3,
                backoff_factor: float = 1.5,
                timeout: int = 30):
        """
        Initialize a data collector.
        
        Args:
            name: Collector name for identification
            priority: Priority level (1-10, higher means more authoritative)
            rate_limit_calls: Number of calls allowed in rate_limit_period
            rate_limit_period: Period in seconds for rate limiting
            max_retries: Maximum number of retries on failure
            backoff_factor: Exponential backoff factor for retries
            timeout: Request timeout in seconds
        """
        self.name = name
        self.priority = priority
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        
        # Rate limiting state
        self.calls = []  # List of timestamps of recent calls
        self._lock = threading.RLock()
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_data_points = 0
        self.last_call_time = None
        self.last_error = None
        self.last_success_time = None
        
        logger.info(f"Initialized {name} collector with priority {priority}")
    
    def apply_rate_limiting(self) -> bool:
        """
        Apply rate limiting based on configuration.
        
        Returns:
            True if the call is allowed, False if it would exceed rate limits
        """
        with self._lock:
            current_time = time.time()
            
            # Remove old calls from tracking
            cutoff = current_time - self.rate_limit_period
            self.calls = [t for t in self.calls if t > cutoff]
            
            # Check if we're at the limit
            if len(self.calls) >= self.rate_limit_calls:
                logger.warning(f"{self.name} rate limit exceeded: {len(self.calls)} calls in last {self.rate_limit_period} seconds")
                return False
            
            # Add this call
            self.calls.append(current_time)
            return True
    
    def wait_for_rate_limit(self, max_wait: int = 30) -> bool:
        """
        Wait until a call is allowed by rate limiting.
        
        Args:
            max_wait: Maximum seconds to wait
            
        Returns:
            True if call is allowed after waiting, False if max_wait exceeded
        """
        start = time.time()
        while not self.apply_rate_limiting():
            # Calculate how long to wait
            wait_time = 0.5  # Default wait
            
            if self.calls:
                # Wait until oldest call expires
                oldest_call = min(self.calls)
                wait_time = oldest_call + self.rate_limit_period - time.time()
                wait_time = max(0.1, min(wait_time, 5))  # Between 0.1 and 5 seconds
            
            if time.time() - start > max_wait:
                logger.warning(f"{self.name} exceeded max wait time for rate limiting")
                return False
                
            # Wait
            time.sleep(wait_time)
        
        return True
    
    def _handle_request_with_retries(self, 
                                    request_func, 
                                    *args, 
                                    **kwargs) -> Tuple[Any, bool]:
        """
        Handle a request with retries and exponential backoff.
        
        Args:
            request_func: Function to call for the request
            *args, **kwargs: Arguments to pass to request_func
            
        Returns:
            Tuple of (response, success)
        """
        self.total_calls += 1
        self.last_call_time = datetime.now()
        
        retry_wait = 1.0
        
        for attempt in range(self.max_retries):
            try:
                # Respect rate limits
                if not self.wait_for_rate_limit():
                    raise Exception("Rate limit waiting timeout exceeded")
                
                # Make request
                response = request_func(*args, **kwargs)
                
                # Success
                self.successful_calls += 1
                self.last_success_time = datetime.now()
                return response, True
                
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"{self.name} attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry with exponential backoff
                    time.sleep(retry_wait)
                    retry_wait *= self.backoff_factor
        
        # All retries failed
        self.failed_calls += 1
        return None, False
    
    def make_http_request(self, 
                        url: str, 
                        method: str = 'GET', 
                        params: Dict = None,
                        headers: Dict = None,
                        data: Dict = None,
                        json_data: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request with retries and error handling.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: URL parameters
            headers: HTTP headers
            data: Form data
            json_data: JSON data for POST/PUT
            
        Returns:
            Response data (usually JSON decoded) or None on failure
        """
        def do_request():
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                data=data,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Try to parse JSON
            try:
                return response.json()
            except ValueError:
                # Not JSON
                return response.text
        
        result, success = self._handle_request_with_retries(do_request)
        return result if success else None
    
    def standardize_pool_data(self, 
                            pool_data: Dict[str, Any], 
                            source: str) -> Dict[str, Any]:
        """
        Standardize pool data to a common format.
        
        Args:
            pool_data: Raw pool data
            source: Data source identifier
            
        Returns:
            Standardized pool data
        """
        # Extract required fields with fallbacks
        pool_id = pool_data.get('poolId', pool_data.get('id', pool_data.get('address')))
        
        if not pool_id:
            raise ValueError("Pool data missing ID")
        
        # Get timestamp
        timestamp = datetime.now().isoformat()
        
        # Extract metrics
        metrics = pool_data.get('metrics', {})
        if isinstance(metrics, list) and metrics:
            metrics = metrics[0]  # Some sources use arrays
        
        # Get name and extract tokens if needed
        name = pool_data.get('name', '')
        token1 = None
        token2 = None
        
        tokens = pool_data.get('tokens', [])
        if tokens and len(tokens) >= 2:
            token1 = tokens[0].get('symbol', tokens[0].get('name'))
            token2 = tokens[1].get('symbol', tokens[1].get('name'))
        else:
            # Try to extract from name
            if name and '-' in name:
                parts = name.split('-')
                if len(parts) >= 2:
                    token1 = parts[0].strip()
                    token2_parts = parts[1].split(' ')
                    token2 = token2_parts[0].strip()
        
        # Create standardized data
        standardized = {
            'id': pool_id,
            'poolId': pool_id,  # Ensure both formats available
            'name': name,
            'timestamp': timestamp,
            'source': pool_data.get('source', pool_data.get('dex')),
            'data_source': source,
            'tokens': [],
            'metrics': {
                'tvl': metrics.get('tvl', metrics.get('liquidity')),
                'apy': metrics.get('apy', metrics.get('apr')),
                'apy24h': metrics.get('apy24h', metrics.get('apr24h')),
                'apy7d': metrics.get('apy7d', metrics.get('apr7d')),
                'volume24h': metrics.get('volumeUsd', metrics.get('volume24h', metrics.get('volume'))),
                'fee': metrics.get('fee', 0),
                'priceRatio': metrics.get('priceRatio', metrics.get('price_ratio'))
            }
        }
        
        # Add tokens if found
        if token1:
            standardized['tokens'].append({'symbol': token1})
        if token2:
            standardized['tokens'].append({'symbol': token2})
        
        # Add direct token references for easier access
        standardized['token1'] = token1
        standardized['token2'] = token2
        
        return standardized
    
    def save_backup(self, 
                   data: List[Dict[str, Any]], 
                   filename: str = None,
                   backup_dir: str = None) -> Optional[str]:
        """
        Save a backup of collected data.
        
        Args:
            data: Data to save
            filename: Custom filename, defaults to timestamped name
            backup_dir: Directory to save in, defaults to standard backup location
            
        Returns:
            Path to backup file or None on failure
        """
        try:
            if not backup_dir:
                from ..config import BACKUP_DIR
                backup_dir = BACKUP_DIR
            
            # Ensure directory exists
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.name}_{timestamp}.json"
            
            # Full path
            filepath = Path(backup_dir) / filename
            
            # Save data
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved backup to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save backup: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collector statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'name': self.name,
            'priority': self.priority,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / max(1, self.total_calls),
            'total_data_points': self.total_data_points,
            'last_call_time': self.last_call_time.isoformat() if self.last_call_time else None,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
            'last_error': self.last_error
        }
    
    @abstractmethod
    def collect(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect data from the source.
        
        Returns:
            Tuple of (data_list, success)
        """
        pass