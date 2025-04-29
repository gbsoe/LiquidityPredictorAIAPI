"""
Base collector for data services.

This module provides the base class for all data collectors,
ensuring a consistent interface and shared functionality.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class CollectorStatus(Enum):
    """Status of a collector."""
    IDLE = "idle"
    COLLECTING = "collecting"
    SUCCESS = "success"
    ERROR = "error"

class BaseCollector(ABC):
    """
    Abstract base class for data collectors.
    
    All data collectors should inherit from this class and implement
    the required methods to ensure a consistent interface.
    """
    
    def __init__(self, name: str):
        """
        Initialize the collector.
        
        Args:
            name: Name of the collector
        """
        self.name = name
        self.status = CollectorStatus.IDLE
        self.last_collection_time = None
        self.last_error = None
        self.collection_count = 0
        self.error_count = 0
        self.last_collection_items = 0
        
    def collect(self) -> List[Dict[str, Any]]:
        """
        Collect data from the source.
        
        Returns:
            List of collected data items
        """
        # Record the status and start time
        self.status = CollectorStatus.COLLECTING
        start_time = time.time()
        
        try:
            # Call the implementation-specific collection method
            data = self._collect_data()
            
            # Update statistics
            self.last_collection_time = time.time()
            self.last_collection_items = len(data) if data else 0
            self.collection_count += 1
            self.status = CollectorStatus.SUCCESS
            self.last_error = None
            
            # Log success
            logger.info(
                f"Collector '{self.name}' successfully collected {self.last_collection_items} items "
                f"in {time.time() - start_time:.2f}s"
            )
            
            return data
        except Exception as e:
            # Update error statistics
            self.error_count += 1
            self.last_error = str(e)
            self.status = CollectorStatus.ERROR
            
            # Log error
            logger.error(f"Collector '{self.name}' failed: {str(e)}")
            
            # Re-raise the exception
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the collector.
        
        Returns:
            Dictionary with collector statistics
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "last_collection_time": self.last_collection_time,
            "last_collection_items": self.last_collection_items,
            "collection_count": self.collection_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "success_rate": (
                (self.collection_count - self.error_count) / self.collection_count
                if self.collection_count > 0
                else 0
            )
        }
    
    @abstractmethod
    def _collect_data(self) -> List[Dict[str, Any]]:
        """
        Implementation-specific method to collect data.
        
        Returns:
            List of collected data items
        """
        pass