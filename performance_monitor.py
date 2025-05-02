"""
Performance Monitoring System

This module tracks system performance metrics including:
1. Initialization time
2. Data loading time
3. Time to first prediction
4. Token resolution statistics

Usage:
    from performance_monitor import PerformanceMonitor
    
    # Start tracking at application entry point
    monitor = PerformanceMonitor()
    monitor.start_tracking("app_initialization")
    
    # ... load data, initialize modules ...
    
    # Mark key performance checkpoints
    monitor.mark_checkpoint("data_loaded")
    
    # ... continue with app logic ...
    
    # Mark when predictions are available
    monitor.mark_checkpoint("predictions_ready")
    
    # Get report at any time
    performance_report = monitor.get_report()
"""

import time
import json
import os
import logging
import psutil
from datetime import datetime
from threading import Timer, Lock
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_monitor")

# Singleton instance
_instance = None
_instance_lock = Lock()

class PerformanceMonitor:
    """
    System performance tracker for monitoring application loading and execution times.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        # Track segments of code execution time
        self.segments = {}
        self.active_segments = {}
        
        # Track checkpoints for key events
        self.checkpoints = {}
        
        # Track overall start time
        self.start_time = time.time()
        
        # Track system statistics
        self.stats = {
            'token_cache_size': 0,
            'token_lookup_times': [],
            'prediction_times': [],
            'total_pools_loaded': 0,
            'tokens_by_symbol': 0,
            'tokens_by_address': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'peak_memory_mb': 0
        }
        
        # Track timeline of events
        self.timeline = []
        self._add_timeline_event("monitor_initialized")
        
        # System monitoring
        self.collection_timer = None
        self.collecting_stats = False
        
        # Track memory usage at start
        self.initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
        
        logger.info("Performance monitor initialized")
    
    def start_tracking(self, label: str = "application") -> None:
        """
        Start performance tracking with a given label.
        
        Args:
            label: Label for this tracking session
        """
        self.active_segments[label] = time.time()
        self._add_timeline_event(f"{label}_tracking_started")
        logger.debug(f"Started tracking segment: {label}")
    
    def stop_tracking(self, label: str) -> float:
        """
        Stop tracking a specific segment and return the duration.
        
        Args:
            label: Label of the segment to stop tracking
            
        Returns:
            Duration in seconds
        """
        if label in self.active_segments:
            start_time = self.active_segments[label]
            duration = time.time() - start_time
            self.segments[label] = duration
            del self.active_segments[label]
            self._add_timeline_event(f"{label}_tracking_stopped", {"duration": f"{duration:.2f}s"})
            logger.debug(f"Stopped tracking segment: {label}, duration: {duration:.2f}s")
            return duration
        return 0
    
    def mark_checkpoint(self, name: str) -> None:
        """
        Mark a named checkpoint with the current time.
        
        Args:
            name: Name of the checkpoint
        """
        current_time = time.time()
        self.checkpoints[name] = current_time
        time_since_start = current_time - self.start_time
        self._add_timeline_event(f"checkpoint_{name}", {"time_since_start": f"{time_since_start:.2f}s"})
        logger.debug(f"Marked checkpoint: {name} at {time_since_start:.2f}s after start")
    
    def get_checkpoint_time(self, name: str) -> Optional[float]:
        """
        Get the timestamp for a specific checkpoint.
        
        Args:
            name: Name of the checkpoint
            
        Returns:
            Timestamp or None if checkpoint doesn't exist
        """
        return self.checkpoints.get(name)
    
    def time_since_start(self) -> float:
        """
        Get time elapsed since tracking started.
        
        Returns:
            Elapsed time in seconds or 0 if tracking hasn't started
        """
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def time_between_checkpoints(self, start_point: str, end_point: str) -> Optional[float]:
        """
        Calculate time between two checkpoints.
        
        Args:
            start_point: Starting checkpoint name
            end_point: Ending checkpoint name
            
        Returns:
            Time difference in seconds or None if checkpoints don't exist
        """
        start_time = self.get_checkpoint_time(start_point)
        end_time = self.get_checkpoint_time(end_point)
        
        if start_time and end_time:
            return end_time - start_time
        return None
    
    def update_system_stats(self, stats_dict: Dict[str, Any]) -> None:
        """
        Update system statistics with new values.
        
        Args:
            stats_dict: Dictionary of stats to update
        """
        for key, value in stats_dict.items():
            self.stats[key] = value
            
        # Always update memory usage
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], current_memory)
    
    def start_periodic_stats_collection(self, interval: int = 10, token_service=None, 
                                        data_service=None, prediction_service=None) -> None:
        """
        Start collecting stats periodically from services.
        
        Args:
            interval: Collection interval in seconds
            token_service: Token data service instance
            data_service: Data service instance
            prediction_service: Prediction service instance
        """
        if self.collecting_stats:
            return
        
        self.collecting_stats = True
        
        def collect_stats():
            try:
                # Update memory usage
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
                self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], current_memory)
                
                # Collect token service stats
                if token_service and hasattr(token_service, 'get_stats'):
                    token_stats = token_service.get_stats()
                    for key, value in token_stats.items():
                        if key in self.stats:
                            self.stats[key] = value
                
                # Schedule next collection if still active
                if self.collecting_stats:
                    self.collection_timer = Timer(interval, collect_stats)
                    self.collection_timer.daemon = True
                    self.collection_timer.start()
            except Exception as e:
                logger.error(f"Error in stats collection: {str(e)}")
        
        # Start initial collection
        collect_stats()
    
    def stop_periodic_stats_collection(self) -> None:
        """Stop the periodic stats collection."""
        self.collecting_stats = False
        if self.collection_timer:
            self.collection_timer.cancel()
            self.collection_timer = None
    
    def _add_timeline_event(self, event: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add an event to the timeline.
        
        Args:
            event: Event name
            metadata: Additional event metadata
        """
        timestamp = time.time()
        time_since_start = timestamp - self.start_time
        
        event_data = {
            "timestamp": timestamp,
            "time": f"{time_since_start:.2f}s",
            "event": event,
            "datetime": datetime.now().isoformat()
        }
        
        if metadata:
            event_data.update(metadata)
            
        self.timeline.append(event_data)
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate key metrics
        time_to_data = self.time_between_checkpoints('monitor_initialized', 'data_loaded')
        time_to_predictions = self.time_between_checkpoints('monitor_initialized', 'predictions_ready')
        time_to_ui = self.time_between_checkpoints('monitor_initialized', 'ui_ready')
        
        # Calculate memory usage
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
        memory_growth = current_memory - self.initial_memory
        
        # Finalize any active segments
        active_segments = list(self.active_segments.keys())
        for segment in active_segments:
            self.stop_tracking(segment)
        
        # Build report
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_runtime": f"{self.time_since_start():.2f}s",
                "time_to_data_loaded": f"{time_to_data:.2f}s" if time_to_data else "N/A",
                "time_to_predictions": f"{time_to_predictions:.2f}s" if time_to_predictions else "N/A", 
                "time_to_ui_ready": f"{time_to_ui:.2f}s" if time_to_ui else "N/A",
                "peak_memory_mb": f"{self.stats['peak_memory_mb']:.2f} MB",
                "memory_growth_mb": f"{memory_growth:.2f} MB"
            },
            "segments": {key: f"{value:.2f}s" for key, value in self.segments.items()},
            "checkpoints": {key: datetime.fromtimestamp(value).isoformat() for key, value in self.checkpoints.items()},
            "stats": self.stats,
            "timeline": self.timeline
        }
        
        return report
    
    def save_interim_report(self) -> None:
        """Save current stats to an interim report file."""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Generate interim report filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/performance_interim_{timestamp}.json"
            
            # Get current report
            report = self.get_report()
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved interim performance report to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving interim report: {str(e)}")
            return None
    
    def save_final_report(self) -> str:
        """
        Save the final performance report to a file.
        
        Returns:
            Path to the saved report file
        """
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/performance_report_{timestamp}.json"
            
            # Get final report
            report = self.get_report()
            
            # Add any final metrics
            report["summary"]["final_token_cache_size"] = self.stats.get("token_cache_size", 0)
            report["summary"]["total_pools_loaded"] = self.stats.get("total_pools_loaded", 0)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            # Also save a simpler summary file
            summary_filename = f"logs/performance_summary_{timestamp}.txt"
            with open(summary_filename, 'w') as f:
                f.write("PERFORMANCE SUMMARY\n")
                f.write("=================\n\n")
                f.write(f"Timestamp: {report['summary']['timestamp']}\n")
                f.write(f"Total runtime: {report['summary']['total_runtime']}\n")
                f.write(f"Time to data loaded: {report['summary']['time_to_data_loaded']}\n")
                f.write(f"Time to predictions: {report['summary']['time_to_predictions']}\n")
                f.write(f"Time to UI ready: {report['summary']['time_to_ui_ready']}\n")
                f.write(f"Peak memory usage: {report['summary']['peak_memory_mb']}\n")
                f.write(f"Token cache size: {report['summary']['final_token_cache_size']}\n")
                f.write(f"Total pools loaded: {report['summary']['total_pools_loaded']}\n")
                
            logger.info(f"Saved final performance report to {filename}")
            logger.info(f"Saved summary to {summary_filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving final report: {str(e)}")
            return "Error saving report"

def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the singleton performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    global _instance
    
    with _instance_lock:
        if _instance is None:
            _instance = PerformanceMonitor()
        
    return _instance