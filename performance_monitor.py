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
from datetime import datetime, timedelta
import threading
import logging
import json
import os
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    System performance tracker for monitoring application loading and execution times.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        # Main tracking timestamps
        self.start_time = None
        self.checkpoints = {}
        self.segments = {}
        self.active_tracking = {}
        
        # System stats
        self.system_stats = {
            "token_cache_size": 0,
            "tokens_by_symbol": 0,
            "tokens_by_address": 0,
            "token_cache_hits": 0,
            "token_cache_misses": 0,
            "address_cache_hits": 0,
            "total_pools_loaded": 0,
            "total_tokens_loaded": 0
        }
        
        # Periodic stats collection
        self.stats_thread = None
        self.collecting_stats = False
        
        # Ensure directory exists
        self.report_dir = "logs"
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
    
    def start_tracking(self, label: str = "application") -> None:
        """
        Start performance tracking with a given label.
        
        Args:
            label: Label for this tracking session
        """
        if self.start_time is None:
            self.start_time = time.time()
            logger.info(f"Started performance tracking at {datetime.now().isoformat()}")
        
        # Mark the start of this specific tracking segment
        self.active_tracking[label] = time.time()
        logger.info(f"Started tracking segment: {label}")
        
        # Record as a checkpoint too
        self.checkpoints[f"{label}_start"] = time.time()
    
    def stop_tracking(self, label: str) -> float:
        """
        Stop tracking a specific segment and return the duration.
        
        Args:
            label: Label of the segment to stop tracking
            
        Returns:
            Duration in seconds
        """
        if label in self.active_tracking:
            start = self.active_tracking[label]
            end = time.time()
            duration = end - start
            
            # Record the segment duration
            self.segments[label] = duration
            
            # Record as a checkpoint too
            self.checkpoints[f"{label}_end"] = end
            
            # Remove from active tracking
            del self.active_tracking[label]
            
            logger.info(f"Completed segment '{label}' in {duration:.2f}s")
            return duration
        else:
            logger.warning(f"Attempted to stop tracking '{label}' which wasn't started")
            return 0
    
    def mark_checkpoint(self, name: str) -> None:
        """
        Mark a named checkpoint with the current time.
        
        Args:
            name: Name of the checkpoint
        """
        self.checkpoints[name] = time.time()
        
        # Calculate time since start if we have a start time
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            logger.info(f"Checkpoint '{name}' reached at {elapsed:.2f}s")
    
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
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def time_between_checkpoints(self, start_point: str, end_point: str) -> Optional[float]:
        """
        Calculate time between two checkpoints.
        
        Args:
            start_point: Starting checkpoint name
            end_point: Ending checkpoint name
            
        Returns:
            Time difference in seconds or None if checkpoints don't exist
        """
        if start_point in self.checkpoints and end_point in self.checkpoints:
            return self.checkpoints[end_point] - self.checkpoints[start_point]
        return None
    
    def update_system_stats(self, stats_dict: Dict[str, Any]) -> None:
        """
        Update system statistics with new values.
        
        Args:
            stats_dict: Dictionary of stats to update
        """
        self.system_stats.update(stats_dict)
    
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
        if self.stats_thread is not None and self.stats_thread.is_alive():
            logger.warning("Stats collection already running")
            return
        
        self.collecting_stats = True
        
        def collect_stats():
            while self.collecting_stats:
                try:
                    # Collect token service stats
                    if token_service is not None and hasattr(token_service, 'get_stats'):
                        token_stats = token_service.get_stats()
                        self.system_stats.update({
                            "token_cache_size": token_stats.get("cache_size", 0),
                            "tokens_by_symbol": token_stats.get("tokens_by_symbol", 0),
                            "tokens_by_address": token_stats.get("tokens_by_address", 0),
                            "token_cache_hits": token_stats.get("cache_hits", 0),
                            "token_cache_misses": token_stats.get("cache_misses", 0),
                            "address_cache_hits": token_stats.get("direct_address_hits", 0)
                        })
                    
                    # Collect data service stats
                    if data_service is not None and hasattr(data_service, 'get_stats'):
                        data_stats = data_service.get_stats()
                        self.system_stats.update({
                            "total_pools_loaded": data_stats.get("total_pools", 0),
                            "pool_load_time": data_stats.get("load_time", 0)
                        })
                    
                    # Collect prediction service stats
                    if prediction_service is not None and hasattr(prediction_service, 'get_stats'):
                        pred_stats = prediction_service.get_stats()
                        self.system_stats.update({
                            "prediction_models_loaded": pred_stats.get("models_loaded", 0),
                            "prediction_time_avg": pred_stats.get("prediction_time_avg", 0)
                        })
                    
                    # Add timestamp
                    self.system_stats["last_updated"] = datetime.now().isoformat()
                    
                    # Save current stats to file
                    self.save_interim_report()
                    
                except Exception as e:
                    logger.error(f"Error collecting stats: {str(e)}")
                
                # Wait for next collection
                time.sleep(interval)
        
        self.stats_thread = threading.Thread(target=collect_stats, daemon=True)
        self.stats_thread.start()
        logger.info(f"Started periodic stats collection every {interval}s")
    
    def stop_periodic_stats_collection(self) -> None:
        """Stop the periodic stats collection."""
        self.collecting_stats = False
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=1.0)
        logger.info("Stopped periodic stats collection")
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        now = time.time()
        total_time = now - self.start_time if self.start_time else 0
        
        # Calculate time to key events
        time_to_data_loaded = self.time_between_checkpoints("app_initialization_start", "data_loaded") or 0
        time_to_predictions = self.time_between_checkpoints("app_initialization_start", "predictions_ready") or 0
        time_to_ui_ready = self.time_between_checkpoints("app_initialization_start", "ui_ready") or 0
        
        # Calculate segment durations
        segment_durations = {}
        for segment, duration in self.segments.items():
            segment_durations[segment] = f"{duration:.2f}s"
        
        # Create checkpoint timeline
        timeline = []
        if self.start_time:
            for checkpoint, timestamp in sorted(self.checkpoints.items(), key=lambda x: x[1]):
                elapsed = timestamp - self.start_time
                timeline.append({
                    "event": checkpoint,
                    "time": f"{elapsed:.2f}s",
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat()
                })
        
        # Build the report
        report = {
            "summary": {
                "total_runtime": f"{total_time:.2f}s",
                "time_to_data_loaded": f"{time_to_data_loaded:.2f}s",
                "time_to_predictions": f"{time_to_predictions:.2f}s",
                "time_to_ui_ready": f"{time_to_ui_ready:.2f}s",
                "started_at": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "report_generated_at": datetime.now().isoformat()
            },
            "system_stats": self.system_stats,
            "segments": segment_durations,
            "timeline": timeline
        }
        
        return report
    
    def save_interim_report(self) -> None:
        """Save current stats to an interim report file."""
        try:
            report = self.get_report()
            filename = os.path.join(self.report_dir, "performance_interim.json")
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving interim report: {str(e)}")
    
    def save_final_report(self) -> str:
        """
        Save the final performance report to a file.
        
        Returns:
            Path to the saved report file
        """
        try:
            report = self.get_report()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.report_dir, f"performance_report_{timestamp}.json")
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved performance report to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving performance report: {str(e)}")
            return ""

# Singleton instance for global access
_instance = None
_lock = threading.Lock()

def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the singleton performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    global _instance, _lock
    
    with _lock:
        if _instance is None:
            _instance = PerformanceMonitor()
    
    return _instance