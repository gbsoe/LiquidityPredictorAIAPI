"""
Quick Performance Monitor Check
"""

import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('perf_check')

logger.info("Starting performance monitor check")

# Import the performance monitor
from performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()

# Test basic functionality
logger.info("Testing basic timing functionality")
monitor.start_tracking("test_segment")
time.sleep(1)  # Simulate work
monitor.mark_checkpoint("test_checkpoint")
time.sleep(0.5)  # More work
monitor.stop_tracking("test_segment")

# Get basic stats
segment_time = monitor.segments.get("test_segment", 0)
logger.info(f"Segment time: {segment_time:.2f}s")

# Generate a report
report = monitor.get_report()
logger.info(f"Generated report with {len(report)} items")

# Save report to file
report_file = monitor.save_final_report()
logger.info(f"Saved report to {report_file}")

logger.info("Performance monitor check completed")