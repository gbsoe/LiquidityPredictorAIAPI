"""
Quick test runner for DeFi API
"""

from defi_aggregation_api import DefiAggregationAPI

# Run the test
api = DefiAggregationAPI()
api.run_quick_test(max_calls=3)