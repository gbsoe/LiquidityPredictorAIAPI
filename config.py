"""
Configuration file for the Solana Liquidity Pool Analysis System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Raydium API settings
RAYDIUM_API_KEY = os.environ.get('RAYDIUM_API_KEY', 'demo-api-key')
RAYDIUM_API_URL = os.environ.get('RAYDIUM_API_URL', 'https://raydium-api.example.com')

# Database settings
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database/liquidity_pools.db')