"""
Configuration file for the Solana Liquidity Pool Analysis System
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

# Raydium API configuration
RAYDIUM_API_KEY = os.getenv("RAYDIUM_API_KEY")
RAYDIUM_API_URL = os.getenv("RAYDIUM_API_URL")

# Data collection settings
COLLECTION_INTERVAL_MINUTES = int(os.getenv("COLLECTION_INTERVAL_MINUTES", "60"))
MAX_POOLS_TO_PROCESS = int(os.getenv("MAX_POOLS_TO_PROCESS", "100"))

# Machine learning settings
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./models")
PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", "7"))
TRAINING_INTERVAL_HOURS = int(os.getenv("TRAINING_INTERVAL_HOURS", "24"))

# System settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "True").lower() == "true"

# Dashboard settings
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", "300"))
TOP_POOLS_COUNT = int(os.getenv("TOP_POOLS_COUNT", "10"))

# Solana blockchain settings
SOLANA_RPC_ENDPOINT = os.getenv("SOLANA_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")