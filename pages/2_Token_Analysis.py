
"""
Token Analysis - Comprehensive Token Explorer for Solpool Insight

This page provides detailed analysis for tokens across Solana DEXes,
with comprehensive metadata and visualization of token relationships.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our token service
from token_data_service import get_token_service

# Page configuration
st.set_page_config(
    page_title="Token Analysis - Solpool Insight",
    page_icon="💰",
    layout="wide"
)
