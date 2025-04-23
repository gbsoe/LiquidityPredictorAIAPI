import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import sqlite3
import requests
import plotly.express as px
import plotly.graph_objects as go

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.visualization import create_metrics_chart, create_pool_comparison_chart
from utils.data_processor import get_top_pools, get_blockchain_stats, get_prediction_metrics
from database.db_operations import DBManager

# Page configuration
st.set_page_config(
    page_title="Solana Liquidity Pool Analysis System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection
@st.cache_resource
def get_db_connection():
    return DBManager()

db = get_db_connection()

# Header with title and description
st.title("Solana Liquidity Pool Analysis and Prediction System")

# Dashboard introduction with images
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    This application provides real-time analysis and machine learning-powered predictions for Solana blockchain liquidity pools.
    Monitor performance metrics, analyze trends, and get predictive insights to optimize your liquidity provision strategy.
    """)

with col2:
    st.image("https://images.unsplash.com/photo-1526378800651-c32d170fe6f8", 
             caption="Blockchain Technology Visualization")

# Blockchain stats section
st.subheader("Solana Blockchain Metrics")

try:
    blockchain_stats = get_blockchain_stats(db)
    
    if blockchain_stats is not None:
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            st.metric("Current Slot", f"{blockchain_stats['slot']:,}", "")
        
        with metrics_cols[1]:
            st.metric("Block Height", f"{blockchain_stats['block_height']:,}", "")
        
        with metrics_cols[2]:
            st.metric("Average TPS", f"{blockchain_stats['avg_tps']:.2f}", "")
        
        with metrics_cols[3]:
            sol_price = blockchain_stats.get('sol_price', 0)
            if sol_price:
                st.metric("SOL Price", f"${sol_price:.2f}", "")
            else:
                st.metric("SOL Price", "Not available", "")
    else:
        st.warning("Blockchain statistics are currently unavailable. Please check back later.")
except Exception as e:
    st.error(f"Error loading blockchain metrics: {str(e)}")

# Top pools overview
st.subheader("Top Liquidity Pools Overview")

top_pools_tabs = st.tabs(["By Liquidity", "By Volume", "By APR"])

try:
    # By Liquidity Tab
    with top_pools_tabs[0]:
        liquidity_pools = get_top_pools(db, 'liquidity', 5)
        if not liquidity_pools.empty:
            # Create a bar chart for liquidity
            fig = px.bar(
                liquidity_pools,
                x='name',
                y='liquidity',
                title="Top Pools by Liquidity",
                labels={'liquidity': 'Liquidity (USD)', 'name': 'Pool Name'},
                color='liquidity',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show as table
            st.dataframe(
                liquidity_pools[['name', 'liquidity', 'volume_24h', 'apr']].rename(
                    columns={'name': 'Pool Name', 'liquidity': 'Liquidity (USD)', 
                            'volume_24h': 'Volume 24h (USD)', 'apr': 'APR (%)'}
                ).reset_index(drop=True),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No pool data available. The system may be collecting initial data.")

    # By Volume Tab
    with top_pools_tabs[1]:
        volume_pools = get_top_pools(db, 'volume', 5)
        if not volume_pools.empty:
            # Create a bar chart for volume
            fig = px.bar(
                volume_pools,
                x='name',
                y='volume_24h',
                title="Top Pools by 24h Volume",
                labels={'volume_24h': 'Volume (USD)', 'name': 'Pool Name'},
                color='volume_24h',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show as table
            st.dataframe(
                volume_pools[['name', 'liquidity', 'volume_24h', 'apr']].rename(
                    columns={'name': 'Pool Name', 'liquidity': 'Liquidity (USD)', 
                            'volume_24h': 'Volume 24h (USD)', 'apr': 'APR (%)'}
                ).reset_index(drop=True),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No volume data available. The system may be collecting initial data.")

    # By APR Tab
    with top_pools_tabs[2]:
        apr_pools = get_top_pools(db, 'apr', 5)
        if not apr_pools.empty:
            # Create a bar chart for APR
            fig = px.bar(
                apr_pools,
                x='name',
                y='apr',
                title="Top Pools by APR",
                labels={'apr': 'APR (%)', 'name': 'Pool Name'},
                color='apr',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show as table
            st.dataframe(
                apr_pools[['name', 'liquidity', 'volume_24h', 'apr']].rename(
                    columns={'name': 'Pool Name', 'liquidity': 'Liquidity (USD)', 
                            'volume_24h': 'Volume 24h (USD)', 'apr': 'APR (%)'}
                ).reset_index(drop=True),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No APR data available. The system may be collecting initial data.")
except Exception as e:
    st.error(f"Error loading pool data: {str(e)}")

# ML Prediction Overview
st.subheader("ML Prediction Insights")

try:
    prediction_metrics = get_prediction_metrics(db)
    
    if prediction_metrics is not None and not prediction_metrics.empty:
        # Show prediction overview
        st.write("Latest machine learning predictions for top pools:")
        
        # Format the dataframe for display
        display_df = prediction_metrics[['pool_name', 'predicted_apr', 'performance_class', 'risk_score']].copy()
        display_df.columns = ['Pool Name', 'Predicted APR (%)', 'Performance Class', 'Risk Score']
        
        # Add color coding for performance class
        def highlight_performance(val):
            if val == 'high':
                return 'background-color: rgba(102, 187, 106, 0.2)'
            elif val == 'medium':
                return 'background-color: rgba(255, 193, 7, 0.2)'
            elif val == 'low':
                return 'background-color: rgba(244, 67, 54, 0.2)'
            return ''
        
        # Add color coding for risk score
        def highlight_risk(val):
            try:
                val_float = float(val)
                if val_float < 0.3:
                    return 'background-color: rgba(102, 187, 106, 0.2)'
                elif val_float < 0.7:
                    return 'background-color: rgba(255, 193, 7, 0.2)'
                else:
                    return 'background-color: rgba(244, 67, 54, 0.2)'
            except:
                return ''
        
        # Apply styling
        styled_df = display_df.style\
            .format({'Predicted APR (%)': '{:.2f}', 'Risk Score': '{:.2f}'})\
            .applymap(highlight_performance, subset=['Performance Class'])\
            .applymap(highlight_risk, subset=['Risk Score'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Show a note about ML predictions
        st.info("""
        **Performance Class**: Categorizes pools as high, medium, or low performers based on historical APR trends.
        
        **Risk Score**: Indicates potential impermanent loss risk, with higher values representing higher risk.
        
        For detailed predictions and analysis, visit the Predictions page.
        """)
    else:
        st.info("No prediction data available yet. The models may still be training or collecting initial data.")
except Exception as e:
    st.error(f"Error loading prediction insights: {str(e)}")

# System status
st.subheader("System Status")

try:
    # Check backend status
    backend_status = "Online"
    backend_error = None
    
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code != 200:
            backend_status = "Error"
            backend_error = f"Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        backend_status = "Offline"
        backend_error = str(e)
    
    # Get database metrics
    pool_count = 0
    latest_update = "Unknown"
    
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Count pools
        cursor.execute("SELECT COUNT(*) FROM pool_data")
        pool_count = cursor.fetchone()[0]
        
        # Get latest update time
        cursor.execute("SELECT MAX(timestamp) FROM pool_metrics")
        latest_timestamp = cursor.fetchone()[0]
        
        if latest_timestamp:
            latest_datetime = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
            latest_update = latest_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        conn.close()
    except Exception as e:
        latest_update = f"Error: {str(e)}"
    
    # Display status
    status_cols = st.columns(3)
    
    with status_cols[0]:
        if backend_status == "Online":
            st.success("Backend Services: Online")
        elif backend_status == "Error":
            st.warning(f"Backend Services: Error - {backend_error}")
        else:
            st.error(f"Backend Services: Offline - {backend_error}")
    
    with status_cols[1]:
        st.info(f"Monitored Pools: {pool_count}")
    
    with status_cols[2]:
        st.info(f"Latest Data Update: {latest_update}")
    
except Exception as e:
    st.error(f"Error checking system status: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Solana Liquidity Pool Analysis System with Machine Learning • Data updates hourly")
