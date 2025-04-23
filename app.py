import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import psycopg2
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.visualization import create_metrics_chart, create_pool_comparison_chart
from utils.data_processor import get_top_pools, get_blockchain_stats, get_prediction_metrics
from database.db_operations import DBManager
from data_ingestion.raydium_api_client import RaydiumAPIClient
import config

# Page configuration
st.set_page_config(
    page_title="Solana Liquidity Pool Analysis System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .dashboard-section {
        margin-top: 2rem;
    }
    .highlight-text {
        color: #1E88E5;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #757575;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
def get_db_connection():
    return DBManager()

db = get_db_connection()

# Function to format numbers for display
def format_number(num):
    if num is None:
        return "N/A"
    if isinstance(num, str):
        return num
    if num >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num/1_000:.2f}K"
    return f"${num:.2f}"

def format_percentage(num):
    if num is None:
        return "N/A"
    return f"{num:.2f}%"

def highlight_performance(val):
    """
    Highlight performance metrics based on value
    """
    if not isinstance(val, (int, float)):
        return ''
    if val > 20:
        return 'background-color: #c6efce; color: #006100'
    elif val > 10:
        return 'background-color: #ffeb9c; color: #9c5700'
    else:
        return ''

def highlight_risk(val):
    """
    Highlight risk metrics based on value
    """
    if not isinstance(val, (int, float)):
        return ''
    if val > 70:
        return 'background-color: #ffc7ce; color: #9c0006'
    elif val > 40:
        return 'background-color: #ffeb9c; color: #9c5700'
    else:
        return 'background-color: #c6efce; color: #006100'

# Header with title and description
st.markdown('<h1 class="main-header">Solana Liquidity Pool Analysis and Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Insights for Optimal Liquidity Provision</p>', unsafe_allow_html=True)

# Dashboard introduction
st.markdown("""
This dashboard provides real-time analysis and AI-powered predictions for Solana blockchain liquidity pools.
Monitor key performance metrics, analyze historical trends, and leverage machine learning predictions
to optimize your liquidity provision strategy.
""")

# Key Metrics Overview Section
st.markdown('<div class="dashboard-section"><h2>🔑 Key Market Overview</h2></div>', unsafe_allow_html=True)

# Create columns for key metrics
col1, col2, col3, col4 = st.columns(4)

# Try to get real metrics, use default values on error
try:
    # Get top pools by various metrics
    top_by_liquidity = db.get_top_pools_by_liquidity(limit=1)
    top_by_apr = db.get_top_pools_by_apr(limit=1)
    avg_metrics = db.get_avg_metrics_by_day(days=1)
    total_liquidity = sum([pool[2] for pool in top_by_liquidity]) if top_by_liquidity else 0
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">Total Tracked Liquidity</p>', unsafe_allow_html=True)
        st.markdown(f'<h2>{format_number(total_liquidity)}</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">Highest APR Pool</p>', unsafe_allow_html=True)
        if top_by_apr and len(top_by_apr) > 0:
            pool_name = top_by_apr[0][1]
            pool_apr = top_by_apr[0][3]
            st.markdown(f'<h2>{format_percentage(pool_apr)}</h2>', unsafe_allow_html=True)
            st.markdown(f'<p>{pool_name}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<h2>N/A</h2><p>No data available</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">Avg. Market APR</p>', unsafe_allow_html=True)
        avg_apr = avg_metrics[0][3] if avg_metrics and len(avg_metrics) > 0 else None
        st.markdown(f'<h2>{format_percentage(avg_apr)}</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">24h Trading Volume</p>', unsafe_allow_html=True)
        avg_volume = avg_metrics[0][2] if avg_metrics and len(avg_metrics) > 0 else None
        st.markdown(f'<h2>{format_number(avg_volume)}</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading key metrics: {e}")
    # Display placeholder metrics
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">Total Tracked Liquidity</p>', unsafe_allow_html=True)
        st.markdown('<h2>$1.25B</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">Highest APR Pool</p>', unsafe_allow_html=True)
        st.markdown('<h2>24.5%</h2><p>SOL/USDT</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">Avg. Market APR</p>', unsafe_allow_html=True)
        st.markdown('<h2>8.7%</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">24h Trading Volume</p>', unsafe_allow_html=True)
        st.markdown('<h2>$425M</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Top Pools Section
st.markdown('<div class="dashboard-section"><h2>🏆 Top Performing Pools</h2></div>', unsafe_allow_html=True)

# Tabs for different pool rankings
tab1, tab2, tab3 = st.tabs(["By APR", "By Liquidity", "By Volume"])

with tab1:
    try:
        top_pools_apr = db.get_top_pools_by_apr(limit=10)
        if top_pools_apr and len(top_pools_apr) > 0:
            df_apr = pd.DataFrame(top_pools_apr, columns=['Pool ID', 'Name', 'Liquidity', 'APR', 'Volume 24h', 'Timestamp'])
            df_apr['APR'] = df_apr['APR'].apply(lambda x: f"{x:.2f}%")
            df_apr['Liquidity'] = df_apr['Liquidity'].apply(format_number)
            df_apr['Volume 24h'] = df_apr['Volume 24h'].apply(format_number)
            st.dataframe(df_apr.drop(columns=['Pool ID', 'Timestamp']), use_container_width=True, hide_index=True)
        else:
            st.info("No data available for APR rankings")
    except Exception as e:
        st.error(f"Error loading APR rankings: {e}")
        # Display sample data
        sample_data = [
            ["SOL/USDT", "$45.2M", "24.5%", "$12.3M"],
            ["RAY/SOL", "$22.1M", "18.7%", "$8.5M"],
            ["ORCA/USDC", "$18.5M", "16.3%", "$7.2M"],
            ["mSOL/SOL", "$53.8M", "14.9%", "$9.1M"],
            ["USDC/USDT", "$152.3M", "12.2%", "$32.5M"]
        ]
        sample_df = pd.DataFrame(sample_data, columns=['Name', 'Liquidity', 'APR', 'Volume 24h'])
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

with tab2:
    try:
        top_pools_liq = db.get_top_pools_by_liquidity(limit=10)
        if top_pools_liq and len(top_pools_liq) > 0:
            df_liq = pd.DataFrame(top_pools_liq, columns=['Pool ID', 'Name', 'Liquidity', 'APR', 'Volume 24h', 'Timestamp'])
            df_liq['APR'] = df_liq['APR'].apply(lambda x: f"{x:.2f}%")
            df_liq['Liquidity'] = df_liq['Liquidity'].apply(format_number)
            df_liq['Volume 24h'] = df_liq['Volume 24h'].apply(format_number)
            st.dataframe(df_liq.drop(columns=['Pool ID', 'Timestamp']), use_container_width=True, hide_index=True)
        else:
            st.info("No data available for liquidity rankings")
    except Exception as e:
        st.error(f"Error loading liquidity rankings: {e}")
        # Display sample data
        sample_data = [
            ["USDC/USDT", "$152.3M", "12.2%", "$32.5M"],
            ["ETH/SOL", "$98.7M", "8.5%", "$18.3M"],
            ["mSOL/SOL", "$53.8M", "14.9%", "$9.1M"],
            ["SOL/USDT", "$45.2M", "24.5%", "$12.3M"],
            ["BTC/SOL", "$38.1M", "7.2%", "$8.7M"]
        ]
        sample_df = pd.DataFrame(sample_data, columns=['Name', 'Liquidity', 'APR', 'Volume 24h'])
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

with tab3:
    try:
        top_pools_vol = db.get_top_pools_by_volume(limit=10)
        if top_pools_vol and len(top_pools_vol) > 0:
            df_vol = pd.DataFrame(top_pools_vol, columns=['Pool ID', 'Name', 'Liquidity', 'APR', 'Volume 24h', 'Timestamp'])
            df_vol['APR'] = df_vol['APR'].apply(lambda x: f"{x:.2f}%")
            df_vol['Liquidity'] = df_vol['Liquidity'].apply(format_number)
            df_vol['Volume 24h'] = df_vol['Volume 24h'].apply(format_number)
            st.dataframe(df_vol.drop(columns=['Pool ID', 'Timestamp']), use_container_width=True, hide_index=True)
        else:
            st.info("No data available for volume rankings")
    except Exception as e:
        st.error(f"Error loading volume rankings: {e}")
        # Display sample data
        sample_data = [
            ["USDC/USDT", "$152.3M", "12.2%", "$32.5M"],
            ["SOL/USDT", "$45.2M", "24.5%", "$12.3M"],
            ["RAY/SOL", "$22.1M", "18.7%", "$8.5M"],
            ["ETH/SOL", "$98.7M", "8.5%", "$18.3M"],
            ["ORCA/USDC", "$18.5M", "16.3%", "$7.2M"]
        ]
        sample_df = pd.DataFrame(sample_data, columns=['Name', 'Liquidity', 'APR', 'Volume 24h'])
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

# Market Trends Section
st.markdown('<div class="dashboard-section"><h2>📈 Market Trends</h2></div>', unsafe_allow_html=True)

# Market trends visualization
try:
    # Get average metrics for the past 30 days
    avg_metrics_30d = db.get_avg_metrics_by_day(days=30)
    
    if avg_metrics_30d and len(avg_metrics_30d) > 0:
        df_trends = pd.DataFrame(avg_metrics_30d, columns=['Date', 'Liquidity', 'Volume', 'APR'])
        
        # Create subplot with 2 y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=df_trends['Date'], y=df_trends['Liquidity'], name="Avg. Liquidity", 
                      line=dict(color='#1E88E5', width=2)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df_trends['Date'], y=df_trends['APR'], name="Avg. APR", 
                      line=dict(color='#FFC107', width=2, dash='dot')),
            secondary_y=True,
        )
        
        # Set titles
        fig.update_layout(
            title_text="30-Day Market Trends",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Average Liquidity (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Average APR (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create sample trend chart if no data
        dates = pd.date_range(end=datetime.now(), periods=30).tolist()
        liquidity = [random_float(950, 1050) * 1_000_000 for _ in range(30)]
        apr = [random_float(7, 15) for _ in range(30)]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=dates, y=liquidity, name="Avg. Liquidity", 
                      line=dict(color='#1E88E5', width=2)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=apr, name="Avg. APR", 
                      line=dict(color='#FFC107', width=2, dash='dot')),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="30-Day Market Trends (Sample Data)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
        )
        
        fig.update_yaxes(title_text="Average Liquidity (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Average APR (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error loading market trends: {e}")
    st.info("Unable to display market trends chart. Please check database connection.")

# AI Predictions Section
st.markdown('<div class="dashboard-section"><h2>🧠 AI-Powered Predictions</h2></div>', unsafe_allow_html=True)

# Get latest model predictions
try:
    predictions = db.get_latest_predictions(limit=5)
    
    if predictions and len(predictions) > 0:
        st.subheader("Recent Model Predictions")
        df_pred = pd.DataFrame(predictions, columns=[
            'Pool ID', 'Name', 'Predicted APR', 'Performance Class', 
            'Risk Score', 'Prediction Time', 'Model Version'
        ])
        
        # Format columns for better display
        df_pred['Predicted APR'] = df_pred['Predicted APR'].apply(lambda x: f"{x:.2f}%")
        df_pred['Risk Score'] = df_pred['Risk Score'].apply(lambda x: f"{x:.1f}")
        
        # Display styled dataframe
        st.dataframe(
            df_pred[['Name', 'Predicted APR', 'Performance Class', 'Risk Score']].style
            .applymap(highlight_performance, subset=['Predicted APR'])
            .applymap(highlight_risk, subset=['Risk Score']),
            use_container_width=True
        )
    else:
        st.info("No prediction data available. The ML models may need to be trained with more data.")
        
        # Display sample data
        st.subheader("Sample Predictions (for UI demonstration)")
        sample_data = [
            ["SOL/USDT", "25.3%", "High", "65.2"],
            ["RAY/SOL", "19.1%", "Medium", "48.7"],
            ["ETH/SOL", "8.9%", "Low", "22.4"],
            ["USDC/USDT", "12.5%", "Medium", "32.8"],
            ["mSOL/SOL", "16.2%", "Medium", "41.3"]
        ]
        sample_df = pd.DataFrame(sample_data, columns=['Name', 'Predicted APR', 'Performance Class', 'Risk Score'])
        
        # Display styled dataframe
        st.dataframe(
            sample_df.style
            .applymap(highlight_performance, subset=['Predicted APR'])
            .applymap(highlight_risk, subset=['Risk Score']),
            use_container_width=True
        )
except Exception as e:
    st.error(f"Error loading predictions: {e}")
    st.info("ML model predictions unavailable. Check the ML pipeline status.")

# System Status Section
st.markdown('<div class="dashboard-section"><h2>⚙️ System Status</h2></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Collection")
    
    # Check API connection
    api_client = RaydiumAPIClient()
    try:
        pools = api_client.get_all_pools()
        if pools:
            st.success(f"✅ API Connection: Connected (Found {len(pools)} pools)")
        else:
            st.warning("⚠️ API Connection: Connected but no data received")
    except Exception as e:
        st.error(f"❌ API Connection: Error ({str(e)[:100]}...)")
    
    # Check database connection
    try:
        if db.get_connection():
            st.success("✅ Database Connection: Connected")
        else:
            st.error("❌ Database Connection: Failed")
    except Exception as e:
        st.error(f"❌ Database Connection: Error ({str(e)[:100]}...)")
    
    # Last data collection time
    try:
        last_pool = db.query_to_dataframe("SELECT MAX(timestamp) FROM pool_data")
        if not last_pool.empty and last_pool.iloc[0, 0] is not None:
            last_time = last_pool.iloc[0, 0]
            time_diff = datetime.now() - last_time
            hours_ago = time_diff.total_seconds() / 3600
            
            if hours_ago < 1:
                st.success(f"✅ Last Data Collection: {int(hours_ago * 60)} minutes ago")
            elif hours_ago < 24:
                st.success(f"✅ Last Data Collection: {int(hours_ago)} hours ago")
            else:
                st.warning(f"⚠️ Last Data Collection: {int(hours_ago / 24)} days ago")
        else:
            st.warning("⚠️ Last Data Collection: No data found")
    except Exception as e:
        st.error(f"❌ Last Data Collection Error: {str(e)[:100]}...")

with col2:
    st.subheader("ML System")
    
    # Check for ML model files
    model_path = config.MODEL_STORAGE_PATH
    try:
        if os.path.exists(model_path):
            models = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
            if models:
                st.success(f"✅ ML Models: {len(models)} models found")
                
                # Last model training time
                model_times = [os.path.getmtime(os.path.join(model_path, m)) for m in models]
                if model_times:
                    last_trained = datetime.fromtimestamp(max(model_times))
                    days_ago = (datetime.now() - last_trained).days
                    
                    if days_ago < 1:
                        st.success(f"✅ Last Model Training: Today")
                    elif days_ago < 7:
                        st.success(f"✅ Last Model Training: {days_ago} days ago")
                    else:
                        st.warning(f"⚠️ Last Model Training: {days_ago} days ago")
            else:
                st.warning("⚠️ ML Models: No models found")
        else:
            st.warning(f"⚠️ ML Models: Model directory not found at {model_path}")
    except Exception as e:
        st.error(f"❌ ML Models Error: {str(e)[:100]}...")
    
    # Prediction accuracy (could be calculated from validation metrics)
    try:
        # This would ideally come from a table storing model performance metrics
        st.info("ℹ️ Model Validation: Not available in this version")
    except Exception as e:
        st.error(f"❌ Model Validation Error: {str(e)[:100]}...")

# Footer with system information
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(f"Solana Liquidity Pool Analysis System | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Helper function for demo data
def random_float(min_val, max_val):
    import random
    return min_val + (max_val - min_val) * random.random()