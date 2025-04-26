import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import create_risk_heat_map, create_impermanent_loss_chart
from utils.data_processor import get_pool_list, get_top_predictions, get_pool_metrics, get_token_prices
from database.db_operations import DBManager

# Page configuration
st.set_page_config(
    page_title="Risk Assessment - Solana Liquidity Pool Analysis",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection
@st.cache_resource
def get_db_connection():
    return DBManager()

db = get_db_connection()

# Header
st.title("Risk Assessment")
st.markdown("Advanced risk metrics and analysis for Solana liquidity pools.")

st.image("https://images.unsplash.com/photo-1472220625704-91e1462799b2", 
         caption="Risk Assessment Visualization")

# Get pool data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_pool_data():
    pool_list = get_pool_list(db)
    risk_predictions = get_top_predictions(db, "risk", 50, True)  # Sort by risk, ascending
    return pool_list, risk_predictions

try:
    pool_list, risk_predictions = load_pool_data()
    
    # Market overview section
    st.header("Market Risk Overview")
    
    if not risk_predictions.empty:
        # Risk distribution
        st.subheader("Pool Risk Distribution")
        
        # Calculate risk level counts
        risk_predictions['risk_level'] = pd.cut(
            risk_predictions['risk_score'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        risk_counts = risk_predictions['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        # Sort by risk level
        risk_level_order = {'Low': 0, 'Medium': 1, 'High': 2}
        risk_counts['sort_order'] = risk_counts['Risk Level'].map(risk_level_order)
        risk_counts = risk_counts.sort_values('sort_order').drop('sort_order', axis=1)
        
        # Create color map
        color_map = {'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'}
        
        # Create pie chart
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Risk Level',
            title="Distribution of Pool Risk Levels",
            color='Risk Level',
            color_discrete_map=color_map
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk heat map
        st.subheader("Risk vs. Reward Heat Map")
        
        # Create risk heat map
        st.plotly_chart(
            create_risk_heat_map(risk_predictions),
            use_container_width=True
        )
        
        # Top safe and risky pools
        risk_cols = st.columns(2)
        
        with risk_cols[0]:
            st.subheader("Lowest Risk Pools")
            
            lowest_risk = risk_predictions.sort_values('risk_score').head(5)
            
            for i, (_, pool) in enumerate(lowest_risk.iterrows()):
                # Color based on risk score
                if pool['risk_score'] < 0.3:
                    risk_color = "🟢"
                elif pool['risk_score'] < 0.7:
                    risk_color = "🟡"
                else:
                    risk_color = "🔴"
                    
                st.markdown(f"**{i+1}. {pool['pool_name']}**")
                st.markdown(f"Risk Score: {risk_color} {pool['risk_score']:.2f}")
                st.markdown(f"Predicted APR: {pool['predicted_apr']:.2f}%")
                st.markdown("---")
        
        with risk_cols[1]:
            st.subheader("Highest Risk Pools")
            
            highest_risk = risk_predictions.sort_values('risk_score', ascending=False).head(5)
            
            for i, (_, pool) in enumerate(highest_risk.iterrows()):
                # Color based on risk score
                if pool['risk_score'] < 0.3:
                    risk_color = "🟢"
                elif pool['risk_score'] < 0.7:
                    risk_color = "🟡"
                else:
                    risk_color = "🔴"
                    
                st.markdown(f"**{i+1}. {pool['pool_name']}**")
                st.markdown(f"Risk Score: {risk_color} {pool['risk_score']:.2f}")
                st.markdown(f"Predicted APR: {pool['predicted_apr']:.2f}%")
                st.markdown("---")
    else:
        st.warning("No risk prediction data available yet. The models may still be training.")
    
    # Impermanent loss calculator
    st.header("Impermanent Loss Calculator")
    
    il_cols = st.columns([3, 2])
    
    with il_cols[0]:
        st.markdown("""
        Impermanent Loss (IL) occurs when providing liquidity to a pool and the price ratio between the two tokens changes. 
        This calculator helps you understand the potential impact of price changes on your liquidity position.
        """)
        
        # Pool selection for IL calculation
        if not pool_list.empty:
            pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
            
            selected_pool_option = st.selectbox(
                "Select Pool:",
                options=pool_options,
                key="il_pool_selector"
            )
            
            # Extract pool info
            selected_pool_id = selected_pool_option.split("(")[-1].split(")")[0]
            selected_pool_name = selected_pool_option.split(" (")[0]
            
            # Extract token symbols
            if '/' in selected_pool_name:
                token_symbols = selected_pool_name.split('/')
                token_symbol1 = token_symbols[0].strip()
                token_symbol2 = token_symbols[1].strip()
                
                # Get token prices
                token_prices = get_token_prices(db, [token_symbol1, token_symbol2], 1)
                
                # Get the pool details to see if we have prices stored directly in the pool data
                pool_details = pool_list[pool_list['pool_id'] == selected_pool_id]
                
                initial_price1 = 0
                initial_price2 = 0
                
                # First try to get prices from the pool data (added by our token_price_service integration)
                if not pool_details.empty and 'token1_price' in pool_details.columns and 'token2_price' in pool_details.columns:
                    token1_price = pool_details.iloc[0].get('token1_price', 0)
                    token2_price = pool_details.iloc[0].get('token2_price', 0)
                    
                    if token1_price > 0:
                        initial_price1 = token1_price
                    
                    if token2_price > 0:
                        initial_price2 = token2_price
                
                # If we still don't have prices, try the database prices
                if initial_price1 == 0 or initial_price2 == 0:
                    if not token_prices.empty:
                        token1_prices = token_prices[token_prices['token_symbol'] == token_symbol1]
                        token2_prices = token_prices[token_prices['token_symbol'] == token_symbol2]
                        
                        if not token1_prices.empty and initial_price1 == 0:
                            initial_price1 = token1_prices.iloc[-1]['price_usd']
                        
                        if not token2_prices.empty and initial_price2 == 0:
                            initial_price2 = token2_prices.iloc[-1]['price_usd']
                
                # If we still don't have prices, try to get them directly from CoinGecko
                if initial_price1 == 0 or initial_price2 == 0:
                    try:
                        # Import here to avoid circular imports
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from token_price_service import get_token_price
                        
                        if initial_price1 == 0:
                            cg_price1 = get_token_price(token_symbol1)
                            if cg_price1 > 0:
                                initial_price1 = cg_price1
                        
                        if initial_price2 == 0:
                            cg_price2 = get_token_price(token_symbol2)
                            if cg_price2 > 0:
                                initial_price2 = cg_price2
                    except Exception as e:
                        st.warning(f"Could not get token prices from CoinGecko: {e}")
                
                # Input fields for token prices
                st.subheader("Initial Prices")
                
                price_cols = st.columns(2)
                
                with price_cols[0]:
                    initial_price1 = st.number_input(
                        f"{token_symbol1} Price (USD)",
                        min_value=0.0,
                        value=float(initial_price1) if initial_price1 > 0 else 1.0,
                        step=0.01
                    )
                
                with price_cols[1]:
                    initial_price2 = st.number_input(
                        f"{token_symbol2} Price (USD)",
                        min_value=0.0,
                        value=float(initial_price2) if initial_price2 > 0 else 1.0,
                        step=0.01
                    )
                
                # Ensure both prices are at least slightly different to avoid slider issues
                if initial_price1 == 0:
                    initial_price1 = 0.01
                if initial_price2 == 0:
                    initial_price2 = 0.01
                
                st.subheader("Price Change Scenario")
                
                # Price change sliders
                # First token slider
                try:
                    price_change1 = st.slider(
                        f"{token_symbol1} Price Change",
                        min_value=-90,
                        max_value=900,
                        value=0,
                        step=10,
                        format="%d%%"
                    )
                except Exception as e:
                    st.warning(f"Slider error for {token_symbol1}: {e}. Using default value.")
                    price_change1 = 0
                
                # Second token slider
                try:
                    price_change2 = st.slider(
                        f"{token_symbol2} Price Change",
                        min_value=-90,
                        max_value=900,
                        value=0,
                        step=10,
                        format="%d%%"
                    )
                except Exception as e:
                    st.warning(f"Slider error for {token_symbol2}: {e}. Using default value.")
                    price_change2 = 0
                
                # Calculate new prices
                new_price1 = initial_price1 * (1 + price_change1 / 100)
                new_price2 = initial_price2 * (1 + price_change2 / 100)
                
                # Calculate impermanent loss
                # Formula: IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
                # Where price_ratio = (new_price1/new_price2) / (initial_price1/initial_price2)
                
                initial_ratio = initial_price1 / initial_price2
                new_ratio = new_price1 / new_price2
                price_ratio = new_ratio / initial_ratio
                
                impermanent_loss = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
                impermanent_loss_pct = impermanent_loss * 100
                
                # Display results
                st.subheader("Impermanent Loss Results")
                
                result_cols = st.columns(3)
                
                with result_cols[0]:
                    st.metric(
                        "Impermanent Loss",
                        f"{impermanent_loss_pct:.2f}%",
                        delta=None
                    )
                
                with result_cols[1]:
                    st.metric(
                        f"New {token_symbol1} Price",
                        f"${new_price1:.2f}",
                        delta=f"{price_change1}%",
                        delta_color="normal"
                    )
                
                with result_cols[2]:
                    st.metric(
                        f"New {token_symbol2} Price",
                        f"${new_price2:.2f}",
                        delta=f"{price_change2}%",
                        delta_color="normal"
                    )
            else:
                st.warning("Selected pool name does not contain token information in the expected format (TOKEN1/TOKEN2).")
        else:
            st.warning("No pools available for impermanent loss calculation.")
    
    with il_cols[1]:
        # Initialize default price changes (will be overridden if sliders are used)
        price_change1_param = None
        price_change2_param = None
        
        # Check if we're in the context where price changes have been set via sliders
        if 'token_symbol1' in locals() and 'token_symbol2' in locals():
            if 'price_change1' in locals() and 'price_change2' in locals():
                # Only pass the price changes if valid values are available
                price_change1_param = price_change1
                price_change2_param = price_change2
        
        # Create the impermanent loss chart
        if price_change1_param is not None and price_change2_param is not None:
            st.plotly_chart(
                create_impermanent_loss_chart(
                    token1_change=price_change1_param,
                    token2_change=price_change2_param
                ),
                use_container_width=True
            )
        else:
            st.plotly_chart(
                create_impermanent_loss_chart(),
                use_container_width=True
            )
        
        # IL explanation
        st.markdown("""
        ### Understanding Impermanent Loss
        
        **What is it?**  
        Impermanent Loss is the difference between holding tokens and providing liquidity when prices change.
        
        **Why does it happen?**  
        Liquidity pools maintain a constant product formula (x * y = k), requiring the pool to rebalance as prices change.
        
        **When is it highest?**  
        IL is highest when the price ratio between tokens changes significantly in either direction.
        
        **How to mitigate?**  
        - Choose stable pairs
        - Look for pools with fee rewards that offset IL
        - Consider pools with IL protection mechanisms
        """)
    
    # Risk factors analysis
    st.header("Risk Factors Analysis")
    
    # Select pool for risk analysis
    if not pool_list.empty:
        pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
        
        selected_pool_option = st.selectbox(
            "Select Pool for Risk Analysis:",
            options=pool_options,
            key="risk_pool_selector"
        )
        
        # Extract pool info
        selected_pool_id = selected_pool_option.split("(")[-1].split(")")[0]
        selected_pool_name = selected_pool_option.split(" (")[0]
        
        # Get pool metrics
        pool_metrics = get_pool_metrics(db, selected_pool_id, 30)  # 30 days
        
        if not pool_metrics.empty:
            # Calculate risk metrics
            pool_metrics['timestamp'] = pd.to_datetime(pool_metrics['timestamp'])
            
            # 1. Liquidity volatility
            liquidity_std = pool_metrics['liquidity'].std()
            liquidity_mean = pool_metrics['liquidity'].mean()
            liquidity_volatility = liquidity_std / liquidity_mean if liquidity_mean > 0 else 0
            
            # 2. APR volatility
            apr_std = pool_metrics['apr'].std()
            apr_mean = pool_metrics['apr'].mean()
            apr_volatility = apr_std / apr_mean if apr_mean > 0 else 0
            
            # 3. Volume inconsistency
            volume_std = pool_metrics['volume'].std()
            volume_mean = pool_metrics['volume'].mean()
            volume_inconsistency = volume_std / volume_mean if volume_mean > 0 else 0
            
            # 4. Declining trends
            # Calculate 7-day rolling average for trend analysis
            pool_metrics['liquidity_7d_avg'] = pool_metrics['liquidity'].rolling(7).mean()
            pool_metrics['apr_7d_avg'] = pool_metrics['apr'].rolling(7).mean()
            
            # Check if current values are below the average (indicating decline)
            current_liquidity = pool_metrics['liquidity'].iloc[-1]
            current_apr = pool_metrics['apr'].iloc[-1]
            avg_liquidity = pool_metrics['liquidity_7d_avg'].iloc[-1]
            avg_apr = pool_metrics['apr_7d_avg'].iloc[-1]
            
            liquidity_declining = current_liquidity < avg_liquidity
            apr_declining = current_apr < avg_apr
            
            # Create risk factor visualizations
            st.subheader(f"Risk Factors: {selected_pool_name}")
            
            # Risk gauge visualization
            risk_factors = {
                "Liquidity Volatility": min(liquidity_volatility * 10, 1.0),
                "APR Volatility": min(apr_volatility * 5, 1.0),
                "Volume Inconsistency": min(volume_inconsistency * 8, 1.0),
                "Declining Trends": 0.7 if (liquidity_declining and apr_declining) else 0.3 if (liquidity_declining or apr_declining) else 0.0
            }
            
            # Display risk factors as gauges
            gauge_cols = st.columns(4)
            
            for i, (factor, value) in enumerate(risk_factors.items()):
                with gauge_cols[i]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value * 100,
                        title={"text": factor},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 33], "color": "green"},
                                {"range": [33, 67], "color": "yellow"},
                                {"range": [67, 100], "color": "red"}
                            ]
                        }
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time series charts for risk visualization
            st.subheader("Historical Risk Indicators")
            
            # Create tabs for different risk metrics
            risk_tabs = st.tabs(["Liquidity Stability", "APR Volatility", "Volume Analysis"])
            
            # Liquidity Stability Tab
            with risk_tabs[0]:
                # Calculate rolling std dev of liquidity (measure of instability)
                pool_metrics['liquidity_7d_std'] = pool_metrics['liquidity'].rolling(7).std()
                
                # Create dual-axis chart (liquidity and its volatility)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add liquidity line
                fig.add_trace(
                    go.Scatter(
                        x=pool_metrics['timestamp'],
                        y=pool_metrics['liquidity'],
                        name="Liquidity",
                        line=dict(color="#1f77b4")
                    ),
                    secondary_y=False
                )
                
                # Add volatility line
                fig.add_trace(
                    go.Scatter(
                        x=pool_metrics['timestamp'],
                        y=pool_metrics['liquidity_7d_std'],
                        name="7-Day Volatility",
                        line=dict(color="#ff7f0e", dash="dash")
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig.update_layout(
                    title="Liquidity and 7-Day Volatility",
                    xaxis_title="Date",
                    legend=dict(x=0, y=1, orientation="h")
                )
                
                fig.update_yaxes(title_text="Liquidity (USD)", secondary_y=False)
                fig.update_yaxes(title_text="Volatility", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # APR Volatility Tab
            with risk_tabs[1]:
                # Calculate rolling std dev of APR
                pool_metrics['apr_7d_std'] = pool_metrics['apr'].rolling(7).std()
                
                # Create dual-axis chart (APR and its volatility)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add APR line
                fig.add_trace(
                    go.Scatter(
                        x=pool_metrics['timestamp'],
                        y=pool_metrics['apr'],
                        name="APR",
                        line=dict(color="#2ca02c")
                    ),
                    secondary_y=False
                )
                
                # Add volatility line
                fig.add_trace(
                    go.Scatter(
                        x=pool_metrics['timestamp'],
                        y=pool_metrics['apr_7d_std'],
                        name="7-Day Volatility",
                        line=dict(color="#d62728", dash="dash")
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig.update_layout(
                    title="APR and 7-Day Volatility",
                    xaxis_title="Date",
                    legend=dict(x=0, y=1, orientation="h")
                )
                
                fig.update_yaxes(title_text="APR (%)", secondary_y=False)
                fig.update_yaxes(title_text="Volatility", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Volume Analysis Tab
            with risk_tabs[2]:
                # Create volume histogram
                fig = go.Figure()
                
                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=pool_metrics['timestamp'],
                        y=pool_metrics['volume'],
                        name="Volume",
                        marker_color="#9467bd"
                    )
                )
                
                # Add volume trend line (7-day moving average)
                fig.add_trace(
                    go.Scatter(
                        x=pool_metrics['timestamp'],
                        y=pool_metrics['volume'].rolling(7).mean(),
                        name="7-Day Moving Avg",
                        line=dict(color="#e377c2")
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title="Trading Volume History",
                    xaxis_title="Date",
                    yaxis_title="Volume (USD)",
                    legend=dict(x=0, y=1, orientation="h")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk summary
            st.subheader("Risk Assessment Summary")
            
            # Calculate overall risk score (weighted average of factors)
            overall_risk = (
                risk_factors["Liquidity Volatility"] * 0.3 +
                risk_factors["APR Volatility"] * 0.3 +
                risk_factors["Volume Inconsistency"] * 0.2 +
                risk_factors["Declining Trends"] * 0.2
            )
            
            # Risk category
            risk_category = "Low"
            risk_color = "green"
            
            if overall_risk > 0.7:
                risk_category = "High"
                risk_color = "red"
            elif overall_risk > 0.3:
                risk_category = "Medium"
                risk_color = "orange"
            
            # Display risk summary
            st.markdown(f"### Overall Risk: <span style='color:{risk_color}'>{risk_category} ({overall_risk:.2%})</span>", unsafe_allow_html=True)
            
            # Risk explanation
            st.markdown(f"""
            **Key findings for {selected_pool_name}:**
            
            - **Liquidity Stability**: {'Low' if risk_factors["Liquidity Volatility"] < 0.3 else 'Medium' if risk_factors["Liquidity Volatility"] < 0.7 else 'High'} volatility
            - **APR Consistency**: {'Stable' if risk_factors["APR Volatility"] < 0.3 else 'Moderately volatile' if risk_factors["APR Volatility"] < 0.7 else 'Highly volatile'}
            - **Volume Pattern**: {'Consistent' if risk_factors["Volume Inconsistency"] < 0.3 else 'Variable' if risk_factors["Volume Inconsistency"] < 0.7 else 'Erratic'}
            - **Trend Analysis**: {'Positive' if risk_factors["Declining Trends"] < 0.3 else 'Neutral' if risk_factors["Declining Trends"] < 0.7 else 'Negative'} trajectory
            
            **Recommendation**: {'Consider this pool for stable liquidity provision.' if overall_risk < 0.3 else 'Monitor this pool closely if providing liquidity.' if overall_risk < 0.7 else 'Exercise caution when providing liquidity to this pool.'}
            """)
        else:
            st.warning("No metrics data available for this pool. The system may not have collected enough historical data yet.")
    else:
        st.warning("No pools available for risk analysis.")
except Exception as e:
    st.error(f"Error loading risk assessment data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Solana Liquidity Pool Analysis System • Risk Assessment")
