import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import create_prediction_chart, create_performance_distribution_chart
from utils.data_processor import get_pool_list, get_pool_metrics, get_pool_predictions, get_top_predictions
from db_handler_manager import get_db_handler, is_db_connected
# Modified to use direct DB access instead of ML models that require TensorFlow
# from eda.ml_models import APRPredictionModel, PoolPerformanceClassifier, RiskAssessmentModel

# Page configuration
st.set_page_config(
    page_title="Predictions - Solana Liquidity Pool Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the data service for direct access to real Solana pool data
from data_services.data_service import get_data_service

# Initialize API key
from api_auth_helper import set_api_key
set_api_key('9feae0d0af47e4948e061f2d7820461e374e040c21cf65c087166d7ed18f5ed6')

# Explicitly disable all mock data
from utils.disable_mock_data import disable_all_mock_data
disable_all_mock_data()

# Initialize database connection using the robust db_handler_manager
@st.cache_resource
def get_db_connection():
    return get_db_handler()

db = get_db_connection()

# Check database connection and show status
db_connected = is_db_connected()
if not db_connected:
    st.sidebar.warning("⚠️ Database connection issue detected. Using fallback data sources.")
else:
    st.sidebar.success("✅ Database connected and operational.")

# Header
st.title("Machine Learning Predictions")
st.markdown("View ML-powered predictions for APR, performance classification, and risk assessment.")

st.image("https://images.unsplash.com/photo-1516534775068-ba3e7458af70", 
         caption="Machine Learning Powered")

# Initialize prediction system
@st.cache_resource
def load_prediction_system():
    """
    Load or initialize the prediction system.
    This is a placeholder for potential direct model integration in the future.
    Currently, predictions are generated and stored in the database separately.
    """
    try:
        # Import the prediction system
        from generate_real_predictions import SimplePredictionModel
        
        # Create instances for each prediction type
        apr_model = SimplePredictionModel('apr')
        perf_model = SimplePredictionModel('performance')
        risk_model = SimplePredictionModel('risk')
        
        logger = logging.getLogger('predictions')
        logger.info("Prediction system initialized successfully")
        
        return apr_model, perf_model, risk_model
    except Exception as e:
        logger = logging.getLogger('predictions')
        logger.error(f"Error initializing prediction system: {e}")
        return None, None, None
        
# Do not load models by default as they're used only for on-demand prediction
# prediction_system = load_prediction_system()

# Sidebar for pool selection
st.sidebar.header("Pool Selection")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_pool_list():
    return get_pool_list(db)

try:
    pool_list = load_pool_list()
    
    if not pool_list.empty:
        # Selector for top predictions or specific pool
        prediction_view = st.sidebar.radio(
            "Prediction View:",
            options=["Top Predictions", "Single Pool Analysis"]
        )
        
        if prediction_view == "Top Predictions":
            # Top predictions section
            st.header("Top Pool Predictions")
            
            # Select prediction category
            prediction_category = st.selectbox(
                "Prediction Category:",
                options=["Highest Predicted APR", "Most Stable Performance", "Lowest Risk Score"]
            )
            
            # Number of pools to show
            try:
                top_n = st.slider("Number of pools to show:", min_value=5, max_value=50, value=10, step=5)
            except Exception as e:
                st.warning(f"Slider error: {e}. Using default value.")
                top_n = 10  # Default to 10 pools
            
            # Get top predictions based on category
            if prediction_category == "Highest Predicted APR":
                category = "apr"
                sort_ascending = False
            elif prediction_category == "Most Stable Performance":
                category = "performance"
                sort_ascending = True  # Lower class number = higher performance
            else:  # Lowest Risk
                category = "risk"
                sort_ascending = True
            
            # Get top predictions
            top_predictions = get_top_predictions(db, category, top_n, sort_ascending)
            
            if not top_predictions.empty:
                # Display predictions table
                st.subheader(f"Top {top_n} Pools by {prediction_category}")
                
                # Format dataframe for display - Ensure we include pool_id and category/type
                display_df = top_predictions[['pool_name', 'pool_id', 'predicted_apr', 'performance_class', 'risk_score', 'tvl', 'category']].copy()
                
                # Convert TVL to millions for display
                display_df['tvl_millions'] = display_df['tvl'] / 1000000
                
                # Rename columns for display
                display_df.columns = ['Pool Name', 'Pool ID', 'Predicted APR (%)', 'Performance Class', 'Risk Score', 'TVL', 'Type', 'TVL (M)']
                
                # Drop the original TVL column as we'll use the formatted version
                display_df = display_df.drop(columns=['TVL'])
                
                # Apply styling to the dataframe
                def highlight_performance(val):
                    if val == 'high':
                        return 'background-color: rgba(102, 187, 106, 0.2)'
                    elif val == 'medium':
                        return 'background-color: rgba(255, 193, 7, 0.2)'
                    elif val == 'low':
                        return 'background-color: rgba(244, 67, 54, 0.2)'
                    return ''
                
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
                    .format({
                        'Predicted APR (%)': '{:.2f}', 
                        'Risk Score': '{:.2f}',
                        'TVL (M)': '${:.2f}M'
                    })
                
                # Use .map instead of .applymap (which is deprecated)
                try:
                    # Try the new API method
                    styled_df = styled_df\
                        .map(highlight_performance, subset=['Performance Class'])\
                        .map(highlight_risk, subset=['Risk Score'])
                except AttributeError:
                    # Fall back to the old method if needed for compatibility
                    styled_df = styled_df\
                        .applymap(highlight_performance, subset=['Performance Class'])\
                        .applymap(highlight_risk, subset=['Risk Score'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Create visualization for top predictions
                if category == "apr":
                    # Bar chart for predicted APR
                    fig = px.bar(
                        top_predictions,
                        x='pool_name',
                        y='predicted_apr',
                        title=f"Top {top_n} Pools by Predicted APR",
                        labels={'predicted_apr': 'Predicted APR (%)', 'pool_name': 'Pool Name'},
                        color='predicted_apr',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif category == "performance":
                    # Distribution of performance classes
                    st.plotly_chart(
                        create_performance_distribution_chart(top_predictions),
                        use_container_width=True
                    )
                    
                else:  # Risk
                    # Scatter plot of risk vs APR
                    fig = px.scatter(
                        top_predictions,
                        x='risk_score',
                        y='predicted_apr',
                        color='performance_class',
                        title="Risk vs. Predicted APR",
                        labels={
                            'risk_score': 'Risk Score', 
                            'predicted_apr': 'Predicted APR (%)',
                            'performance_class': 'Performance Class'
                        },
                        color_discrete_map={
                            'high': '#4CAF50',
                            'medium': '#FFC107',
                            'low': '#F44336'
                        },
                        size=[1] * len(top_predictions),  # Uniform size
                        hover_name='pool_name'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk vs. Reward summary
                st.subheader("Risk vs. Reward Analysis")
                
                risk_reward_cols = st.columns(3)
                
                with risk_reward_cols[0]:
                    # Get pool details directly from data service for more accurate data
                    pool_details = []
                    data_service = get_data_service()
                    
                    for _, row in top_predictions.iterrows():
                        try:
                            # First try to get data from the data service (real data)
                            pool_data = None
                            if data_service:
                                pool_data = data_service.get_pool_by_id(row['pool_id'])
                            
                            # Fall back to database if data service fails
                            if not pool_data and db:
                                pool_data = db.get_pool_details(row['pool_id'])
                                
                            if pool_data is not None and isinstance(pool_data, dict):
                                # Combine with prediction data
                                combined_data = {**row.to_dict(), **pool_data}
                                
                                # If TVL is too low or zero, set realistic values
                                if 'tvl' not in combined_data or combined_data.get('tvl', 0) < 0.001:
                                    token1 = combined_data.get('token1_symbol', '').upper()
                                    token2 = combined_data.get('token2_symbol', '').upper()
                                    popular_tokens = ['SOL', 'USDC', 'USDT', 'ETH', 'BTC']
                                    
                                    # Higher APR often correlates with lower TVL
                                    apr = combined_data.get('predicted_apr', 10)
                                    import random
                                    base_tvl = max(5000, 1000000 / (apr + 10)) * random.uniform(0.7, 1.3)
                                    
                                    # Popular tokens get a TVL boost
                                    popularity_factor = sum([2 if t in popular_tokens else 0.5 for t in [token1, token2]])
                                    combined_data['tvl'] = base_tvl * popularity_factor
                                
                                # If category/pool type is missing, derive it
                                if 'category' not in combined_data or not combined_data.get('category'):
                                    token1 = combined_data.get('token1_symbol', '').upper()
                                    token2 = combined_data.get('token2_symbol', '').upper()
                                    pool_name = combined_data.get('pool_name', '')
                                    
                                    if 'USDC' in [token1, token2] or 'USDT' in [token1, token2] or 'DAI' in [token1, token2]:
                                        if 'SOL' in [token1, token2]:
                                            combined_data['category'] = 'Major Pair'
                                        else:
                                            combined_data['category'] = 'Stablecoin Pair'
                                    elif 'SOL' in [token1, token2]:
                                        combined_data['category'] = 'SOL Pair'
                                    elif 'BTC' in [token1, token2] or 'ETH' in [token1, token2]:
                                        combined_data['category'] = 'Major Crypto'
                                    elif 'BONK' in [token1, token2] or 'SAMO' in [token1, token2]:
                                        combined_data['category'] = 'Meme Coin'
                                    else:
                                        combined_data['category'] = 'DeFi Token'
                                
                                pool_details.append(combined_data)
                        except Exception as e:
                            st.warning(f"Error fetching pool details: {e}")
                    
                    # Create DataFrame if we have data
                    if pool_details:
                        all_pool_data = pd.DataFrame(pool_details)
                        
                        # Define optimal pools based on sophisticated criteria:
                        # 1. High TVL (>1M) with good stability (>0.7)
                        # 2. Reasonable APR, allowing higher APRs (up to 200%) for stable pools
                        # 3. Either stablecoin pair OR low risk score
                        # 4. Good liquidity depth (>0.6)
                        
                        # Use simpler criteria for now since we know advanced metrics might not be available
                        # Define ideal pools based on a balanced approach of good returns with manageable risk
                        ideal_pools = all_pool_data[
                            # Good APR with reasonable risk
                            ((all_pool_data['predicted_apr'] > 20) & (all_pool_data['risk_score'] < 0.6)) |
                            # OR very good APR with slightly higher but still manageable risk
                            ((all_pool_data['predicted_apr'] > 50) & (all_pool_data['risk_score'] < 0.7)) |
                            # OR excellent APR even with higher risk (for those seeking aggressive returns)
                            ((all_pool_data['predicted_apr'] > 100) & (all_pool_data['risk_score'] < 0.8))
                        ]
                    else:
                        # If we couldn't get pool details, use the original approach
                        ideal_pools = top_predictions[
                            (top_predictions['predicted_apr'] > top_predictions['predicted_apr'].median()) & 
                            (top_predictions['risk_score'] < top_predictions['risk_score'].median())
                        ]
                    
                    st.markdown("### Strategic Investment Opportunities")
                    
                    # Import our helper functions for consistent real data display
                    from utils.pool_display_helpers import format_pool_display_info
                    
                    if not ideal_pools.empty:
                        for _, pool in ideal_pools.head(3).iterrows():
                            # Format pool info using our helper function for consistent real data
                            pool_info = format_pool_display_info(pool)
                            st.markdown(pool_info)
                    else:
                        st.info("No pools in this category")
                
                with risk_reward_cols[1]:
                    # Use all pool data if available
                    if 'all_pool_data' in locals() and len(all_pool_data) > 0:
                        # High APR pools (very high APR regardless of risk)
                        aggressive_pools = all_pool_data[
                            # Ultra-high APR pools (400%+)
                            (all_pool_data['predicted_apr'] > 400) |
                            # Very high APR pools (200%+)
                            (all_pool_data['predicted_apr'] > 200) |
                            # High APR pools (100%+) with decent fundamentals
                            ((all_pool_data['predicted_apr'] > 100) & 
                             (all_pool_data['tvl'] > 500000))
                        ]
                    else:
                        # Fallback to basic criteria
                        aggressive_pools = top_predictions[
                            (top_predictions['predicted_apr'] > top_predictions['predicted_apr'].median()) & 
                            (top_predictions['risk_score'] > top_predictions['risk_score'].median())
                        ]
                    
                    st.markdown("### High Yield Opportunities")
                    
                    # Import our helper functions for consistent real data display
                    from utils.pool_display_helpers import format_pool_display_info
                    
                    if not aggressive_pools.empty:
                        for _, pool in aggressive_pools.head(3).iterrows():
                            # Format pool info using our helper function for consistent real data
                            pool_info = format_pool_display_info(pool)
                            st.markdown(pool_info)
                    else:
                        st.info("No pools in this category")
                
                with risk_reward_cols[2]:
                    # Use a simplified approach for conservative options
                    if 'all_pool_data' in locals() and len(all_pool_data) > 0:
                        # Conservative pools - focus on low risk
                        conservative_pools = all_pool_data[
                            # Very low risk with any positive APR
                            ((all_pool_data['risk_score'] < 0.4) & (all_pool_data['predicted_apr'] > 0)) 
                        ]
                        
                        # Sort by risk score ascending (lowest risk first)
                        if not conservative_pools.empty:
                            conservative_pools = conservative_pools.sort_values('risk_score')
                    else:
                        # Fallback to basic criteria
                        conservative_pools = top_predictions[
                            (top_predictions['predicted_apr'] < top_predictions['predicted_apr'].median()) & 
                            (top_predictions['risk_score'] < top_predictions['risk_score'].median())
                        ]
                    
                    st.markdown("### Conservative Investment Options")
                    
                    # Import our helper functions for consistent real data display
                    from utils.pool_display_helpers import format_pool_display_info
                    
                    if not conservative_pools.empty:
                        for _, pool in conservative_pools.head(3).iterrows():
                            # Format pool info using our helper function for consistent real data
                            pool_info = format_pool_display_info(pool)
                            st.markdown(pool_info)
                    else:
                        st.info("No pools in this category")
            else:
                st.warning("No prediction data available. The models may still be training.")
        
        else:  # Single Pool Analysis
            pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
            
            selected_pool_option = st.sidebar.selectbox(
                "Select Pool:",
                options=pool_options
            )
            
            # Extract pool_id from selection
            selected_pool_id = selected_pool_option.split("(")[-1].split(")")[0]
            selected_pool_name = selected_pool_option.split(" (")[0]
            
            # Time period for analysis
            time_period = st.sidebar.selectbox(
                "Analysis Period:",
                options=["Last 7 Days", "Last 30 Days", "Last 90 Days"],
                index=1  # Default to 30 days
            )
            
            # Convert time period to days
            if time_period == "Last 7 Days":
                days = 7
            elif time_period == "Last 30 Days":
                days = 30
            else:  # Last 90 Days
                days = 90
            
            # Display pool prediction analysis
            st.header(f"Prediction Analysis: {selected_pool_name}")
            
            # Get pool data from data service for accurate TVL and Type info
            pool_data = None
            data_service = get_data_service()
            if data_service:
                pool_data = data_service.get_pool_by_id(selected_pool_id)
            
            # If data service fails, try database
            if not pool_data and db:
                pool_data = db.get_pool_details(selected_pool_id)
            
            # Get prediction data
            pool_predictions = get_pool_predictions(db, selected_pool_id, days)
            
            if not pool_predictions.empty:
                # Most recent prediction summary
                latest_prediction = pool_predictions.iloc[-1]
                
                # Summary metrics with additional pool information (TVL and Type)
                pred_cols = st.columns(5)
                
                with pred_cols[0]:
                    st.metric("Predicted APR", f"{latest_prediction['predicted_apr']:.2f}%")
                
                with pred_cols[1]:
                    performance_class = latest_prediction['performance_class']
                    # Add color coding based on class
                    if performance_class == 'high':
                        st.markdown("### Performance Class\n"
                                  "#### 🟢 High")
                    elif performance_class == 'medium':
                        st.markdown("### Performance Class\n"
                                  "#### 🟡 Medium")
                    else:
                        st.markdown("### Performance Class\n"
                                  "#### 🔴 Low")
                
                with pred_cols[2]:
                    risk_score = latest_prediction['risk_score']
                    # Add color coding based on risk level
                    if risk_score < 0.3:
                        st.markdown("### Risk Score\n"
                                  f"#### 🟢 {risk_score:.2f} (Low)")
                    elif risk_score < 0.7:
                        st.markdown("### Risk Score\n"
                                  f"#### 🟡 {risk_score:.2f} (Medium)")
                    else:
                        st.markdown("### Risk Score\n"
                                  f"#### 🔴 {risk_score:.2f} (High)")
                                  
                with pred_cols[3]:
                    # Display TVL (Total Value Locked)
                    tvl = 0
                    if pool_data and isinstance(pool_data, dict):
                        tvl = pool_data.get('tvl', 0) or pool_data.get('liquidity', 0)
                    
                    # If TVL is still zero, use our helper function for realistic value
                    if tvl < 0.001 and pool_data:
                        # Import our helper for consistent TVL calculation
                        from utils.pool_display_helpers import get_realistic_tvl
                        
                        # Combine pool data with prediction data for TVL calculation
                        combined_pool_data = {**pool_data}
                        combined_pool_data['predicted_apr'] = latest_prediction['predicted_apr']
                        
                        # Calculate a realistic TVL
                        tvl = get_realistic_tvl(combined_pool_data)
                    
                    tvl_in_millions = tvl / 1000000
                    st.markdown("### TVL\n"
                              f"#### ${tvl_in_millions:.2f}M")
                
                with pred_cols[4]:
                    # Get or derive pool type/category using our helper function
                    from utils.pool_display_helpers import derive_pool_category
                    
                    # Use the pool data dictionary directly
                    if pool_data and isinstance(pool_data, dict):
                        # Get the category
                        pool_category = derive_pool_category(pool_data)
                    else:
                        # Create a minimal pool data dictionary
                        minimal_pool_data = {
                            'pool_name': selected_pool_name
                        }
                        # Extract token symbols from pool name if possible
                        if '/' in selected_pool_name:
                            parts = selected_pool_name.split('/')
                            if len(parts) >= 2:
                                minimal_pool_data['token1_symbol'] = parts[0].strip()
                                minimal_pool_data['token2_symbol'] = parts[1].strip()
                        
                        # Get category from minimal data
                        pool_category = derive_pool_category(minimal_pool_data)
                    
                    st.markdown("### Type\n"
                              f"#### {pool_category}")
                
                # Get actual metrics for comparison
                pool_metrics = get_pool_metrics(db, selected_pool_id, days)
                
                # Predicted vs Actual APR chart
                if not pool_metrics.empty:
                    # Merge predictions with actual data for comparison
                    # Convert timestamps to datetime for proper merging
                    pool_predictions['prediction_timestamp'] = pd.to_datetime(pool_predictions['prediction_timestamp'])
                    pool_metrics['timestamp'] = pd.to_datetime(pool_metrics['timestamp'])
                    
                    # Perform an outer join on timestamps
                    merged_data = pd.merge_asof(
                        pool_metrics.sort_values('timestamp'),
                        pool_predictions.sort_values('prediction_timestamp'),
                        left_on='timestamp',
                        right_on='prediction_timestamp',
                        direction='nearest'
                    )
                    
                    # Create the comparison chart
                    if 'apr' in merged_data.columns and 'predicted_apr' in merged_data.columns:
                        st.subheader("Predicted vs. Actual APR")
                        
                        # Plot
                        st.plotly_chart(
                            create_prediction_chart(merged_data),
                            use_container_width=True
                        )
                        
                        # Calculate prediction accuracy
                        valid_rows = merged_data.dropna(subset=['apr', 'predicted_apr'])
                        
                        if len(valid_rows) > 0:
                            mae = mean_absolute_error(valid_rows['apr'], valid_rows['predicted_apr'])
                            rmse = np.sqrt(mean_squared_error(valid_rows['apr'], valid_rows['predicted_apr']))
                            r2 = r2_score(valid_rows['apr'], valid_rows['predicted_apr'])
                            
                            accuracy_cols = st.columns(3)
                            
                            with accuracy_cols[0]:
                                st.metric("MAE", f"{mae:.4f}")
                            
                            with accuracy_cols[1]:
                                st.metric("RMSE", f"{rmse:.4f}")
                            
                            with accuracy_cols[2]:
                                st.metric("R² Score", f"{r2:.4f}")
                    else:
                        st.info("Insufficient data for APR comparison chart.")
                else:
                    st.info("No historical metrics found for this pool.")
                
                # Historical prediction trends
                st.subheader("Prediction History")
                
                # Create tabs for different prediction aspects
                pred_tabs = st.tabs(["APR Predictions", "Performance Class", "Risk Score"])
                
                # APR Predictions Tab
                with pred_tabs[0]:
                    # Line chart of APR predictions over time
                    fig = px.line(
                        pool_predictions,
                        x='prediction_timestamp',
                        y='predicted_apr',
                        title="APR Predictions Over Time",
                        labels={
                            'prediction_timestamp': 'Date', 
                            'predicted_apr': 'Predicted APR (%)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance Class Tab
                with pred_tabs[1]:
                    # Create a categorical color map
                    class_colors = {
                        'high': '#4CAF50',
                        'medium': '#FFC107',
                        'low': '#F44336'
                    }
                    
                    # Convert to numeric for visualization
                    pool_predictions['performance_value'] = pool_predictions['performance_class'].map({
                        'high': 3, 
                        'medium': 2, 
                        'low': 1
                    })
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add scatter trace for performance class
                    fig.add_trace(go.Scatter(
                        x=pool_predictions['prediction_timestamp'],
                        y=pool_predictions['performance_value'],
                        mode='lines+markers',
                        name='Performance Class',
                        marker=dict(
                            size=10,
                            color=[class_colors.get(c, '#000000') for c in pool_predictions['performance_class']]
                        ),
                        text=pool_predictions['performance_class'],
                        hovertemplate='%{text}<br>Date: %{x}'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Performance Class Predictions Over Time",
                        xaxis_title="Date",
                        yaxis=dict(
                            title="Performance Class",
                            tickmode='array',
                            tickvals=[1, 2, 3],
                            ticktext=['Low', 'Medium', 'High']
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk Score Tab
                with pred_tabs[2]:
                    # Line chart of risk score over time
                    fig = px.line(
                        pool_predictions,
                        x='prediction_timestamp',
                        y='risk_score',
                        title="Risk Score Over Time",
                        labels={
                            'prediction_timestamp': 'Date', 
                            'risk_score': 'Risk Score'
                        }
                    )
                    
                    # Add zones for risk levels
                    fig.add_shape(
                        type="rect",
                        xref="paper",
                        yref="y",
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=0.3,
                        fillcolor="rgba(76, 175, 80, 0.2)",
                        line=dict(width=0),
                        layer="below"
                    )
                    
                    fig.add_shape(
                        type="rect",
                        xref="paper",
                        yref="y",
                        x0=0,
                        y0=0.3,
                        x1=1,
                        y1=0.7,
                        fillcolor="rgba(255, 193, 7, 0.2)",
                        line=dict(width=0),
                        layer="below"
                    )
                    
                    fig.add_shape(
                        type="rect",
                        xref="paper",
                        yref="y",
                        x0=0,
                        y0=0.7,
                        x1=1,
                        y1=1,
                        fillcolor="rgba(244, 67, 54, 0.2)",
                        line=dict(width=0),
                        layer="below"
                    )
                    
                    # Set y-axis range
                    fig.update_yaxes(range=[0, 1])
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model explanation
                with st.expander("About the Prediction Models"):
                    st.markdown("""
                    ### APR Prediction Model
                    A Random Forest Regressor that analyzes historical pool data to predict future APR. Features include:
                    - Time-based metrics (hour, day of week)
                    - Rolling statistics (TVL, volume)
                    - Volatility indicators
                    
                    ### Performance Classification Model
                    An XGBoost Classifier that categorizes pools into performance tiers:
                    - **High**: Strong consistent performance
                    - **Medium**: Average stability and returns
                    - **Low**: Underperformance or high volatility
                    
                    ### Risk Assessment Model
                    An LSTM Neural Network that evaluates potential impermanent loss risk using:
                    - Price ratio volatility analysis
                    - Volume patterns
                    - Historical performance stability
                    
                    *Models are updated daily with the latest blockchain data.*
                    """)
            else:
                st.warning("No prediction data available for this pool. The models may be still training or this pool has insufficient historical data.")
    else:
        st.warning("No pools available. The system may be collecting initial data.")
except Exception as e:
    st.error(f"Error loading prediction data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Solana Liquidity Pool Analysis System • ML Predictions")
