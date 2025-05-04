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
from database.db_operations import DBManager
# Modified to use direct DB access instead of ML models that require TensorFlow
# from eda.ml_models import APRPredictionModel, PoolPerformanceClassifier, RiskAssessmentModel

# Page configuration
st.set_page_config(
    page_title="Predictions - Solana Liquidity Pool Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection
@st.cache_resource
def get_db_connection():
    return DBManager()

db = get_db_connection()

# Header
st.title("Machine Learning Predictions")
st.markdown("View ML-powered predictions for APR, performance classification, and risk assessment.")

st.image("https://images.unsplash.com/photo-1516534775068-ba3e7458af70", 
         caption="Machine Learning Concept")

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
                
                # Format dataframe for display - Ensure we include pool_id 
                display_df = top_predictions[['pool_name', 'pool_id', 'predicted_apr', 'performance_class', 'risk_score']].copy()
                display_df.columns = ['Pool Name', 'Pool ID', 'Predicted APR (%)', 'Performance Class', 'Risk Score']
                
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
                    .format({'Predicted APR (%)': '{:.2f}', 'Risk Score': '{:.2f}'})
                
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
                    # Get pool details for additional metrics
                    pool_details = []
                    for _, row in top_predictions.iterrows():
                        try:
                            # Get additional pool data
                            pool_data = db.get_pool_details(row['pool_id'])
                            if pool_data is not None and isinstance(pool_data, dict):
                                # Combine with prediction data
                                combined_data = {**row.to_dict(), **pool_data}
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
                        
                        # First check if we have the required columns
                        required_cols = ['tvl', 'tvl_stability', 'category', 'liquidity_depth']
                        has_advanced_data = all(col in all_pool_data.columns for col in required_cols)
                        
                        if has_advanced_data:
                            # Advanced optimal pool criteria
                            ideal_pools = all_pool_data[
                                # High TVL with stability
                                ((all_pool_data['tvl'] > 2000000) & (all_pool_data['tvl_stability'] > 0.7)) |
                                # OR medium TVL but very stable
                                ((all_pool_data['tvl'] > 1000000) & (all_pool_data['tvl_stability'] > 0.85)) |
                                # OR medium TVL with stablecoin and good APR
                                ((all_pool_data['tvl'] > 1000000) & 
                                 (all_pool_data['category'].str.contains('stablecoin')) &
                                 (all_pool_data['predicted_apr'] > 15))
                            ]
                            
                            # Further filter by risk or liquidity depth
                            ideal_pools = ideal_pools[
                                # Either low risk
                                (ideal_pools['risk_score'] < 0.4) |
                                # OR stablecoin pair with acceptable risk
                                ((ideal_pools['category'].str.contains('stablecoin')) & 
                                 (ideal_pools['risk_score'] < 0.7)) |
                                # OR very high APR (100%+) with manageable risk
                                ((ideal_pools['predicted_apr'] > 100) & 
                                 (ideal_pools['risk_score'] < 0.8) &
                                 (ideal_pools['tvl_stability'] > 0.7)) |
                                # OR excellent liquidity depth
                                (ideal_pools['liquidity_depth'] > 0.8)
                            ]
                        else:
                            # Fallback to basic criteria if advanced data isn't available
                            ideal_pools = all_pool_data[
                                (all_pool_data['predicted_apr'] > all_pool_data['predicted_apr'].median()) & 
                                (all_pool_data['risk_score'] < all_pool_data['risk_score'].median())
                            ]
                    else:
                        # If we couldn't get pool details, use the original approach
                        ideal_pools = top_predictions[
                            (top_predictions['predicted_apr'] > top_predictions['predicted_apr'].median()) & 
                            (top_predictions['risk_score'] < top_predictions['risk_score'].median())
                        ]
                    
                    st.markdown("### Strategic Investment Opportunities")
                    if not ideal_pools.empty:
                        for _, pool in ideal_pools.head(3).iterrows():
                            # Check if we have detailed metrics to show
                            has_tvl = 'tvl' in pool and pool['tvl'] is not None
                            has_stability = 'tvl_stability' in pool and pool['tvl_stability'] is not None
                            has_category = 'category' in pool and pool['category'] is not None
                            
                            # Basic info about the pool with Pool ID
                            pool_info = f"**{pool['pool_name']}**  \n" \
                                      f"Pool ID: {pool['pool_id']}  \n" \
                                      f"APR: {pool['predicted_apr']:.2f}%  \n" \
                                      f"Risk: {pool['risk_score']:.2f}"
                            
                            # Add advanced metrics if available
                            if has_tvl:
                                pool_info += f"  \nTVL: ${pool['tvl']/1000000:.2f}M"
                            if has_stability:
                                pool_info += f"  \nStability: {pool['tvl_stability']*100:.0f}%"
                            if has_category:
                                pool_info += f"  \nType: {pool['category']}"
                                
                            st.markdown(pool_info)
                    else:
                        st.info("No pools in this category")
                
                with risk_reward_cols[1]:
                    # Use the advanced data if available
                    if 'all_pool_data' in locals() and 'has_advanced_data' in locals() and pool_details and has_advanced_data:
                        # High APR pools (very high APR regardless of risk, but with some stability)
                        aggressive_pools = all_pool_data[
                            # Ultra-high APR pools (400%+)
                            ((all_pool_data['predicted_apr'] > 400) & 
                             (all_pool_data['tvl_stability'] > 0.4)) |
                            # Very high APR pools (200%+) with some stability
                            ((all_pool_data['predicted_apr'] > 200) & 
                             (all_pool_data['tvl_stability'] > 0.5)) |
                            # High APR pools (100%+) with decent fundamentals
                            ((all_pool_data['predicted_apr'] > 100) & 
                             (all_pool_data['tvl'] > 500000) &
                             (all_pool_data['tvl_stability'] > 0.6))
                        ]
                    else:
                        # Fallback to basic criteria
                        aggressive_pools = top_predictions[
                            (top_predictions['predicted_apr'] > top_predictions['predicted_apr'].median()) & 
                            (top_predictions['risk_score'] > top_predictions['risk_score'].median())
                        ]
                    
                    st.markdown("### High Yield Opportunities")
                    if not aggressive_pools.empty:
                        for _, pool in aggressive_pools.head(3).iterrows():
                            # Check if we have detailed metrics to show
                            has_tvl = 'tvl' in pool and pool['tvl'] is not None
                            has_stability = 'tvl_stability' in pool and pool['tvl_stability'] is not None
                            
                            # Basic info about the pool with Pool ID
                            pool_info = f"**{pool['pool_name']}**  \n" \
                                      f"APR: {pool['predicted_apr']:.2f}%  \n" \
                                      f"Risk: {pool['risk_score']:.2f}"
                            
                            # Add advanced metrics if available
                            if has_tvl:
                                pool_info += f"  \nTVL: ${pool['tvl']/1000000:.2f}M"
                            if has_stability:
                                pool_info += f"  \nStability: {pool['tvl_stability']*100:.0f}%"
                                
                            st.markdown(pool_info)
                    else:
                        st.info("No pools in this category")
                
                with risk_reward_cols[2]:
                    # Use the advanced data if available
                    if 'all_pool_data' in locals() and 'has_advanced_data' in locals() and pool_details and has_advanced_data:
                        # Very low risk pools (focused on capital preservation)
                        conservative_pools = all_pool_data[
                            # Very low risk with decent APR
                            ((all_pool_data['risk_score'] < 0.3) & 
                             (all_pool_data['predicted_apr'] > 5)) |
                            # Stablecoin pairs with high TVL stability
                            ((all_pool_data['category'].str.contains('stablecoin')) & 
                             (all_pool_data['tvl'] > 1000000) &
                             (all_pool_data['tvl_stability'] > 0.8))
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
                    if not conservative_pools.empty:
                        for _, pool in conservative_pools.head(3).iterrows():
                            # Check if we have detailed metrics to show
                            has_tvl = 'tvl' in pool and pool['tvl'] is not None
                            has_stability = 'tvl_stability' in pool and pool['tvl_stability'] is not None
                            has_category = 'category' in pool and pool['category'] is not None
                            
                            # Basic info about the pool with Pool ID
                            pool_info = f"**{pool['pool_name']}**  \n" \
                                      f"APR: {pool['predicted_apr']:.2f}%  \n" \
                                      f"Risk: {pool['risk_score']:.2f}"
                            
                            # Add advanced metrics if available
                            if has_tvl:
                                pool_info += f"  \nTVL: ${pool['tvl']/1000000:.2f}M"
                            if has_stability:
                                pool_info += f"  \nStability: {pool['tvl_stability']*100:.0f}%"
                            if has_category:
                                pool_info += f"  \nType: {pool['category']}"
                                
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
            
            # Get prediction data
            pool_predictions = get_pool_predictions(db, selected_pool_id, days)
            
            if not pool_predictions.empty:
                # Most recent prediction summary
                latest_prediction = pool_predictions.iloc[-1]
                
                # Summary metrics
                pred_cols = st.columns(3)
                
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
