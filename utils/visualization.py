import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random

def create_metrics_chart(df, metric, metric_label, title):
    """Create a line chart for a specific metric"""
    fig = px.line(
        df,
        x='timestamp',
        y=metric,
        title=title,
        labels={
            'timestamp': 'Date',
            metric: metric_label
        }
    )
    
    # Add marker points
    fig.update_traces(mode='lines+markers')
    
    # Format layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=metric_label,
        height=400
    )
    
    return fig

def create_liquidity_volume_chart(df):
    """Create a dual-axis chart for liquidity and volume"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['liquidity'],
            name="Liquidity",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            line=dict(color='green', width=2)
        ),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text="Liquidity and Volume Over Time",
        height=400
    )
    
    # Set axes titles
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Liquidity (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume (USD)", secondary_y=True)
    
    return fig

def create_token_price_chart(df, token1, token2):
    """Create a dual-axis chart for token prices with source attribution"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Filter data for each token
    token1_data = df[df['token_symbol'] == token1]
    token2_data = df[df['token_symbol'] == token2]
    
    # Get price sources if available
    token1_source = "unknown"
    token2_source = "unknown"
    
    if 'price_source' in df.columns:
        if not token1_data.empty and 'price_source' in token1_data.columns:
            token1_source = token1_data['price_source'].iloc[0] if not token1_data['price_source'].empty else "unknown"
        
        if not token2_data.empty and 'price_source' in token2_data.columns:
            token2_source = token2_data['price_source'].iloc[0] if not token2_data['price_source'].empty else "unknown"
    
    # Add traces with source information in the name
    fig.add_trace(
        go.Scatter(
            x=token1_data['timestamp'],
            y=token1_data['price_usd'],
            name=f"{token1} Price (via {token1_source})",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=token2_data['timestamp'],
            y=token2_data['price_usd'],
            name=f"{token2} Price (via {token2_source})",
            line=dict(color='green', width=2)
        ),
        secondary_y=True,
    )
    
    # Add figure title with source attribution
    fig.update_layout(
        title_text=f"{token1} and {token2} Price History with Source Attribution",
        height=400,
        legend_title_text="Price Data Sources"
    )
    
    # Add annotation for data sources
    fig.add_annotation(
        text=f"Data sources: {token1} via {token1_source}, {token2} via {token2_source}",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10)
    )
    
    # Set axes titles
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"{token1} Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text=f"{token2} Price (USD)", secondary_y=True)
    
    return fig

def create_pool_comparison_chart(df, metric, metric_label):
    """Create a line chart comparing multiple pools for a specific metric"""
    fig = px.line(
        df,
        x='timestamp',
        y=metric,
        color='pool_name',
        title=f"{metric_label} Comparison",
        labels={
            'timestamp': 'Date',
            metric: metric_label,
            'pool_name': 'Pool'
        }
    )
    
    # Add marker points
    fig.update_traces(mode='lines+markers')
    
    # Format layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=metric_label,
        height=500
    )
    
    return fig

def create_risk_heat_map(df):
    """Create a risk vs. reward heat map"""
    if df.empty:
        # Create empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Handle unrealistic APR values if present
    display_df = df.copy()
    max_apr_warning = None
    
    # Check for unrealistically high APR values
    max_apr = df['predicted_apr'].max()
    if max_apr > 50:
        max_apr_warning = f"Capping unrealistically high APR values ({max_apr:.2f}%) to 50% for visualization."
        display_df['predicted_apr'] = display_df['predicted_apr'].clip(upper=50)
    
    # Create scatter plot
    fig = px.scatter(
        display_df,
        x='risk_score',
        y='predicted_apr',
        color='predicted_apr',
        size='tvl' if 'tvl' in display_df.columns else None,
        hover_name='pool_name',
        color_continuous_scale='Viridis',
        labels={
            'risk_score': 'Risk Score',
            'predicted_apr': 'Predicted APR (%)',
            'tvl': 'TVL (USD)'
        }
    )
    
    # Calculate max Y value for risk zones
    max_y = min(50, display_df['predicted_apr'].max() * 1.1)
    
    # Add risk zones
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=0.3, y1=max_y,
        line=dict(width=0),
        fillcolor="rgba(0,255,0,0.1)"
    )
    fig.add_shape(
        type="rect",
        x0=0.3, y0=0,
        x1=0.7, y1=max_y,
        line=dict(width=0),
        fillcolor="rgba(255,255,0,0.1)"
    )
    fig.add_shape(
        type="rect",
        x0=0.7, y0=0,
        x1=1.0, y1=max_y,
        line=dict(width=0),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    # Add text annotations for risk zones
    annotation_y = max_y * 0.9
    fig.add_annotation(
        x=0.15, y=annotation_y,
        text="Low Risk",
        showarrow=False,
        font=dict(color="darkgreen", size=12)
    )
    fig.add_annotation(
        x=0.5, y=annotation_y,
        text="Medium Risk",
        showarrow=False,
        font=dict(color="darkgoldenrod", size=12)
    )
    fig.add_annotation(
        x=0.85, y=annotation_y,
        text="High Risk",
        showarrow=False,
        font=dict(color="darkred", size=12)
    )
    
    # Format layout
    fig.update_layout(
        title="Risk vs. Reward Analysis",
        xaxis=dict(
            title="Risk Score",
            range=[0, 1]
        ),
        yaxis=dict(
            title="Predicted APR (%)",
            range=[0, max_y]
        ),
        height=500
    )
    
    # Add warning annotation if needed
    if max_apr_warning:
        fig.add_annotation(
            text=max_apr_warning,
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(color="red", size=10)
        )
    
    return fig

def create_impermanent_loss_chart(token1_change=None, token2_change=None):
    """
    Create a chart showing impermanent loss for different price ratios
    
    Args:
        token1_change: Optional highlight point for token1 price change (percentage)
        token2_change: Optional highlight point for token2 price change (percentage)
    """
    # Generate price ratio changes
    price_changes = np.linspace(-0.9, 9, 100)  # -90% to +900%
    
    # Calculate impermanent loss for each price change
    # Assuming the other token's price remains constant
    impermanent_losses = []
    
    for change in price_changes:
        price_ratio = 1 + change
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        impermanent_losses.append(il * 100)  # Convert to percentage
    
    # Create DataFrame
    il_data = pd.DataFrame({
        'Price Change (%)': price_changes * 100,
        'Impermanent Loss (%)': impermanent_losses
    })
    
    # Create line chart
    fig = px.line(
        il_data,
        x='Price Change (%)',
        y='Impermanent Loss (%)',
        title="Impermanent Loss vs. Price Change",
        labels={
            'Price Change (%)': 'Price Change (%)',
            'Impermanent Loss (%)': 'Impermanent Loss (%)'
        }
    )
    
    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Add highlight point if token changes are specified
    if token1_change is not None and token2_change is not None:
        # Calculate price ratio for the specific changes
        price_ratio = (1 + token1_change/100) / (1 + token2_change/100)
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        il_pct = il * 100
        
        # If one token is stable and one changes
        if token2_change == 0:
            # Add highlight point
            fig.add_trace(go.Scatter(
                x=[token1_change],
                y=[il_pct],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Your Position',
                hovertemplate='Price Change: %{x:.1f}%<br>IL: %{y:.2f}%'
            ))
        else:
            # Add annotation explaining the combined effect
            relative_change = (1 + token1_change/100) / (1 + token2_change/100) - 1
            relative_change_pct = relative_change * 100
            
            fig.add_annotation(
                x=relative_change_pct,
                y=il_pct,
                text=f"Your Position (IL: {il_pct:.2f}%)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                yshift=10
            )
    
    # Format layout
    fig.update_layout(
        height=400,
        yaxis=dict(
            tickformat='.2f'
        )
    )
    
    return fig

def create_prediction_chart(df, metric='predicted_apr'):
    """Create a line chart for prediction history"""
    if df.empty:
        # Create empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # For safer visualization, ensure data is sorted by timestamp
    if 'prediction_timestamp' in df.columns:
        df = df.sort_values('prediction_timestamp')
    
    # Handle unrealistic APR values if present
    display_df = df.copy()
    max_apr_warning = None
    
    if metric == 'predicted_apr':
        max_apr = df[metric].max()
        if max_apr > 50:
            max_apr_warning = f"Capping unrealistically high APR values ({max_apr:.2f}%) to 50% for visualization."
            display_df[metric] = display_df[metric].clip(upper=50)
    
    # Create the line chart
    fig = px.line(
        display_df,
        x='prediction_timestamp',
        y=metric,
        title=f"{'Predicted APR' if metric == 'predicted_apr' else 'Risk Score'} Over Time",
        labels={
            'prediction_timestamp': 'Date',
            'predicted_apr': 'Predicted APR (%)',
            'risk_score': 'Risk Score'
        }
    )
    
    # Add marker points
    fig.update_traces(mode='lines+markers')
    
    # Format layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Predicted APR (%)' if metric == 'predicted_apr' else 'Risk Score',
        height=400
    )
    
    # Set appropriate y-axis limits
    if metric == 'predicted_apr':
        # Set y-axis to max out at 50% or the actual maximum if lower
        ymax = min(50, display_df[metric].max() * 1.1)  # Add 10% padding
        fig.update_layout(yaxis=dict(range=[0, ymax]))
    elif metric == 'risk_score':
        # Risk is always 0-1
        fig.update_layout(yaxis=dict(range=[0, 1]))
    
    # Add warning annotation if needed
    if max_apr_warning:
        fig.add_annotation(
            text=max_apr_warning,
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(color="red", size=10)
        )
    
    return fig

def create_performance_distribution_chart(df):
    """Create a pie chart showing performance class distribution"""
    if df.empty:
        # Create empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Count pools in each performance class
    try:
        perf_counts = df['performance_class'].value_counts().reset_index()
        perf_counts.columns = ['Performance Class', 'Count']
        
        # Map class numbers/strings to labels
        # First check if we have numeric or string values
        if perf_counts['Performance Class'].dtype == 'object':
            # String format (already have 'high', 'medium', 'low')
            class_map = {
                'high': 'High',
                'medium': 'Medium', 
                'low': 'Low'
            }
        else:
            # Numeric format: 1, 2, 3
            # Note that in db_operations.py: high=3, medium=2, low=1
            class_map = {
                3: 'High', 
                2: 'Medium', 
                1: 'Low'
            }
        
        # Apply mapping with fallback for unknown values
        perf_counts['Class'] = perf_counts['Performance Class'].apply(
            lambda x: class_map.get(x, 'Unknown')
        )
        
        # Define colors
        color_map = {
            'High': '#4CAF50',    # Green
            'Medium': '#FFC107',  # Amber
            'Low': '#F44336',     # Red
            'Unknown': '#9E9E9E'  # Gray
        }
        
        # Create pie chart
        fig = px.pie(
            perf_counts,
            values='Count',
            names='Class',
            title="Distribution of Performance Classes",
            color='Class',
            color_discrete_map=color_map
        )
        
        # Format layout
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        # If something goes wrong, return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating performance distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="red")
        )
        return fig