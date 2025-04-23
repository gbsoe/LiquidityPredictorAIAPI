import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def create_metrics_chart(metrics_df, metric_col, metric_name, title):
    """
    Create a time series chart for a specific metric
    
    Args:
        metrics_df: DataFrame with metrics data
        metric_col: Column name for the metric to plot
        metric_name: Display name for the metric
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    # Ensure timestamp is datetime
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
    
    # Sort by timestamp
    sorted_df = metrics_df.sort_values('timestamp')
    
    # Create figure
    fig = go.Figure()
    
    # Add line for the metric
    fig.add_trace(go.Scatter(
        x=sorted_df['timestamp'],
        y=sorted_df[metric_col],
        mode='lines',
        name=metric_name,
        line=dict(width=2)
    ))
    
    # Add moving average
    if len(sorted_df) > 5:
        sorted_df['ma'] = sorted_df[metric_col].rolling(window=5).mean()
        fig.add_trace(go.Scatter(
            x=sorted_df['timestamp'],
            y=sorted_df['ma'],
            mode='lines',
            name=f'{metric_name} (5-point MA)',
            line=dict(width=1, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=metric_name,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_liquidity_volume_chart(metrics_df):
    """
    Create a dual-axis chart for liquidity and volume
    
    Args:
        metrics_df: DataFrame with metrics data
    
    Returns:
        Plotly figure object
    """
    # Ensure timestamp is datetime
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
    
    # Sort by timestamp
    sorted_df = metrics_df.sort_values('timestamp')
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add liquidity trace
    fig.add_trace(
        go.Scatter(
            x=sorted_df['timestamp'],
            y=sorted_df['liquidity'],
            name="Liquidity",
            line=dict(color="blue", width=2)
        ),
        secondary_y=False
    )
    
    # Add volume trace
    fig.add_trace(
        go.Scatter(
            x=sorted_df['timestamp'],
            y=sorted_df['volume'],
            name="Volume",
            line=dict(color="green", width=2)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Liquidity and Volume Over Time",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Liquidity (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume (USD)", secondary_y=True)
    
    return fig

def create_token_price_chart(token_prices, token1, token2):
    """
    Create a chart for token price histories
    
    Args:
        token_prices: DataFrame with token price data
        token1: First token symbol
        token2: Second token symbol
    
    Returns:
        Plotly figure object
    """
    # Ensure timestamp is datetime
    token_prices['timestamp'] = pd.to_datetime(token_prices['timestamp'])
    
    # Create figure
    fig = go.Figure()
    
    # Filter data for each token
    token1_data = token_prices[token_prices['token_symbol'] == token1]
    token2_data = token_prices[token_prices['token_symbol'] == token2]
    
    # Add trace for first token
    if not token1_data.empty:
        fig.add_trace(go.Scatter(
            x=token1_data['timestamp'],
            y=token1_data['price_usd'],
            name=f"{token1} Price",
            line=dict(color="blue", width=2)
        ))
    
    # Add trace for second token
    if not token2_data.empty:
        fig.add_trace(go.Scatter(
            x=token2_data['timestamp'],
            y=token2_data['price_usd'],
            name=f"{token2} Price",
            line=dict(color="red", width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{token1} and {token2} Price History",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_pool_comparison_chart(comparison_data, metric_col, metric_name):
    """
    Create a chart comparing a specific metric across multiple pools
    
    Args:
        comparison_data: DataFrame with data for multiple pools
        metric_col: Column name for the metric to plot
        metric_name: Display name for the metric
    
    Returns:
        Plotly figure object
    """
    # Ensure timestamp is datetime
    comparison_data['timestamp'] = pd.to_datetime(comparison_data['timestamp'])
    
    # Create figure
    fig = px.line(
        comparison_data,
        x='timestamp',
        y=metric_col,
        color='pool_name',
        title=f"{metric_name} Comparison",
        labels={
            'timestamp': 'Time',
            metric_col: metric_name,
            'pool_name': 'Pool'
        }
    )
    
    # Update layout
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_prediction_chart(merged_data):
    """
    Create a chart comparing predicted vs. actual values
    
    Args:
        merged_data: DataFrame with actual and predicted values
    
    Returns:
        Plotly figure object
    """
    # Ensure timestamp is datetime
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])
    
    # Create figure
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=merged_data['timestamp'],
        y=merged_data['apr'],
        mode='lines',
        name='Actual APR',
        line=dict(color='blue', width=2)
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=merged_data['timestamp'],
        y=merged_data['predicted_apr'],
        mode='lines',
        name='Predicted APR',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Predicted vs. Actual APR',
        xaxis_title='Time',
        yaxis_title='APR (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_performance_distribution_chart(predictions_df):
    """
    Create a chart showing distribution of performance classes
    
    Args:
        predictions_df: DataFrame with prediction data
    
    Returns:
        Plotly figure object
    """
    # Count pools in each performance class
    performance_counts = predictions_df['performance_class'].value_counts().reset_index()
    performance_counts.columns = ['Performance Class', 'Count']
    
    # Map performance to numeric order for sorting
    class_order = {'high': 0, 'medium': 1, 'low': 2}
    performance_counts['sort_order'] = performance_counts['Performance Class'].map(class_order)
    performance_counts = performance_counts.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Create color map
    color_map = {'high': '#4CAF50', 'medium': '#FFC107', 'low': '#F44336'}
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=performance_counts['Performance Class'],
        y=performance_counts['Count'],
        marker_color=[color_map.get(pc, '#000000') for pc in performance_counts['Performance Class']],
        text=performance_counts['Count'],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title='Distribution of Performance Classes',
        xaxis_title='Performance Class',
        yaxis_title='Number of Pools'
    )
    
    return fig

def create_risk_heat_map(predictions_df):
    """
    Create a heat map of risk vs. APR
    
    Args:
        predictions_df: DataFrame with prediction data
    
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=predictions_df['risk_score'],
        y=predictions_df['predicted_apr'],
        mode='markers',
        marker=dict(
            size=12,
            color=predictions_df['risk_score'],
            colorscale='Viridis',
            colorbar=dict(title='Risk Score'),
            line=dict(width=1, color='black')
        ),
        text=predictions_df['pool_name'],
        hovertemplate='<b>%{text}</b><br>Risk Score: %{x:.2f}<br>Predicted APR: %{y:.2f}%'
    ))
    
    # Add background zones
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        y0=predictions_df['predicted_apr'].min() * 0.9,
        x1=0.3,
        y1=predictions_df['predicted_apr'].max() * 1.1,
        fillcolor="rgba(76, 175, 80, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0.3,
        y0=predictions_df['predicted_apr'].min() * 0.9,
        x1=0.7,
        y1=predictions_df['predicted_apr'].max() * 1.1,
        fillcolor="rgba(255, 193, 7, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0.7,
        y0=predictions_df['predicted_apr'].min() * 0.9,
        x1=1.0,
        y1=predictions_df['predicted_apr'].max() * 1.1,
        fillcolor="rgba(244, 67, 54, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Update layout
    fig.update_layout(
        title='Risk vs. Predicted APR',
        xaxis_title='Risk Score',
        yaxis_title='Predicted APR (%)',
        xaxis=dict(range=[0, 1]),
        hovermode='closest'
    )
    
    return fig

def create_impermanent_loss_chart():
    """
    Create a chart showing impermanent loss for different price ratios
    
    Returns:
        Plotly figure object
    """
    # Generate x values (price ratios)
    price_ratios = np.linspace(0.1, 5.0, 100)
    
    # Calculate impermanent loss for each price ratio
    # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
    impermanent_loss = 2 * np.sqrt(price_ratios) / (1 + price_ratios) - 1
    
    # Convert to percentage
    impermanent_loss_pct = impermanent_loss * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for impermanent loss
    fig.add_trace(go.Scatter(
        x=price_ratios,
        y=impermanent_loss_pct,
        mode='lines',
        name='Impermanent Loss',
        line=dict(color='red', width=3)
    ))
    
    # Add annotations for key points
    key_points = {
        0.5: "50% price ratio",
        2.0: "200% price ratio",
        4.0: "400% price ratio"
    }
    
    for price_ratio, label in key_points.items():
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        il_pct = il * 100
        
        fig.add_annotation(
            x=price_ratio,
            y=il_pct,
            text=f"{label}<br>{il_pct:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            ax=20,
            ay=-30
        )
    
    # Update layout
    fig.update_layout(
        title='Impermanent Loss vs. Price Ratio Change',
        xaxis_title='Price Ratio (New / Initial)',
        yaxis_title='Impermanent Loss (%)',
        yaxis=dict(
            range=[min(impermanent_loss_pct) * 1.1, 0]  # Invert y-axis to show loss as negative
        )
    )
    
    # Add reference lines
    fig.add_shape(
        type="line",
        x0=0.1,
        y0=0,
        x1=5.0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    return fig
