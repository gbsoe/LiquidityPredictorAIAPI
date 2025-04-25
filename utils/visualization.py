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
    """Create a dual-axis chart for token prices"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Filter data for each token
    token1_data = df[df['token_symbol'] == token1]
    token2_data = df[df['token_symbol'] == token2]
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=token1_data['timestamp'],
            y=token1_data['price_usd'],
            name=f"{token1} Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=token2_data['timestamp'],
            y=token2_data['price_usd'],
            name=f"{token2} Price",
            line=dict(color='green', width=2)
        ),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text=f"{token1} and {token2} Price History",
        height=400
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
    # Create scatter plot
    fig = px.scatter(
        df,
        x='risk_score',
        y='predicted_apr',
        color='predicted_apr',
        size='tvl',
        hover_name='pool_name',
        color_continuous_scale='Viridis',
        labels={
            'risk_score': 'Risk Score',
            'predicted_apr': 'Predicted APR (%)',
            'tvl': 'TVL (USD)'
        }
    )
    
    # Add risk zones
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=0.3, y1=df['predicted_apr'].max() * 1.1,
        line=dict(width=0),
        fillcolor="rgba(0,255,0,0.1)"
    )
    fig.add_shape(
        type="rect",
        x0=0.3, y0=0,
        x1=0.7, y1=df['predicted_apr'].max() * 1.1,
        line=dict(width=0),
        fillcolor="rgba(255,255,0,0.1)"
    )
    fig.add_shape(
        type="rect",
        x0=0.7, y0=0,
        x1=1.0, y1=df['predicted_apr'].max() * 1.1,
        line=dict(width=0),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    # Add text annotations for risk zones
    fig.add_annotation(
        x=0.15, y=df['predicted_apr'].max(),
        text="Low Risk",
        showarrow=False,
        font=dict(color="darkgreen", size=12)
    )
    fig.add_annotation(
        x=0.5, y=df['predicted_apr'].max(),
        text="Medium Risk",
        showarrow=False,
        font=dict(color="darkgoldenrod", size=12)
    )
    fig.add_annotation(
        x=0.85, y=df['predicted_apr'].max(),
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
            title="Predicted APR (%)"
        ),
        height=500
    )
    
    return fig

def create_impermanent_loss_chart():
    """Create a chart showing impermanent loss for different price ratios"""
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
    
    # Format layout
    fig.update_layout(
        height=400,
        yaxis=dict(
            tickformat='.2f'
        )
    )
    
    return fig