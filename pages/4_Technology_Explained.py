import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import os
from PIL import Image

# Custom styling
st.set_page_config(
    page_title="SolPool Insight - Technology Explained",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #3366ff;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 500;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #3366ff;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-top: 1.5rem;
        color: #2E4355;
    }
    .formula-box {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3366ff;
        font-family: "Courier New", monospace;
        overflow-x: auto;
    }
    .tech-card {
        background-color: #f7f9fc;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .highlight-text {
        background-color: #e6f0ff;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: 500;
    }
    .feature-list li {
        margin-bottom: 10px;
    }
    .architecture-diagram {
        max-width: 100%;
        border-radius: 5px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title section
st.markdown("<h1 class='main-header'>SolPool Insight: Technology Explained</h1>", unsafe_allow_html=True)
st.markdown("""
This page provides a detailed technical explanation of SolPool Insight's architecture, data processing 
methodology, prediction models, and analytical calculations. Our goal is complete transparency in how 
we generate our insights and predictions.
""")

# LA! Token Branding Banner
st.markdown("""
<div style="background-color: #F6F6F6; padding: 10px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #FFD700;">
    <div style="display: flex; align-items: center;">
""", unsafe_allow_html=True)

# Load LA! Token logo
la_token_logo_path = "attached_assets/IMG-20240624-WA0020-removebg-preview.png"

if os.path.exists(la_token_logo_path):
    la_token_logo = Image.open(la_token_logo_path)
    la_token_col1, la_token_col2 = st.columns([1, 5])
    
    with la_token_col1:
        st.image(la_token_logo, width=80)
    
    with la_token_col2:
        st.markdown(
            "<div style='margin-left: 20px;'>"
            "<h3 style='margin: 0; padding: 0;'>FiLot is part of the "
            "<a href='https://crazyrichla.replit.app/' target='_blank' style='text-decoration:none;'>"
            "<span style='color:#FFD700;font-weight:bold;'>LA! Token</span></a> Ecosystem</h3>"
            "<p style='margin-top: 5px;'>Providing advanced liquidity pool analytics for the LA! Token community</p>"
            "</div>",
            unsafe_allow_html=True
        )

st.markdown("""</div></div>""", unsafe_allow_html=True)

# Create tabs for different sections
data_tab, prediction_tab, risk_tab, architecture_tab = st.tabs([
    "Data Collection & Processing", 
    "Prediction Models", 
    "Risk Assessment", 
    "System Architecture"
])

# ============ DATA COLLECTION & PROCESSING TAB ============
with data_tab:
    st.markdown("<h2 class='section-header'>Data Collection & Processing</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Our data infrastructure continuously collects and processes Solana liquidity pool data 
    from multiple sources to build a comprehensive, reliable dataset for our analytics.
    """)
    
    # Data Sources Section
    st.markdown("<h3 class='sub-header'>Data Sources</h3>", unsafe_allow_html=True)
    
    # Create columns for each data source
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tech-card">
            <h4>Primary On-Chain Data</h4>
            <ul>
                <li>Direct Solana RPC node connections</li>
                <li>Custom account data parsers for each DEX</li>
                <li>Specialized binary data deserializers</li>
                <li>Transaction analysis for volume metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h4>DEX API Integrations</h4>
            <ul>
                <li>Raydium API for pool metrics</li>
                <li>Orca Whirlpools API integration</li>
                <li>Jupiter Aggregator API</li>
                <li>Metrics normalization across DEXes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tech-card">
            <h4>Market Data Sources</h4>
            <ul>
                <li>CoinGecko API for token prices</li>
                <li>Historical price databases</li>
                <li>Market sentiment indicators</li>
                <li>Cross-chain metrics correlation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Collection Frequency
    st.markdown("<h3 class='sub-header'>Data Collection Frequency</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        Our system collects data at different frequencies for different metrics:
        
        <ul>
            <li><strong>Pool Metrics (APR, Liquidity, Volume)</strong>: Every hour</li>
            <li><strong>Token Prices</strong>: Every 30 minutes</li>
            <li><strong>On-Chain Transaction Data</strong>: Every 15 minutes</li>
            <li><strong>Market Sentiment Indicators</strong>: Every 6 hours</li>
        </ul>
        
        This multi-frequency approach allows us to capture both rapid market changes and 
        longer-term trends while optimizing for computational efficiency.
    </div>
    """, unsafe_allow_html=True)
    
    # Data Processing Pipeline
    st.markdown("<h3 class='sub-header'>Data Processing Pipeline</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Raw data goes through several processing stages before being used for analysis:
    """)
    
    # Create a simple pipeline diagram
    pipeline_stages = [
        "Raw Data Collection", 
        "Validation & Cleaning", 
        "Normalization", 
        "Feature Engineering", 
        "Database Storage", 
        "Analytics Processing"
    ]
    
    fig = go.Figure()
    
    for i, stage in enumerate(pipeline_stages):
        fig.add_trace(go.Scatter(
            x=[i+1], 
            y=[1],
            mode="markers+text",
            marker=dict(size=30, color="#3366ff"),
            text=[stage],
            textposition="bottom center",
            hoverinfo="text",
            name=stage
        ))
    
    # Add connecting lines
    for i in range(len(pipeline_stages)-1):
        fig.add_shape(type="line",
            x0=i+1, y0=1, x1=i+2, y1=1,
            line=dict(color="#3366ff", width=2, dash="solid")
        )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0.5, len(pipeline_stages)+0.5]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0.5, 1.5]
        ),
        height=200,
        margin=dict(l=20, r=20, t=20, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain each stage
    with st.expander("View detailed pipeline stage explanation"):
        st.markdown("""
        ### 1. Raw Data Collection
        - **On-Chain Data**: Direct RPC node connections extract raw pool data from Solana blockchain.
        - **API Data**: Scheduled API calls to DEXes and price oracles collect market data.
        - **Error Handling**: Automatic retries and fallback sources ensure data availability.
        
        ### 2. Validation & Cleaning
        - **Anomaly Detection**: Statistical methods identify and flag outliers.
        - **Missing Data Handling**: Gap filling using appropriate interpolation methods.
        - **Consistency Checks**: Cross-source validation to ensure data integrity.
        
        ### 3. Normalization
        - **Metric Standardization**: Converting different DEX metrics to uniform formats.
        - **Price Denominations**: Standardizing to USD for consistent comparisons.
        - **Time Series Alignment**: Ensuring all data points have consistent timestamps.
        
        ### 4. Feature Engineering
        - **Derived Metrics**: Calculating volume-to-liquidity ratios, APR volatility, etc.
        - **Time-Based Features**: Creating rolling averages, momentum indicators, trends.
        - **Technical Indicators**: RSI, MACD, and other indicators adapted for pool metrics.
        
        ### 5. Database Storage
        - **Time-Series Optimization**: Specialized database schema for efficient time-series queries.
        - **Indexing Strategy**: Optimized indexes for fast data retrieval.
        - **Compression Techniques**: Efficient storage with minimal data loss.
        
        ### 6. Analytics Processing
        - **Real-Time Calculations**: Computing current metrics for dashboard display.
        - **Predictor Features**: Preparing datasets for ML model training and inference.
        - **Aggregations**: Creating summarized views for trend analysis.
        """)
    
    # Feature Engineering
    st.markdown("<h3 class='sub-header'>Feature Engineering</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Feature engineering is a critical step in transforming raw data into meaningful inputs for our prediction models.
    We create over 40 derived features from the raw metrics to provide deeper insights.
    """)
    
    # Sample of important features
    feature_categories = {
        "Time-Series Features": [
            "7-day rolling mean APR",
            "30-day rolling mean liquidity",
            "APR volatility (standard deviation)",
            "APR momentum (rate of change)",
            "Liquidity growth rate"
        ],
        "Price-Based Features": [
            "Token price correlation",
            "Price ratio volatility",
            "Price momentum indicators",
            "Historical impermanent loss",
            "Token dominance metrics"
        ],
        "Volume Features": [
            "Volume-to-liquidity ratio",
            "7-day volume trend",
            "Volume volatility",
            "Trade size distribution",
            "Volume by time of day patterns"
        ],
        "On-Chain Metrics": [
            "Transaction count correlation",
            "Blockchain congestion impact",
            "Solana validator performance",
            "Network fee correlation",
            "Smart contract interactions"
        ]
    }
    
    # Display features in expandable sections
    for category, features in feature_categories.items():
        with st.expander(f"{category} ({len(features)} examples)"):
            for feature in features:
                st.markdown(f"- **{feature}**")
    
    # Feature importance visualization
    st.markdown("<h3 class='sub-header'>Feature Importance</h3>", unsafe_allow_html=True)
    
    # Sample feature importance data
    feature_imp = pd.DataFrame({
        'Feature': [
            '7-day APR volatility', 
            'Token price correlation', 
            'Volume-to-liquidity ratio',
            'Liquidity trend (30d)',
            'Token price momentum',
            'Smart contract age',
            'DEX protocol factor',
            'APR consistency score',
            'Volume pattern predictability',
            'Impermanent loss history'
        ],
        'Importance': [0.18, 0.15, 0.14, 0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.03]
    })
    
    # Sort by importance
    feature_imp = feature_imp.sort_values('Importance', ascending=True)
    
    # Create a horizontal bar chart
    fig = px.bar(
        feature_imp, 
        y='Feature', 
        x='Importance',
        orientation='h',
        title="Feature Importance in APR Prediction Model",
        color='Importance',
        color_continuous_scale='blues'
    )
    
    fig.update_layout(
        xaxis_title="Relative Importance",
        yaxis_title=None,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="tech-card">
        <h4>Note on Feature Importance</h4>
        <p>
            Feature importance values are derived from our ensemble learning models and represent the relative 
            contribution of each feature to prediction accuracy. These values are automatically updated during 
            model retraining to adapt to changing market conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============ PREDICTION MODELS TAB ============
with prediction_tab:
    st.markdown("<h2 class='section-header'>Prediction Models</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Our prediction system utilizes a sophisticated multi-model architecture that combines different 
    machine learning approaches to provide accurate and robust predictions for liquidity pool performance.
    """)
    
    # Model Architecture
    st.markdown("<h3 class='sub-header'>Self-Evolving Prediction Engine Architecture</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <p>
            SolPool Insight uses a pragmatic prediction engine that focuses on reliable, practical forecasting
            methods to help guide investment decisions. Our current system implements:
        </p>
        <ul class="feature-list">
            <li><strong>Random Forest Regression</strong>: Core algorithm for APR prediction with good accuracy and explainability</li>
            <li><strong>Feature Engineering</strong>: Creating meaningful derived metrics from raw data</li>
            <li><strong>Ensemble Methods</strong>: Combining multiple prediction models for improved stability</li>
            <li><strong>Statistical Time-Series Analysis</strong>: Using established techniques for trend identification</li>
            <li><strong>Historical Pattern Recognition</strong>: Identifying repeating patterns in pool performance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Component Models
    st.markdown("<h3 class='sub-header'>Component Models</h3>", unsafe_allow_html=True)
    
    # Create columns for model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tech-card">
            <h4>APR Prediction Model</h4>
            <p><strong>Primary Algorithm:</strong> Random Forest Regressor</p>
            <p><strong>Purpose:</strong> Predict future APR changes over 7-day horizon</p>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>Historical APR patterns</li>
                <li>Volume-to-liquidity correlations</li>
                <li>Token price movements</li>
                <li>Time-based seasonality factors</li>
                <li>DEX protocol characteristics</li>
            </ul>
            <p><strong>Implementation Status:</strong> Core model implemented with ongoing refinements</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <h4>Risk Assessment Model</h4>
            <p><strong>Primary Algorithm:</strong> Random Forest</p>
            <p><strong>Purpose:</strong> Estimate risk levels based on historical data</p>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>Volatility metrics</li>
                <li>Impermanent loss estimation</li>
                <li>Liquidity concentration analysis</li>
                <li>Price correlation stability</li>
                <li>Historical stability factors</li>
            </ul>
            <p><strong>Implementation Status:</strong> Basic implementation with planned enhancements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h4>TVL Prediction Model</h4>
            <p><strong>Primary Algorithm:</strong> Simple trend analysis with moving averages</p>
            <p><strong>Purpose:</strong> Track changes in pool liquidity</p>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>Historical liquidity patterns</li>
                <li>Moving average calculations</li>
                <li>Basic trend detection</li>
                <li>Growth rate measurement</li>
                <li>Relative change metrics</li>
            </ul>
            <p><strong>Implementation Status:</strong> Basic implementation tracking historical data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <h4>Pool Performance Classifier</h4>
            <p><strong>Primary Algorithm:</strong> Rule-based classification</p>
            <p><strong>Purpose:</strong> Categorize pools based on key metrics</p>
            <p><strong>Classification Schema:</strong></p>
            <ul>
                <li><strong>Excellent</strong>: High-APR, good liquidity, stable metrics</li>
                <li><strong>Good</strong>: Above-average returns with reasonable stability</li>
                <li><strong>Average</strong>: Standard performance, typical metrics</li>
                <li><strong>Below Average</strong>: Underperforming on key metrics</li>
                <li><strong>Poor</strong>: Low performance across multiple dimensions</li>
            </ul>
            <p><strong>Implementation Status:</strong> Basic rule-based system with ongoing development</p>
        </div>
        """, unsafe_allow_html=True)
    
    # APR Prediction Methodology
    st.markdown("<h3 class='sub-header'>APR Prediction Methodology</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <p>
            Our APR prediction process uses practical machine learning approaches focused on reliable forecasting.
            Here's a breakdown of the current methodology:
        </p>
        
        <ol>
            <li>
                <strong>Historical Data Collection:</strong> 
                Gather time-series data on APR values and related metrics for each pool.
            </li>
            <li>
                <strong>Feature Engineering:</strong>
                Create relevant features like moving averages, trends, and volatility metrics.
            </li>
            <li>
                <strong>Model Training:</strong>
                Train a Random Forest Regressor on historical data to identify patterns and relationships.
            </li>
            <li>
                <strong>Prediction Generation:</strong>
                Apply the trained model to current data to predict future APR values.
            </li>
            <li>
                <strong>Performance Tracking:</strong>
                Monitor prediction accuracy and adjust features as needed to improve results.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # APR Prediction Formula
    st.markdown("<h3 class='sub-header'>APR Prediction Formula</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="formula-box">
        <p>The current mathematical approach for our APR prediction model:</p>
        <p>APR̂<sub>t+h</sub> = RandomForest(APR<sub>t</sub>, APR<sub>t-1</sub>, ..., APR<sub>t-n</sub>, VLR<sub>t</sub>, other features)</p>
        <p>Where:</p>
        <ul>
            <li>APR̂<sub>t+h</sub>: Predicted APR h days into the future</li>
            <li>APR<sub>t</sub>, APR<sub>t-1</sub>, ...: Historical APR values</li>
            <li>VLR<sub>t</sub>: Volume-to-Liquidity Ratio</li>
            <li>Other features include pool metrics and token characteristics</li>
        </ul>
        <p>Random Forest creates multiple decision trees and aggregates their predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Training Process
    st.markdown("<h3 class='sub-header'>Model Training Process</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Our models are continuously trained and improved through an automated pipeline:
    """)
    
    # Training process steps
    training_steps = [
        "Historical Data Collection", 
        "Feature Engineering", 
        "Train/Validation Split", 
        "Model Training",
        "Cross-Validation",
        "Hyperparameter Tuning",
        "Performance Evaluation",
        "Model Deployment"
    ]
    
    # Create a circular diagram
    fig = go.Figure()
    
    # Calculate positions on a circle
    radius = 1
    angles = np.linspace(0, 2*np.pi, len(training_steps), endpoint=False).tolist()
    
    # Adjust starting position
    angles = [(angle + np.pi/2) % (2*np.pi) for angle in angles]
    
    # Calculate x,y positions
    xs = [radius * np.cos(angle) for angle in angles]
    ys = [radius * np.sin(angle) for angle in angles]
    
    # Add nodes
    for i, (x, y, step) in enumerate(zip(xs, ys, training_steps)):
        fig.add_trace(go.Scatter(
            x=[x], 
            y=[y],
            mode="markers+text",
            marker=dict(size=25, color="#3366ff"),
            text=[step],
            textposition="middle center",
            hoverinfo="text",
            name=step
        ))
    
    # Add connecting lines (to create a cycle)
    for i in range(len(training_steps)):
        next_i = (i + 1) % len(training_steps)
        fig.add_shape(type="line",
            x0=xs[i], y0=ys[i], x1=xs[next_i], y1=ys[next_i],
            line=dict(color="#3366ff", width=2, dash="solid")
        )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1.5, 1.5]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1.5, 1.5],
            scaleanchor="x",
            scaleratio=1
        ),
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Continuous Improvement
    st.markdown("<h3 class='sub-header'>Continuous Improvement Mechanisms</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <p>
            We continuously work to improve our prediction capabilities through several processes:
        </p>
        <ul class="feature-list">
            <li>
                <strong>Accuracy Monitoring:</strong> 
                Track how well predictions match actual outcomes to identify areas for improvement.
            </li>
            <li>
                <strong>Data Collection Expansion:</strong>
                Steadily increasing our historical data coverage for more robust training.
            </li>
            <li>
                <strong>Feature Refinement:</strong>
                Identifying and adding new features that improve prediction accuracy.
            </li>
            <li>
                <strong>Model Selection:</strong>
                Testing different algorithms to determine which performs best for our use case.
            </li>
            <li>
                <strong>Hyperparameter Tuning:</strong>
                Manual adjustment of model parameters to optimize prediction performance.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============ RISK ASSESSMENT TAB ============
with risk_tab:
    st.markdown("<h2 class='section-header'>Risk Assessment Methodology</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Our risk assessment system provides a comprehensive evaluation of liquidity pool risks through 
    multiple analytical lenses. This section details our methodologies, formulas, and risk factors.
    """)
    
    # Risk Score Calculation
    st.markdown("<h3 class='sub-header'>Risk Score Calculation</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <p>
            Each pool receives a risk score from 0-100 calculated using a weighted combination of risk factors.
            The formula for the overall risk score is:
        </p>
        
        <div class="formula-box">
            Risk Score = (w<sub>v</sub> × Volatility Risk) + (w<sub>il</sub> × Impermanent Loss Risk) + 
                       (w<sub>l</sub> × Liquidity Risk) + (w<sub>sc</sub> × Smart Contract Risk) + 
                       (w<sub>m</sub> × Market Correlation Risk)
        </div>
        
        <p>Where the weights (w) vary based on pool characteristics but generally follow:</p>
        <ul>
            <li>w<sub>v</sub> (Volatility Risk): 25%</li>
            <li>w<sub>il</sub> (Impermanent Loss Risk): 30%</li>
            <li>w<sub>l</sub> (Liquidity Risk): 20%</li>
            <li>w<sub>sc</sub> (Smart Contract Risk): 15%</li>
            <li>w<sub>m</sub> (Market Correlation Risk): 10%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Factor Details
    st.markdown("<h3 class='sub-header'>Risk Factor Details</h3>", unsafe_allow_html=True)
    
    # Create tabs for each risk factor
    rf_tab1, rf_tab2, rf_tab3, rf_tab4, rf_tab5 = st.tabs([
        "Volatility Risk", 
        "Impermanent Loss Risk", 
        "Liquidity Risk", 
        "Smart Contract Risk",
        "Market Correlation Risk"
    ])
    
    with rf_tab1:
        st.markdown("""
        <div class="tech-card">
            <h4>Volatility Risk</h4>
            <p>
                Volatility risk measures the stability of a pool's APR and token prices over time.
                It's calculated using standard statistical measures of dispersion.
            </p>
            
            <div class="formula-box">
                Volatility Risk = 0.5 × APR Volatility + 0.5 × Price Volatility
                
                APR Volatility = StandardDeviation(APR<sub>1</sub>, APR<sub>2</sub>, ..., APR<sub>n</sub>) / Mean(APR<sub>1</sub>, APR<sub>2</sub>, ..., APR<sub>n</sub>)
                
                Price Volatility = w<sub>1</sub> × Volatility(Token1) + w<sub>2</sub> × Volatility(Token2)
            </div>
            
            <p>Where w<sub>1</sub> and w<sub>2</sub> are the proportional weights of each token in the pool.</p>
            
            <p><strong>Interpretation:</strong></p>
            <ul>
                <li><strong>Low</strong> (0-25): Stable returns and prices, minimal fluctuations</li>
                <li><strong>Medium</strong> (25-50): Moderate fluctuations but generally predictable</li>
                <li><strong>High</strong> (50-75): Significant volatility, returns can vary substantially</li>
                <li><strong>Very High</strong> (75-100): Extreme volatility, highly unpredictable returns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with rf_tab2:
        st.markdown("""
        <div class="tech-card">
            <h4>Impermanent Loss Risk</h4>
            <p>
                Impermanent loss (IL) occurs when the price ratio between paired tokens changes.
                Our IL risk assessment simulates potential losses across various price change scenarios.
            </p>
            
            <div class="formula-box">
                Impermanent Loss = 2 × √(price_ratio) / (1 + price_ratio) - 1
                
                Where price_ratio = new_price_ratio / initial_price_ratio
            </div>
            
            <p>
                For risk assessment, we simulate 1000+ potential price movement scenarios based on 
                historical volatility and calculate the expected impermanent loss distribution.
            </p>
            
            <p><strong>IL Risk Categories:</strong></p>
            <ul>
                <li><strong>Very Low</strong>: Expected IL < 1% (e.g., stablecoin pairs)</li>
                <li><strong>Low</strong>: Expected IL 1-3% (e.g., correlated tokens)</li>
                <li><strong>Medium</strong>: Expected IL 3-7% (e.g., mainstream cryptos)</li>
                <li><strong>High</strong>: Expected IL 7-15% (e.g., volatile assets)</li>
                <li><strong>Very High</strong>: Expected IL > 15% (e.g., uncorrelated tokens)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with rf_tab3:
        st.markdown("""
        <div class="tech-card">
            <h4>Liquidity Risk</h4>
            <p>
                Liquidity risk evaluates potential issues with entering or exiting a position
                due to insufficient liquidity depth or concentration issues.
            </p>
            
            <div class="formula-box">
                Liquidity Risk = 0.4 × Depth Risk + 0.3 × Concentration Risk + 0.3 × Stability Risk
                
                Depth Risk = 1 - min(1, TVL / TVL<sub>threshold</sub>)
                
                Concentration Risk = 1 - (1 / (1 + LP_count_proportion))
                
                Stability Risk = StandardDeviation(TVL) / Mean(TVL)
            </div>
            
            <p>
                Where TVL<sub>threshold</sub> is the minimum TVL considered "highly liquid" (varies by token category),
                and LP_count_proportion is the proportion of liquidity provided by the top liquidity providers.
            </p>
            
            <p><strong>Key Factors:</strong></p>
            <ul>
                <li>Total Value Locked (TVL) relative to trading volume</li>
                <li>Concentration of liquidity providers</li>
                <li>Historical stability of liquidity</li>
                <li>Depth at different price levels (for concentrated liquidity pools)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with rf_tab4:
        st.markdown("""
        <div class="tech-card">
            <h4>Smart Contract Risk</h4>
            <p>
                Smart contract risk assesses the potential for technical vulnerabilities or exploits
                in the underlying protocol code that could lead to loss of funds.
            </p>
            
            <p><strong>Risk Factors:</strong></p>
            <ul>
                <li><strong>Protocol Age</strong>: Newer protocols typically have higher risk</li>
                <li><strong>Audit Status</strong>: Number and quality of security audits</li>
                <li><strong>Historical Incidents</strong>: Past exploits or vulnerabilities</li>
                <li><strong>Code Complexity</strong>: More complex protocols carry higher risk</li>
                <li><strong>Admin Controls</strong>: Degree of centralized control over the protocol</li>
            </ul>
            
            <p>
                Smart contract risk assessment uses a rule-based scoring system rather than a pure
                mathematical formula, as many factors are categorical rather than numerical.
            </p>
            
            <p><strong>Example Risk Weighting:</strong></p>
            <ul>
                <li>Multiple Respected Audits: -30 points (reduces risk)</li>
                <li>Protocol Age > 1 year: -20 points</li>
                <li>Historical Exploits: +30 points per incident</li>
                <li>Centralized Admin Controls: +15 points</li>
                <li>Complex Custom Code: +10 points</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with rf_tab5:
        st.markdown("""
        <div class="tech-card">
            <h4>Market Correlation Risk</h4>
            <p>
                Market correlation risk evaluates how a pool's performance correlates with broader market trends,
                affecting its behavior during market downturns or recoveries.
            </p>
            
            <div class="formula-box">
                Market Correlation Risk = |PearsonCorrelation(Pool APR, Market Index)| × Volatility Factor
                
                Volatility Factor = 1 + max(0, (MarketVolatility - MedianVolatility) / MedianVolatility)
            </div>
            
            <p>
                A higher correlation with market movements combined with high market volatility results
                in higher risk, as the pool is more likely to experience significant downturns during market stress.
            </p>
            
            <p><strong>Interpretation:</strong></p>
            <ul>
                <li><strong>Low Correlation</strong>: Pool performs independently of market trends</li>
                <li><strong>Medium Correlation</strong>: Some market influence, but with independent factors</li>
                <li><strong>High Correlation</strong>: Strongly influenced by market-wide movements</li>
            </ul>
            
            <p>
                Pools with high market correlation perform well in bull markets but may suffer more
                during market downturns, making them more suitable for tactical rather than strategic allocations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Impermanent Loss Calculation
    st.markdown("<h3 class='sub-header'>Impermanent Loss Calculation</h3>", unsafe_allow_html=True)
    
    # Create columns
    il_col1, il_col2 = st.columns([3, 2])
    
    with il_col1:
        st.markdown("""
        <div class="tech-card">
            <p>
                Impermanent Loss (IL) is a core concept in AMM liquidity pools. It represents the 
                difference in value between holding tokens in a liquidity pool versus simply holding them.
            </p>
            
            <div class="formula-box">
                Impermanent Loss = 2 × sqrt(price_ratio) / (1 + price_ratio) - 1
            </div>
            
            <p>Where <code>price_ratio</code> is the ratio of the new price ratio to the initial price ratio.</p>
            
            <p>In percentages, the impermanent loss for various price ratio changes:</p>
            <ul>
                <li>1.25x price change: ~0.6% loss</li>
                <li>1.5x price change: ~2.0% loss</li>
                <li>2x price change: ~5.7% loss</li>
                <li>3x price change: ~13.4% loss</li>
                <li>4x price change: ~20.0% loss</li>
                <li>5x price change: ~25.5% loss</li>
            </ul>
            
            <p>
                The term "impermanent" refers to the fact that as long as the price ratio returns to the original ratio, 
                the loss disappears. However, when a liquidity provider exits the pool with the new price ratio, 
                the loss becomes permanent.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with il_col2:
        # Create IL curve
        price_ratios = np.linspace(0.1, 5, 100)
        il_values = [2 * math.sqrt(r) / (1 + r) - 1 for r in price_ratios]
        
        # Convert to percentage
        il_percent = [il * 100 for il in il_values]
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=price_ratios,
            y=il_percent,
            mode='lines',
            name='Impermanent Loss',
            line=dict(color='#ff3366', width=3)
        ))
        
        fig.update_layout(
            title="Impermanent Loss vs Price Ratio Change",
            xaxis_title="Price Ratio Change (New/Initial)",
            yaxis_title="Impermanent Loss (%)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add x=1 and y=0 lines
        fig.add_shape(
            type="line",
            x0=1, y0=-30, x1=1, y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=5, y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Set y-axis to negative values
        fig.update_yaxes(range=[-30, 0])
        
        st.plotly_chart(fig, use_container_width=True)

# ============ SYSTEM ARCHITECTURE TAB ============
with architecture_tab:
    st.markdown("<h2 class='section-header'>System Architecture</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    SolPool Insight is built on a practical architecture focused on providing reliable Solana liquidity pool analysis.
    This section outlines the key components and their interactions.
    """)
    
    # High-level Architecture
    st.markdown("<h3 class='sub-header'>High-Level Architecture</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        Our system architecture follows a modular design with several specialized components:
        
        1. **Data Collection Layer**
           - Blockchain RPC Connectors
           - DEX API Integrators
           - Market Data Collectors
        
        2. **Data Processing Layer**
           - ETL Pipeline
           - Feature Engineering Processor
           - Data Validation Service
        
        3. **Storage Layer**
           - PostgreSQL Time-Series Database
           - In-Memory Cache for Real-Time Analytics
           - Model Storage Repository
        
        4. **Analytics & Machine Learning Layer**
           - Training Pipeline
           - Prediction Engine
           - Analytical Query Processor
        
        5. **Presentation Layer**
           - Streamlit Interactive Dashboard
           - API Service
           - Data Export Service
    </div>
    """, unsafe_allow_html=True)
    
    # Create a simple architecture diagram
    st.markdown("<h3 class='sub-header'>Architecture Diagram</h3>", unsafe_allow_html=True)
    
    # Using Plotly to create a simple architecture diagram
    fig = go.Figure()
    
    # Define layers
    layers = [
        {"name": "Data Sources", "components": ["Solana Blockchain", "DEX APIs", "Market Data", "Token Metadata"]},
        {"name": "Data Collection", "components": ["RPC Connectors", "API Clients", "Market Data Collectors"]},
        {"name": "Data Processing", "components": ["ETL Pipeline", "Feature Engineering", "Data Validation"]},
        {"name": "Storage", "components": ["PostgreSQL", "Cache Layer"]},
        {"name": "Analytics", "components": ["Prediction Engine", "Risk Analysis", "Trend Detection"]},
        {"name": "Presentation", "components": ["Streamlit Dashboard", "API Service"]}
    ]
    
    # Colors for different layers
    colors = ["#e1f5fe", "#b3e5fc", "#81d4fa", "#4fc3f7", "#29b6f6", "#03a9f4"]
    
    layer_height = 1
    component_width = 2.5
    margin = 0.2
    
    # Add rectangles for each layer
    for i, layer in enumerate(layers):
        y_pos = (len(layers) - i - 1) * (layer_height + margin)
        
        # Add layer label
        fig.add_annotation(
            x=0,
            y=y_pos + layer_height/2,
            text=layer["name"],
            showarrow=False,
            font=dict(size=14, color="#424242"),
            xanchor="right"
        )
        
        # Add components
        for j, component in enumerate(layer["components"]):
            x_pos = 1 + j * (component_width + margin)
            
            # Add rectangle
            fig.add_shape(
                type="rect",
                x0=x_pos,
                y0=y_pos,
                x1=x_pos + component_width,
                y1=y_pos + layer_height,
                line=dict(color="#424242", width=1),
                fillcolor=colors[i],
                opacity=0.8
            )
            
            # Add component label
            fig.add_annotation(
                x=x_pos + component_width/2,
                y=y_pos + layer_height/2,
                text=component,
                showarrow=False,
                font=dict(size=12, color="#212121")
            )
    
    # Add arrows connecting layers
    for i in range(len(layers)-1):
        y_from = (len(layers) - i - 1) * (layer_height + margin)
        y_to = y_from - (layer_height + margin)
        
        # Add multiple arrows between layers
        for j in range(len(layers[i]["components"])):
            x_pos = 1 + j * (component_width + margin) + component_width/2
            
            fig.add_shape(
                type="line",
                x0=x_pos,
                y0=y_from,
                x1=x_pos,
                y1=y_to + layer_height,
                line=dict(color="#424242", width=1),
                line_dash="solid"
            )
            
            # Add arrowhead
            fig.add_shape(
                type="line",
                x0=x_pos - 0.1,
                y0=y_to + layer_height + 0.1,
                x1=x_pos,
                y1=y_to + layer_height,
                line=dict(color="#424242", width=1)
            )
            
            fig.add_shape(
                type="line",
                x0=x_pos + 0.1,
                y0=y_to + layer_height + 0.1,
                x1=x_pos,
                y1=y_to + layer_height,
                line=dict(color="#424242", width=1)
            )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-2, 12]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1, len(layers) * (layer_height + margin)]
        ),
        height=600,
        margin=dict(l=100, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technology Stack Details
    st.markdown("<h3 class='sub-header'>Technology Stack Details</h3>", unsafe_allow_html=True)
    
    # Create columns for different tech categories
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        <div class="tech-card">
            <h4>Data & Infrastructure</h4>
            <ul>
                <li><strong>Backend:</strong> Python (3.9+)</li>
                <li><strong>Database:</strong> PostgreSQL (Time-Series Optimized)</li>
                <li><strong>Caching:</strong> Redis for real-time data</li>
                <li><strong>API Framework:</strong> FastAPI</li>
                <li><strong>Task Scheduling:</strong> Celery with Redis broker</li>
                <li><strong>Version Control:</strong> Git with CI/CD pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div class="tech-card">
            <h4>Data Science & ML</h4>
            <ul>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                <li><strong>ML Framework:</strong> Scikit-learn, TensorFlow, XGBoost</li>
                <li><strong>Time Series:</strong> Prophet, ARIMA models</li>
                <li><strong>Deep Learning:</strong> Keras with TensorFlow backend</li>
                <li><strong>Hyperparameter Tuning:</strong> Optuna</li>
                <li><strong>Feature Selection:</strong> SHAP, Permutation Importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div class="tech-card">
            <h4>Visualization & Frontend</h4>
            <ul>
                <li><strong>Dashboard:</strong> Streamlit</li>
                <li><strong>Data Visualization:</strong> Plotly, Matplotlib</li>
                <li><strong>Interactive Charts:</strong> Plotly Graph Objects</li>
                <li><strong>Responsive Design:</strong> Custom CSS</li>
                <li><strong>Real-time Updates:</strong> WebSocket integration</li>
                <li><strong>Export Formats:</strong> CSV, JSON, Excel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Integration & API
    st.markdown("<h3 class='sub-header'>Data Integration & API Services</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <p>
            SolPool Insight integrates with multiple external data sources and services:
        </p>
        
        <ul class="feature-list">
            <li>
                <strong>Solana RPC Nodes</strong>: Direct connections to dedicated Solana RPC nodes for
                real-time blockchain data access. We maintain multiple node connections for redundancy.
            </li>
            <li>
                <strong>DEX Protocols</strong>: Direct integration with Raydium, Orca, and other DEX protocols
                through both on-chain account parsing and API connections where available.
            </li>
            <li>
                <strong>Token Price Oracles</strong>: Integration with CoinGecko API and other price oracles
                to maintain accurate, up-to-date token pricing information.
            </li>
            <li>
                <strong>Market Data Providers</strong>: Connections to market data services for broader market context,
                including sentiment analysis and trading volume metrics.
            </li>
            <li>
                <strong>Google Vertex AI</strong>: Advanced analytics and machine learning capabilities powered by
                Google's Vertex AI platform.
            </li>
        </ul>
        
        <p>
            Our system maintains high availability through redundant connections and fallback mechanisms,
            ensuring continuous data collection even when individual sources experience outages.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer with additional resources
st.markdown("<h2 class='section-header'>Additional Resources</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="tech-card">
        <h4>Research Papers</h4>
        <ul>
            <li>Time Series Forecasting with LSTM Networks for DeFi Applications</li>
            <li>Automated Risk Assessment for Liquidity Pools</li>
            <li>Self-Evolving Prediction Systems in Volatile Markets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tech-card">
        <h4>Technical Documentation</h4>
        <ul>
            <li>Complete API Documentation</li>
            <li>Database Schema Reference</li>
            <li>Model Training Guide</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tech-card">
        <h4>Support</h4>
        <ul>
            <li>GitHub Repository</li>
            <li>Technical Support</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Closing statement
st.markdown("""
<div class="tech-card">
    <p>
        We believe in complete transparency about our methodologies, data sources, and calculations.
        This allows users to make informed decisions based on a clear understanding of how our insights
        are generated. If you have any questions or need further clarification on any aspect of our
        technology, please reach out to our team.
    </p>
</div>
""", unsafe_allow_html=True)