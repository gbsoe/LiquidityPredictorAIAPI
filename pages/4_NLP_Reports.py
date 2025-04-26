import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import json

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nlp_generator import NLPReportGenerator
from utils.data_processor import get_pool_list, get_pool_metrics, get_pool_details, get_token_prices
from database.db_operations import DBManager

# Page configuration
st.set_page_config(
    page_title="NLP Reports - Solana Liquidity Pool Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fancy title with gradient
st.markdown("""
<style>
.nlp-title {
    background: linear-gradient(90deg, #9333ea, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 12px;
}
.nlp-subtitle {
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 20px;
}
.report-container {
    padding: 20px;
    border-radius: 10px;
    background-color: #f8f9fa;
    border-left: 5px solid #4f46e5;
    margin-bottom: 20px;
}
.report-header {
    font-weight: 600;
    color: #4f46e5;
    margin-bottom: 12px;
}
.report-meta {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 10px;
}
.report-content {
    line-height: 1.6;
    white-space: pre-line;
}
.api-key-notice {
    padding: 15px;
    border-radius: 8px;
    background-color: #fffbd5;
    border-left: 4px solid #f59e0b;
    margin-bottom: 15px;
}
@media (prefers-color-scheme: dark) {
    .report-container {
        background-color: #1e1e1e;
        border-left: 5px solid #6366f1;
    }
    .report-header {
        color: #818cf8;
    }
    .report-meta {
        color: #9ca3af;
    }
    .api-key-notice {
        background-color: #3f3822;
        border-left: 4px solid #f59e0b;
    }
}
</style>
<div class="nlp-title">AI-Generated Market Intelligence</div>
<div class="nlp-subtitle">Advanced NLP analysis of Solana liquidity pool data and predictions</div>
""", unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
def get_db_connection():
    return DBManager()

db = get_db_connection()

# Initialize NLP generator
@st.cache_resource
def get_nlp_generator():
    return NLPReportGenerator()

nlp_generator = get_nlp_generator()

# Add FiLot logo to sidebar
st.sidebar.image("static/filot_logo_new.png", width=130)
st.sidebar.markdown("### FiLot Analytics")
st.sidebar.markdown("---")

# API key management
if not nlp_generator.has_api_key():
    st.sidebar.markdown("### OpenAI API Key Required")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    if st.sidebar.button("Save API Key"):
        try:
            # Save to .env file for persistence
            with open(".env", "a+") as f:
                f.seek(0)
                content = f.read()
                if "OPENAI_API_KEY" not in content:
                    f.write(f"\nOPENAI_API_KEY={api_key}\n")
                else:
                    # Replace existing API key
                    lines = content.split('\n')
                    with open(".env", "w") as f_write:
                        for line in lines:
                            if line.startswith("OPENAI_API_KEY="):
                                f_write.write(f"OPENAI_API_KEY={api_key}\n")
                            else:
                                f_write.write(f"{line}\n")
            
            st.sidebar.success("API key saved! Please refresh the page.")
            
            # Set environment variable for current session
            os.environ["OPENAI_API_KEY"] = api_key
        except Exception as e:
            st.sidebar.error(f"Error saving API key: {e}")
    
    st.markdown("""
    <div class="api-key-notice">
        <strong>OpenAI API Key Required</strong><br>
        To generate NLP reports, please enter your OpenAI API key in the sidebar.
        Your API key will be stored securely in your .env file.
    </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["Market Report", "Pool Analysis", "Custom Analysis"])

# Tab 1: Market Report
with tab1:
    st.header("Solana Liquidity Pool Market Report")
    st.markdown("Generate comprehensive market reports with AI insights on current trends and opportunities.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Report Settings")
        include_predictions = st.toggle("Include Predictions", value=True)
        
        # Safe select_slider implementation with static options
        try:
            report_detail = st.select_slider(
                "Report Detail",
                options=["Concise", "Standard", "Detailed"],
                value="Standard"
            )
        except Exception as e:
            st.warning(f"Error with report detail slider: {e}")
            report_detail = "Standard"  # Default value
        
        if st.button("Generate Market Report", type="primary", use_container_width=True):
            if not nlp_generator.has_api_key():
                st.warning("OpenAI API key is required. Please enter it in the sidebar.")
            else:
                with st.spinner("Generating comprehensive market report..."):
                    # Load all pools data
                    pools_data = get_pool_list(db)
                    
                    # Get top predictions if requested
                    top_predictions = []
                    if include_predictions:
                        prediction_df = db.get_top_predictions(category="apr", limit=10)
                        if not isinstance(prediction_df, list) and hasattr(prediction_df, 'to_dict'):
                            top_predictions = prediction_df.to_dict('records')
                        else:
                            top_predictions = prediction_df
                        
                    # Convert pools_data to list of dictionaries
                    pools_dict_list = pools_data.to_dict('records') if not pools_data.empty else []
                    
                    # Generate report
                    report = nlp_generator.generate_market_report(pools_dict_list, top_predictions)
                    
                    if report:
                        # Save report
                        report_path = nlp_generator.save_report(report, "market_report")
                        
                        # Display report
                        st.markdown(f"""
                        <div class="report-container">
                            <div class="report-header">Solana DeFi Market Report</div>
                            <div class="report-meta">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} • Detail level: {report_detail}</div>
                            <div class="report-content">{report}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show download link
                        if report_path:
                            with open(report_path, 'r') as f:
                                st.download_button(
                                    label="Download Report",
                                    data=f.read(),
                                    file_name=f"solana_market_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.error("Failed to generate report. Please check your API key and try again.")
    
    with col1:
        # Show list of previously generated reports if they exist
        reports_dir = "reports"
        if os.path.exists(reports_dir):
            market_reports = [f for f in os.listdir(reports_dir) if f.startswith("market_report_")]
            
            if market_reports:
                st.markdown("### Previously Generated Reports")
                
                for report_file in sorted(market_reports, reverse=True)[:5]:
                    report_path = os.path.join(reports_dir, report_file)
                    # Extract timestamp from filename
                    try:
                        timestamp_str = report_file.split("_")[2:4]
                        timestamp = datetime.strptime(f"{timestamp_str[0]}_{timestamp_str[1].split('.')[0]}", "%Y%m%d_%H%M%S")
                        date_str = timestamp.strftime("%B %d, %Y at %H:%M")
                    except:
                        date_str = "Unknown date"
                    
                    with open(report_path, 'r') as f:
                        report_preview = f.read()[:200] + "..."
                    
                    with st.expander(f"Market Report - {date_str}"):
                        st.markdown(f"""
                        <div class="report-preview">
                            {report_preview}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with open(report_path, 'r') as f:
                            st.download_button(
                                label="Download Full Report",
                                data=f.read(),
                                file_name=f"solana_market_report_{timestamp.strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                key=f"download_{report_file}"
                            )

# Tab 2: Pool Analysis
with tab2:
    st.header("Individual Pool Analysis")
    st.markdown("Generate in-depth analysis for specific liquidity pools with AI-powered insights.")
    
    # Load pool list
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_pool_list():
        return get_pool_list(db)
    
    try:
        pool_list = load_pool_list()
        
        if not pool_list.empty:
            # Create pool selection options
            pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
            
            selected_pool_option = st.selectbox(
                "Select a pool for detailed analysis:",
                options=pool_options
            )
            
            # Extract pool_id from selection
            selected_pool_id = selected_pool_option.split("(")[-1].split(")")[0]
            
            # Generate analysis button
            if st.button("Generate Pool Analysis", type="primary"):
                if not nlp_generator.has_api_key():
                    st.warning("OpenAI API key is required. Please enter it in the sidebar.")
                else:
                    with st.spinner("Generating detailed pool analysis..."):
                        # Get pool details
                        pool_details = get_pool_details(db, selected_pool_id)
                        
                        if pool_details is not None:
                            # Generate analysis
                            analysis = nlp_generator.generate_pool_summary(pool_details)
                            
                            if analysis:
                                # Save analysis
                                pool_name = pool_details.get('name', 'unknown_pool')
                                analysis_path = nlp_generator.save_report(analysis, f"pool_analysis_{pool_name.replace('/', '_')}")
                                
                                # Display analysis
                                st.markdown(f"""
                                <div class="report-container">
                                    <div class="report-header">Analysis: {pool_details.get('name', 'Unknown Pool')}</div>
                                    <div class="report-meta">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}</div>
                                    <div class="report-content">{analysis}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error("Failed to generate analysis. Please check your API key and try again.")
                        else:
                            st.error(f"Error loading details for pool: {selected_pool_id}")
            
            # Display pool metrics
            if selected_pool_id:
                pool_details = get_pool_details(db, selected_pool_id)
                
                if pool_details is not None:
                    st.subheader("Pool Metrics")
                    
                    # Format metrics for display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Liquidity", f"${pool_details.get('liquidity', 0):,.2f}")
                    
                    with col2:
                        st.metric("24h Volume", f"${pool_details.get('volume_24h', 0):,.2f}")
                    
                    with col3:
                        st.metric("APR", f"{pool_details.get('apr', 0):.2f}%")
                    
                    with col4:
                        st.metric("Category", pool_details.get('category', 'Unknown'))
                    
                    # Display additional metrics if available
                    if 'tvl_change_24h' in pool_details and 'apr_change_24h' in pool_details:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("TVL Change (24h)", f"{pool_details.get('tvl_change_24h', 0):.2f}%",
                                     delta=pool_details.get('tvl_change_24h', 0))
                        
                        with col2:
                            st.metric("APR Change (24h)", f"{pool_details.get('apr_change_24h', 0):.2f}%",
                                     delta=pool_details.get('apr_change_24h', 0))
        else:
            st.warning("No pools available for analysis.")
    except Exception as e:
        st.error(f"Error loading pool data: {str(e)}")

# Tab 3: Custom Analysis
with tab3:
    st.header("Custom Analysis Request")
    st.markdown("Ask specific questions about Solana liquidity pools and get AI-powered analysis.")
    
    # User query input
    user_query = st.text_area(
        "Enter your analysis question:",
        placeholder="Example: What are the key factors affecting APR in BONK/USDC pools over the last week?",
        height=100
    )
    
    # Pool selection for context
    st.subheader("Select Data Context")
    context_option = st.radio(
        "Choose data to include in analysis:",
        ["All pools", "Specific category", "Specific pool"],
        horizontal=True
    )
    
    # Context selection based on user choice
    if context_option == "Specific category":
        # Get unique categories
        try:
            pool_list = load_pool_list()
            categories = pool_list['category'].unique().tolist()
            selected_category = st.selectbox("Select category:", options=categories)
        except:
            selected_category = None
            st.warning("Could not load categories.")
    
    elif context_option == "Specific pool":
        try:
            pool_list = load_pool_list()
            pool_options = [f"{row['name']} ({row['pool_id']})" for _, row in pool_list.iterrows()]
            selected_pool = st.selectbox("Select pool:", options=pool_options)
            selected_pool_id = selected_pool.split("(")[-1].split(")")[0]
        except:
            selected_pool_id = None
            st.warning("Could not load pools.")
    
    # Generate analysis button
    if st.button("Generate Custom Analysis", type="primary"):
        if not user_query:
            st.warning("Please enter a question for analysis.")
        elif not nlp_generator.has_api_key():
            st.warning("OpenAI API key is required. Please enter it in the sidebar.")
        else:
            with st.spinner("Generating custom analysis..."):
                # Prepare context data based on selection
                context_data = {}
                
                try:
                    if context_option == "All pools":
                        pool_list = get_pool_list(db)
                        context_data = {
                            "pools": pool_list.to_dict('records') if not pool_list.empty else [],
                            "context_type": "all_pools"
                        }
                    
                    elif context_option == "Specific category" and selected_category:
                        pool_list = get_pool_list(db)
                        category_pools = pool_list[pool_list['category'] == selected_category]
                        context_data = {
                            "pools": category_pools.to_dict('records') if not category_pools.empty else [],
                            "context_type": "category",
                            "category": selected_category
                        }
                    
                    elif context_option == "Specific pool" and selected_pool_id:
                        pool_details = get_pool_details(db, selected_pool_id)
                        # Get historical metrics too
                        pool_metrics = get_pool_metrics(db, selected_pool_id, days=7)
                        
                        context_data = {
                            "pool_details": pool_details,
                            "pool_metrics": pool_metrics.to_dict('records') if not pool_metrics.empty else [],
                            "context_type": "single_pool"
                        }
                    
                    # Generate analysis
                    analysis = nlp_generator.generate_specific_analysis(user_query, context_data)
                    
                    if analysis:
                        # Save analysis
                        query_summary = user_query[:20].replace(" ", "_")
                        analysis_path = nlp_generator.save_report(analysis, f"custom_analysis_{query_summary}")
                        
                        # Display analysis
                        st.markdown(f"""
                        <div class="report-container">
                            <div class="report-header">Custom Analysis</div>
                            <div class="report-meta">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}</div>
                            <div class="report-meta">Query: "{user_query}"</div>
                            <div class="report-content">{analysis}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show download link
                        if analysis_path:
                            with open(analysis_path, 'r') as f:
                                st.download_button(
                                    label="Download Analysis",
                                    data=f.read(),
                                    file_name=f"custom_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.error("Failed to generate analysis. Please check your API key and try again.")
                
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>Solana Liquidity Pool Analysis System • NLP Reports</div>
    <div style="font-size: 12px; color: #6b7280;">Powered by OpenAI GPT-4</div>
</div>
""", unsafe_allow_html=True)

# Create reports directory if it doesn't exist
os.makedirs('reports', exist_ok=True)