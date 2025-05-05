import streamlit as st
import os
import traceback
import pandas as pd

# Import our simple pool data module
from simple_pool_data import get_pool_data, insert_sample_pool_data

# Configure the page
st.set_page_config(
    page_title="SolPool Insight - Basic Version",
    page_icon="🌊",
    layout="wide"
)

# Main title
st.title('SolPool Insight - Basic Version')
st.write('This simplified version of SolPool Insight connects directly to the database.')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Database Connection", "Pool Explorer", "Data Management"])

# Function to format currency
def format_currency(value):
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

# Function to format percentage
def format_percentage(value):
    return f"{value:.2f}%"

# Database Connection page
if page == "Database Connection":
    st.header("Database Connection Test")
    st.write("Testing database connectivity...")

    try:
        # Get database URL
        db_url = os.environ.get('DATABASE_URL', 'Not found')
        if db_url != 'Not found':
            # Mask sensitive parts of the URL
            parts = db_url.split('@')
            if len(parts) > 1:
                masked_url = parts[0].split(':')[0] + ':******@' + parts[1]
            else:
                masked_url = 'Invalid URL format'
        else:
            masked_url = 'Not found'
        
        st.write(f"Database URL: {masked_url}")
        
        # Test connection
        try:
            import psycopg2
            
            # Attempt to connect
            st.write("Connecting to PostgreSQL...")
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Execute test query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            # Display success message
            st.success(f"✅ Successfully connected to PostgreSQL!")
            st.write(f"Database version: {version}")
            
            # Try to check if our tables exist
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Check if the pools table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'pools'
                );
            """)
            has_pools_table = cursor.fetchone()[0]
            
            # Check if the liquidity_pools table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'liquidity_pools'
                );
            """)
            has_liquidity_pools_table = cursor.fetchone()[0]
            
            # Check if the pool_metrics table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'pool_metrics'
                );
            """)
            has_pool_metrics_table = cursor.fetchone()[0]
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            # Display table information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if has_pools_table:
                    st.write("✅ The pools table exists!")
                else:
                    st.warning("⚠️ The pools table does not exist.")
                
            with col2:
                if has_liquidity_pools_table:
                    st.write("✅ The liquidity_pools table exists!")
                else:
                    st.warning("⚠️ The liquidity_pools table does not exist.")
                    
            with col3:
                if has_pool_metrics_table:
                    st.write("✅ The pool_metrics table exists!")
                else:
                    st.warning("⚠️ The pool_metrics table does not exist.")
            
        except Exception as e:
            st.error(f"❌ Failed to connect to the database: {str(e)}")
            st.code(traceback.format_exc())
            st.write("Please check your database configuration or run init_database.py to initialize the database.")
            
    except Exception as e:
        st.error(f"Error checking database connection: {str(e)}")
        st.code(traceback.format_exc())

    # Show a button to run database initialization script
    if st.button("Re-Initialize Database"):
        try:
            import subprocess
            result = subprocess.run(['python', 'init_database.py'], capture_output=True, text=True)
            st.write("Database initialization output:")
            st.code(result.stdout)
            
            if result.returncode == 0:
                st.success("✅ Database initialized successfully!")
            else:
                st.error(f"❌ Database initialization failed with exit code {result.returncode}")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"Error running database initialization: {str(e)}")
            st.code(traceback.format_exc())

# Pool Explorer page
elif page == "Pool Explorer":
    st.header("Pool Explorer")
    st.write("This page displays liquidity pool data from the database.")
    
    # Check if data exists in the database and add sample data if needed
    with st.expander("Data Status"):
        try:
            import psycopg2
            db_url = os.environ.get('DATABASE_URL')
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM pools;")
            pool_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pool_metrics;")
            metrics_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            st.write(f"Pools in database: {pool_count}")
            st.write(f"Pool metrics in database: {metrics_count}")
            
            if pool_count == 0 or metrics_count == 0:
                st.warning("No pool data found in the database. Add sample data for testing.")
                if st.button("Add Sample Pool Data"):
                    success = insert_sample_pool_data()
                    if success:
                        st.success("✅ Sample pool data added successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add sample pool data.")
        except Exception as e:
            st.error(f"Error checking data status: {str(e)}")
    
    # Get and display pool data
    try:
        pool_df = get_pool_data()
        
        if not pool_df.empty:
            # Add formatting to the dataframe
            display_df = pool_df.copy()
            
            # Format numeric columns
            if 'liquidity' in display_df.columns:
                display_df['liquidity'] = display_df['liquidity'].apply(format_currency)
            if 'volume' in display_df.columns:
                display_df['volume'] = display_df['volume'].apply(format_currency)
            if 'apr' in display_df.columns:
                display_df['apr'] = display_df['apr'].apply(format_percentage)
            
            # Display the data
            st.dataframe(display_df, use_container_width=True)
            
            # Display charts
            st.subheader("Pool Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top Pools by Liquidity")
                
                # Prepare data for chart
                chart_df = pool_df.sort_values('liquidity', ascending=False).head(5)
                chart_df['pool_name'] = chart_df['name'] + ' (' + chart_df['dex'] + ')'
                
                # Create a bar chart
                st.bar_chart(data=chart_df.set_index('pool_name')['liquidity'])
            
            with col2:
                st.write("Top Pools by APR")
                
                # Prepare data for chart
                chart_df = pool_df.sort_values('apr', ascending=False).head(5)
                chart_df['pool_name'] = chart_df['name'] + ' (' + chart_df['dex'] + ')'
                
                # Create a bar chart
                st.bar_chart(data=chart_df.set_index('pool_name')['apr'])
            
        else:
            st.warning("No pool data available. Please add sample data from the Data Management page.")
    except Exception as e:
        st.error(f"Error loading pool data: {str(e)}")
        st.code(traceback.format_exc())

# Data Management page
elif page == "Data Management":
    st.header("Data Management")
    st.write("This page allows you to manage the database data.")
    
    # Add sample data button
    st.subheader("Add Sample Data")
    st.write("Add sample pool data to the database for testing.")
    
    if st.button("Add Sample Pool Data"):
        success = insert_sample_pool_data()
        if success:
            st.success("✅ Sample pool data added successfully!")
        else:
            st.error("❌ Failed to add sample pool data.")
    
    # Database utilities
    st.subheader("Database Utilities")
    st.write("Various database management utilities.")
    
    # Option to clear all pool data
    if st.button("Clear All Pool Data", type="primary", help="This will delete all pool data from the database."):
        try:
            # Get database URL
            db_url = os.environ.get('DATABASE_URL')
            
            # Connect to the database
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Delete all pool metrics first (due to foreign key constraints)
            cursor.execute("DELETE FROM pool_metrics;")
            
            # Then delete all pools
            cursor.execute("DELETE FROM pools;")
            
            # Commit the changes
            conn.commit()
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            st.success("✅ All pool data has been cleared from the database.")
        except Exception as e:
            st.error(f"Error clearing pool data: {str(e)}")
            st.code(traceback.format_exc())
    
    # Option to reinitialize the database
    if st.button("Reinitialize Database"):
        try:
            import subprocess
            result = subprocess.run(['python', 'init_database.py'], capture_output=True, text=True)
            st.write("Database initialization output:")
            st.code(result.stdout)
            
            if result.returncode == 0:
                st.success("✅ Database initialized successfully!")
            else:
                st.error(f"❌ Database initialization failed with exit code {result.returncode}")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"Error running database initialization: {str(e)}")
            st.code(traceback.format_exc())

# Add a footer
st.markdown("---")
st.write("SolPool Insight - Basic Version. For demonstration and testing purposes only.")

# Add a restart button at the bottom
if st.button("Restart App"):
    st.rerun()
