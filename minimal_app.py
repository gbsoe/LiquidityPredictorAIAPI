import streamlit as st
import os
import traceback

st.title('SolPool Insight - Database Connectivity Test')

# Display database connection information
st.write('Testing database connectivity...')

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
    
    st.write(f'Database URL: {masked_url}')
    
    # Test connection
    try:
        import psycopg2
        
        # Attempt to connect
        st.write('Connecting to PostgreSQL...')
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Execute test query
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        # Display success message
        st.success(f'✅ Successfully connected to PostgreSQL!')
        st.write(f'Database version: {version}')
        
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
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        # Display table information
        if has_pools_table:
            st.write('✅ The pools table exists!')
        else:
            st.warning('⚠️ The pools table does not exist. Run init_database.py to create it.')
            
        if has_liquidity_pools_table:
            st.write('✅ The liquidity_pools table exists!')
        else:
            st.warning('⚠️ The liquidity_pools table does not exist. Run init_database.py to create it.')
        
    except Exception as e:
        st.error(f'❌ Failed to connect to the database: {str(e)}')
        st.code(traceback.format_exc())
        st.write('Please check your database configuration or run init_database.py to initialize the database.')
        
except Exception as e:
    st.error(f'Error checking database connection: {str(e)}')
    st.code(traceback.format_exc())

# Show a button to run database initialization script
if st.button('Initialize Database'):
    try:
        import subprocess
        result = subprocess.run(['python', 'init_database.py'], capture_output=True, text=True)
        st.write('Database initialization output:')
        st.code(result.stdout)
        
        if result.returncode == 0:
            st.success('✅ Database initialized successfully!')
        else:
            st.error(f'❌ Database initialization failed with exit code {result.returncode}')
            st.code(result.stderr)
    except Exception as e:
        st.error(f'Error running database initialization: {str(e)}')
        st.code(traceback.format_exc())

# Next steps
st.subheader('Next Steps')
st.write('Once the database connection is working properly, we can continue troubleshooting the main application.')
st.write('Click the button below to restart the minimal app with the updated database connection.')

if st.button('Restart App'):
    st.rerun()
