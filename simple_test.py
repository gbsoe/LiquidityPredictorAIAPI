import streamlit as st
import psycopg2
import os

# Set page config
st.set_page_config(
    page_title="Simple Database Test",
    page_icon="🧪",
    layout="wide"
)

st.title("PostgreSQL Database Connection Test")

# Display environment variables (without showing sensitive values)
def get_masked_env_var(var_name):
    value = os.environ.get(var_name, "Not set")
    if value != "Not set" and len(value) > 8:
        return value[:4] + "*" * (len(value) - 8) + value[-4:]
    return "Not set"

st.subheader("Database Environment Variables")
col1, col2 = st.columns(2)

with col1:
    st.write("DATABASE_URL: ", get_masked_env_var("DATABASE_URL"))
    st.write("PGHOST: ", get_masked_env_var("PGHOST"))
    st.write("PGPORT: ", get_masked_env_var("PGPORT"))

with col2:
    st.write("PGDATABASE: ", get_masked_env_var("PGDATABASE"))
    st.write("PGUSER: ", get_masked_env_var("PGUSER"))
    st.write("PGPASSWORD: ", get_masked_env_var("PGPASSWORD"))

# Test database connection
st.subheader("Database Connection Test")

try:
    # Get database connection parameters from environment variables
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url:
        st.info(f"Trying to connect using DATABASE_URL...")
        conn = psycopg2.connect(db_url)
    else:
        # Fall back to individual connection parameters
        st.info(f"Trying to connect using individual parameters...")
        conn = psycopg2.connect(
            host=os.environ.get("PGHOST"),
            port=os.environ.get("PGPORT"),
            database=os.environ.get("PGDATABASE"),
            user=os.environ.get("PGUSER"),
            password=os.environ.get("PGPASSWORD")
        )
    
    st.success("✅ Successfully connected to PostgreSQL database!")
    
    # Get database info
    cursor = conn.cursor()
    
    # Get PostgreSQL version
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    st.write("PostgreSQL Version:", version[0])
    
    # List tables in the database
    st.subheader("Database Tables")
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    
    if tables:
        table_list = [table[0] for table in tables]
        st.write("Tables in database:", ", ".join(table_list))
        
        # Allow user to select a table to view sample data
        selected_table = st.selectbox("Select a table to view sample data", table_list)
        
        if selected_table:
            st.subheader(f"Sample data from '{selected_table}'")
            try:
                # Get column names
                cursor.execute(f"SELECT * FROM {selected_table} LIMIT 0")
                colnames = [desc[0] for desc in cursor.description]
                
                # Get sample data (first 5 rows)
                cursor.execute(f"SELECT * FROM {selected_table} LIMIT 5")
                rows = cursor.fetchall()
                
                # Display as a table
                if rows:
                    # Create a list of dictionaries for each row
                    data = [dict(zip(colnames, row)) for row in rows]
                    st.write(f"Showing first {len(rows)} rows:")
                    st.table(data)
                else:
                    st.info(f"Table '{selected_table}' is empty.")
            except Exception as e:
                st.error(f"Error querying table: {e}")
    else:
        st.info("No tables found in the database.")
    
    # Close the connection
    cursor.close()
    conn.close()
    
except Exception as e:
    st.error(f"❌ Failed to connect to database: {e}")
    st.info("Please check your database credentials and connection.")

# Simple form to test data insertion
st.subheader("Test Data Insertion")
with st.form("data_insertion_form"):
    test_table = st.text_input("Table name", "test_table")
    test_name = st.text_input("Name", "Test Name")
    test_value = st.number_input("Value", value=42)
    
    submitted = st.form_submit_button("Insert Test Data")
    
    if submitted:
        try:
            # Connect to the database
            db_url = os.environ.get("DATABASE_URL")
            if db_url:
                conn = psycopg2.connect(db_url)
            else:
                conn = psycopg2.connect(
                    host=os.environ.get("PGHOST"),
                    port=os.environ.get("PGPORT"),
                    database=os.environ.get("PGDATABASE"),
                    user=os.environ.get("PGUSER"),
                    password=os.environ.get("PGPASSWORD")
                )
            
            cursor = conn.cursor()
            
            # Check if table exists, create if not
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {test_table} (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    value INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert data
            cursor.execute(f"""
                INSERT INTO {test_table} (name, value)
                VALUES (%s, %s)
            """, (test_name, test_value))
            
            # Commit and close
            conn.commit()
            cursor.close()
            conn.close()
            
            st.success(f"Successfully inserted data into {test_table}!")
        except Exception as e:
            st.error(f"Failed to insert data: {e}")

# Footer
st.markdown("---")
st.markdown("Database connection test app for SolPool Insight")
