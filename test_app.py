import streamlit as st
import os

# Configure the page
st.set_page_config(
    page_title="SolPool Insight - Test App",
    page_icon="🌊",
    layout="wide"
)

# Main header
st.title("SolPool Insight - Database Test")
st.write("This app verifies that the database connection is working properly.")

# Try to get database environment variables
try:
    database_url = os.environ.get("DATABASE_URL", "Not found")
    
    # Mask the database URL for security
    if database_url != "Not found":
        parts = database_url.split("@")
        if len(parts) > 1:
            masked_url = parts[0].split(":")[0] + ":******@" + parts[1]
        else:
            masked_url = "Invalid format"
    else:
        masked_url = "Not found"
        
    # Display the masked URL
    st.write(f"Database URL: {masked_url}")
    
    # Check other PostgreSQL variables
    pg_vars = {
        "PGHOST": os.environ.get("PGHOST", "Not found"),
        "PGPORT": os.environ.get("PGPORT", "Not found"),
        "PGUSER": os.environ.get("PGUSER", "Not found"),
        "PGDATABASE": os.environ.get("PGDATABASE", "Not found"),
        "PGPASSWORD": "[MASKED]" if "PGPASSWORD" in os.environ else "Not found"
    }
    
    st.write("PostgreSQL Environment Variables:")
    st.json(pg_vars)
    
    # Attempt a database connection
    st.write("Attempting to connect to the database...")
    
    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        cursor.close()
        conn.close()
        
        st.success(f"✅ Successfully connected to the database!")
        st.write(f"Database version: {db_version[0]}")
    except Exception as e:
        st.error(f"❌ Failed to connect to the database: {str(e)}")
        st.write("Please check your database configuration.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Add some interactive elements
st.header("Interactive Test")
if st.button("Click me!"):
    st.success("Button clicked successfully!")

st.write("This test app helps verify that both Streamlit and the database connection are working properly.")
