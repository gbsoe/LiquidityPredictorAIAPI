import streamlit as st
import pandas as pd
import db_handler
import json

st.title("SolPool Insight - 100 Pools Test")

# Attempt to get 100 pools from database
st.subheader("Retrieving 100 Pools from Database")

try:
    # Get 100 pools from database
    pools = db_handler.get_pools(limit=100)
    
    if pools and len(pools) > 0:
        st.success(f"✓ Successfully retrieved {len(pools)} pools from database")
        
        # Convert to DataFrame for better display
        pools_df = pd.DataFrame(pools)
        
        # Display the pools in a table
        st.subheader("Pool Data Overview")
        
        # Select specific columns to display in a more organized way
        display_columns = [
            "name", "dex", "category", "token1_symbol", "token2_symbol",
            "liquidity", "volume_24h", "apr", "fee"
        ]
        
        # Format numeric columns
        if 'liquidity' in pools_df.columns:
            pools_df['liquidity'] = pools_df['liquidity'].apply(lambda x: f"${x:,.2f}")
        if 'volume_24h' in pools_df.columns:
            pools_df['volume_24h'] = pools_df['volume_24h'].apply(lambda x: f"${x:,.2f}")
        if 'apr' in pools_df.columns:
            pools_df['apr'] = pools_df['apr'].apply(lambda x: f"{x:.2f}%")
        if 'fee' in pools_df.columns:
            pools_df['fee'] = pools_df['fee'].apply(lambda x: f"{x*100:.2f}%")
            
        # Show the table
        st.dataframe(pools_df[display_columns])
        
        # Show raw data if needed (expandable)
        with st.expander("View Raw JSON Data"):
            st.json(json.dumps(pools, indent=2))
            
    else:
        st.warning("No pools found in the database.")
        st.info("Let's check if the database schema exists and create it if needed.")
        
        # Try to initialize the database schema
        schema_created = db_handler.init_db()
        if schema_created:
            st.success("Database schema created successfully!")
        else:
            st.error("Failed to create database schema.")

except Exception as e:
    st.error(f"Error retrieving pools: {str(e)}")
    import traceback
    st.code(traceback.format_exc(), language="python")

# Add instructions for next steps
st.subheader("Next Steps")
st.markdown("""
If pools were successfully retrieved:
- Examine the data to ensure it contains valid pool information
- Look at available fields and their values

If no pools were found:
- The database might not be populated yet
- We may need to implement data fetching from blockchain APIs
""")