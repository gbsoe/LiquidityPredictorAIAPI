"""
Watchlist Manager - Batch Pool Watchlist Management for SolPool Insight

This page provides comprehensive tools for managing watchlists, 
including batch operations, import/export functionality, and detailed views.
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import database handler for watchlist operations
import db_handler

# Page configuration
st.set_page_config(
    page_title="Watchlist Manager - SolPool Insight",
    page_icon="📋",
    layout="wide"
)

# Custom CSS for better layout and usability
st.markdown("""
<style>
    .watchlist-item {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: rgba(49, 51, 63, 0.7);
    }
    .watchlist-header {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .pool-item {
        margin: 2px 0;
        padding: 5px;
        border-radius: 3px;
        background-color: rgba(70, 70, 90, 0.2);
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        border-radius: 10px;
    }
    .upload-area {
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def format_currency(value: float) -> str:
    """Format a currency value with appropriate scaling"""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format a percentage value"""
    return f"{value:.2f}%"

def format_apr_badge(apr: float) -> str:
    """Format APR with color-coded badge"""
    if apr >= 100:
        color = "red"
    elif apr >= 50:
        color = "orange"
    elif apr >= 20:
        color = "green"
    elif apr >= 5:
        color = "lightgreen"
    else:
        color = "gray"
    
    return f'<span style="color:{color};font-weight:bold">{format_percentage(apr)}</span>'

def get_dex_color(dex: str) -> str:
    """Get color for a DEX"""
    dex_colors = {
        "Raydium": "#9945FF",
        "Orca": "#7CCCDD",
        "Meteora": "#00C0B3",
        "Jupiter": "#F5A114",
        "Saber": "#5372FE"
    }
    return dex_colors.get(dex, "#AAAAAA")

def get_dex_badge(dex: str) -> str:
    """Format DEX as a colored badge"""
    color = get_dex_color(dex)
    return f'<span style="color:{color};font-weight:bold">{dex}</span>'

def display_pool_card(pool: Dict[str, Any]) -> None:
    """Display a pool as a card with detailed information"""
    if not pool:
        return
    
    # Create a container with styling
    with st.container():
        st.markdown(f"""
        <div class="pool-item">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:bold;font-size:1.1em">{pool['name']}</span>
                <span>{get_dex_badge(pool['dex'])}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:3px">
                <span>Liquidity: {format_currency(pool['liquidity'])}</span>
                <span>APR: {format_apr_badge(pool['apr'])}</span>
                <span>Vol(24h): {format_currency(pool['volume_24h'])}</span>
            </div>
            <div style="font-size:0.8em;margin-top:3px">
                ID: <code>{pool['id']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

def add_batch_pools(watchlist_id: int, pool_ids: List[str]) -> int:
    """
    Add multiple pools to a watchlist
    
    Args:
        watchlist_id: ID of the watchlist
        pool_ids: List of pool IDs to add
        
    Returns:
        Number of pools added
    """
    added = 0
    
    # Track IDs that aren't in our database for reporting
    unknown_pools = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pool_id in enumerate(pool_ids):
        # Update progress
        progress = (i + 1) / len(pool_ids)
        progress_bar.progress(progress)
        status_text.text(f"Processing pool {i+1} of {len(pool_ids)}: {pool_id[:10]}...")
        
        # Check if pool exists in our database
        pool_exists = pool_id in pools_df["id"].values if not pools_df.empty else False
        
        if not pool_exists:
            unknown_pools.append(pool_id)
            
        # Add to watchlist (will create placeholder if needed)
        if db_handler.add_pool_to_watchlist(watchlist_id, pool_id):
            added += 1
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Report on unknown pools
    if unknown_pools:
        st.info(f"{len(unknown_pools)} pools were not in the database and added as placeholders. They will be populated with data on next refresh.")
    
    return added

def get_pool_suggestions(pools_df: pd.DataFrame, max_count: int = 5) -> List[Dict[str, Any]]:
    """Generate pool suggestions based on various criteria"""
    suggestions = []
    
    if pools_df.empty:
        return suggestions
    
    # Suggestion 1: Highest APR
    high_apr = pools_df.sort_values('apr', ascending=False).head(3)
    for _, pool in high_apr.iterrows():
        suggestions.append({
            'id': pool['id'],
            'name': pool['name'],
            'reason': f"High APR ({format_percentage(pool['apr'])})"
        })
    
    # Suggestion 2: High volume with decent APR
    high_volume = pools_df[pools_df['apr'] > 5].sort_values('volume_24h', ascending=False).head(2)
    for _, pool in high_volume.iterrows():
        suggestions.append({
            'id': pool['id'],
            'name': pool['name'],
            'reason': f"High volume ({format_currency(pool['volume_24h'])}) with {format_percentage(pool['apr'])} APR"
        })
    
    # Suggestion 3: Stable pairs
    stable_pairs = pools_df[pools_df['category'] == 'Stablecoin'].sort_values('apr', ascending=False).head(2)
    for _, pool in stable_pairs.iterrows():
        suggestions.append({
            'id': pool['id'],
            'name': pool['name'],
            'reason': "Stable pair with good returns"
        })
    
    # Return top suggestions
    return suggestions[:max_count]

def migrate_legacy_watchlist():
    """
    Import legacy watchlist data if it exists.
    This function checks for old-format watchlist data and imports it 
    into the new database structure.
    """
    legacy_file = "tokenmapping.json"
    
    if not os.path.exists(legacy_file):
        return False
    
    try:
        with open(legacy_file, 'r') as f:
            legacy_data = json.load(f)
        
        # Check if the file has the expected format
        if not isinstance(legacy_data, dict) or "watched_pools" not in legacy_data:
            return False
        
        # Get the pools
        pool_ids = legacy_data.get("watched_pools", [])
        if not pool_ids:
            return False
        
        # Create a new watchlist for the legacy data
        watchlist = db_handler.create_watchlist(
            name="Imported Legacy Watchlist",
            description="Automatically imported from tokenmapping.json"
        )
        
        if not watchlist:
            return False
        
        # Add the pools
        added = add_batch_pools(watchlist["id"], pool_ids)
        
        st.success(f"Successfully imported {added} pools from legacy watchlist file")
        return True
        
    except Exception as e:
        st.error(f"Error importing legacy watchlist: {e}")
        return False

def main():
    """Main function for the Watchlist Manager page"""
    st.title("📋 Batch Watchlist Manager")
    
    # Introduction
    st.markdown("""
    Manage your watchlists of Solana liquidity pools, including batch operations for tracking 
    multiple pools. Import/export watchlists and organize pools for easier monitoring.
    """)
    
    # Check for legacy data and migrate if needed
    if "legacy_check_done" not in st.session_state:
        migrate_legacy_watchlist()
        st.session_state["legacy_check_done"] = True
    
    # Get all pools
    all_pools = db_handler.get_pools(limit=None)
    pools_df = pd.DataFrame(all_pools)
    
    # Get all watchlists
    watchlists = db_handler.get_watchlists()
    
    # Page layout with tabs
    tab1, tab2, tab3 = st.tabs(["Manage Watchlists", "Batch Operations", "Import/Export"])
    
    # Tab 1: Manage Watchlists
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        # Left column: Create new watchlist & list existing
        with col1:
            st.subheader("Create New Watchlist")
            with st.form("create_watchlist_form"):
                name = st.text_input("Watchlist Name")
                description = st.text_area("Description (optional)")
                submit = st.form_submit_button("Create Watchlist")
                
                if submit and name:
                    # Create the watchlist and immediately get its ID
                    watchlist = db_handler.create_watchlist(name, description)
                    if watchlist:
                        # Get the ID from the watchlist dictionary
                        watchlist_id = watchlist["id"]
                        # Then immediately store the ID in session state
                        st.success(f"Created watchlist: {name}")
                        st.session_state["selected_watchlist"] = watchlist_id
                        # Force refresh
                        st.rerun()
                    else:
                        st.error("Failed to create watchlist")
            
            st.subheader("Your Watchlists")
            if not watchlists:
                st.info("No watchlists found. Create your first watchlist above.")
            else:
                for watchlist in watchlists:
                    # Get count of pools in this watchlist
                    pool_ids = db_handler.get_pools_in_watchlist(watchlist["id"])
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="watchlist-item">
                            <div class="watchlist-header">{watchlist["name"]}</div>
                            <div>{len(pool_ids)} pools</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    cols = st.columns([3, 2, 2])
                    with cols[0]:
                        if st.button(f"View", key=f"view_{watchlist['id']}"):
                            st.session_state["selected_watchlist"] = watchlist["id"]
                            st.session_state["watchlist_view"] = "details"
                            st.rerun()
                    with cols[1]:
                        if st.button(f"Edit", key=f"edit_{watchlist['id']}"):
                            st.session_state["selected_watchlist"] = watchlist["id"]
                            st.session_state["watchlist_view"] = "edit"
                            st.rerun()
                    with cols[2]:
                        if st.button(f"Delete", key=f"delete_{watchlist['id']}"):
                            if db_handler.delete_watchlist(watchlist["id"]):
                                st.success(f"Deleted watchlist: {watchlist['name']}")
                                # Force refresh
                                st.rerun()
                            else:
                                st.error(f"Failed to delete watchlist: {watchlist['name']}")
        
        # Right column: Watchlist details or edit
        with col2:
            # Check if a watchlist is selected
            selected_id = st.session_state.get("selected_watchlist")
            view_mode = st.session_state.get("watchlist_view", "details")
            
            if selected_id:
                # Get watchlist details
                details = db_handler.get_watchlist_details(selected_id)
                
                if not details:
                    st.error("Watchlist not found")
                else:
                    watchlist_info = details["watchlist"]
                    pools = details["pools"]
                    
                    # Show watchlist name and stats
                    st.subheader(watchlist_info["name"])
                    st.markdown(f"*{watchlist_info['description']}*")
                    
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Pools", len(pools))
                    
                    if pools:
                        total_liquidity = sum(pool.get("liquidity", 0) for pool in pools)
                        avg_apr = sum(pool.get("apr", 0) for pool in pools) / len(pools)
                        total_volume = sum(pool.get("volume_24h", 0) for pool in pools)
                        
                        with stats_cols[1]:
                            st.metric("Total Liquidity", format_currency(total_liquidity))
                        with stats_cols[2]:
                            st.metric("Avg APR", format_percentage(avg_apr))
                        with stats_cols[3]:
                            st.metric("24h Volume", format_currency(total_volume))
                    
                    if view_mode == "details":
                        # Show pools in watchlist
                        st.subheader("Pools in this Watchlist")
                        if not pools:
                            st.info("No pools in this watchlist yet. Add pools from the Batch Operations tab.")
                        else:
                            for pool in pools:
                                display_pool_card(pool)
                                remove_col1, remove_col2 = st.columns([4, 1])
                                with remove_col2:
                                    if st.button("Remove", key=f"remove_{pool['id']}"):
                                        if db_handler.remove_pool_from_watchlist(selected_id, pool['id']):
                                            st.success(f"Removed pool from watchlist")
                                            # Force refresh
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.error("Failed to remove pool")
                    
                    elif view_mode == "edit":
                        # Edit watchlist
                        with st.form("edit_watchlist_form"):
                            new_name = st.text_input("Watchlist Name", value=watchlist_info["name"])
                            new_description = st.text_area("Description", value=watchlist_info["description"])
                            save_button = st.form_submit_button("Save Changes")
                            
                            if save_button:
                                # Update watchlist in database
                                session = db_handler.Session()
                                try:
                                    watchlist = session.query(db_handler.Watchlist).filter_by(id=selected_id).first()
                                    if watchlist:
                                        watchlist.name = new_name
                                        watchlist.description = new_description
                                        session.commit()
                                        st.success("Watchlist updated successfully")
                                        # Switch back to details view
                                        st.session_state["watchlist_view"] = "details"
                                        st.rerun()
                                    else:
                                        st.error("Watchlist not found")
                                except Exception as e:
                                    st.error(f"Error updating watchlist: {e}")
                                    session.rollback()
                                finally:
                                    session.close()
            else:
                st.info("Select a watchlist from the list or create a new one")
    
    # Tab 2: Batch Operations
    with tab2:
        st.subheader("Add Multiple Pools to Watchlist")
        
        # Select target watchlist
        watchlist_options = [{"label": w["name"], "value": w["id"]} for w in watchlists]
        target_watchlist = None
        
        if not watchlist_options:
            st.warning("Please create a watchlist first in the 'Manage Watchlists' tab")
        else:
            target_watchlist = st.selectbox(
                "Select Target Watchlist",
                options=[w["id"] for w in watchlists],
                format_func=lambda x: next((w["name"] for w in watchlists if w["id"] == x), ""),
            )
        
        # Only proceed if a watchlist is selected
        if target_watchlist:
            # Show different methods for adding pools
            method_tab1, method_tab2, method_tab3, method_tab4 = st.tabs([
                "Add by ID", "Search & Filter", "Smart Suggestions", "Paste Multiple IDs"
            ])
            
            # Method 1: Add by ID
            with method_tab1:
                st.markdown("Enter a pool ID to add to the watchlist")
                
                pool_id = st.text_input("Pool ID", key="single_pool_id")
                add_button = st.button("Add to Watchlist", key="add_single")
                
                if add_button and pool_id:
                    # Try to find the pool in our DataFrame
                    pool_exists = False
                    pool_info = None
                    if not pools_df.empty:
                        pool_exists = pool_id in pools_df["id"].values
                        if pool_exists:
                            pool_info = pools_df[pools_df["id"] == pool_id].iloc[0].to_dict()
                    
                    # Add to watchlist (our enhanced db_handler will create a placeholder if needed)
                    if db_handler.add_pool_to_watchlist(target_watchlist, pool_id):
                        st.success(f"Added pool {pool_id} to watchlist")
                        
                        # Show the pool info if we have it
                        if pool_info:
                            display_pool_card(pool_info)
                        else:
                            st.info(f"Pool {pool_id} was added as a placeholder. It will be updated with full details when data is refreshed.")
                    else:
                        st.error(f"Failed to add pool {pool_id} to watchlist")
            
            # Method 2: Search & Filter
            with method_tab2:
                st.markdown("Search and filter pools to add to your watchlist")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'dex' in pools_df.columns:
                        dex_options = ["All"] + sorted(pools_df["dex"].unique().tolist())
                        filter_dex = st.selectbox("Filter by DEX", dex_options)
                    else:
                        filter_dex = "All"
                        
                with col2:
                    if 'category' in pools_df.columns:
                        category_options = ["All"] + sorted(pools_df["category"].unique().tolist())
                        filter_category = st.selectbox("Filter by Category", category_options)
                    else:
                        filter_category = "All"
                        
                with col3:
                    sort_options = ["APR", "Liquidity", "Volume 24h", "Prediction Score"]
                    sort_by = st.selectbox("Sort by", sort_options)
                
                # Search
                search_term = st.text_input("Search by name or token", key="batch_search")
                
                # Apply filters to dataframe
                filtered_df = pools_df.copy()
                
                # Apply filters
                if filter_dex != "All" and 'dex' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["dex"] == filter_dex]
                    
                if filter_category != "All" and 'category' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["category"] == filter_category]
                    
                if search_term:
                    search_mask = False
                    if 'name' in filtered_df.columns:
                        search_mask = search_mask | filtered_df["name"].str.contains(search_term, case=False, na=False)
                    if 'token1_symbol' in filtered_df.columns:
                        search_mask = search_mask | filtered_df["token1_symbol"].str.contains(search_term, case=False, na=False)
                    if 'token2_symbol' in filtered_df.columns:
                        search_mask = search_mask | filtered_df["token2_symbol"].str.contains(search_term, case=False, na=False)
                    
                    filtered_df = filtered_df[search_mask]
                
                # Apply sorting
                sort_column_map = {
                    "APR": "apr",
                    "Liquidity": "liquidity",
                    "Volume 24h": "volume_24h",
                    "Prediction Score": "prediction_score"
                }
                
                sort_column = sort_column_map.get(sort_by, "apr")
                if sort_column in filtered_df.columns:
                    filtered_df = filtered_df.sort_values(sort_column, ascending=False)
                
                # Show results
                st.subheader(f"Results ({len(filtered_df)} pools)")
                
                # Allow selecting multiple pools
                selected_pools = []
                
                # Use a form for batch submission
                with st.form("batch_add_form"):
                    # Show top results with checkboxes
                    max_display = min(20, len(filtered_df))
                    for i in range(max_display):
                        pool = filtered_df.iloc[i].to_dict()
                        pool_selected = st.checkbox(
                            f"{pool['name']} - {get_dex_badge(pool['dex'])} - APR: {format_percentage(pool['apr'])}",
                            key=f"pool_check_{pool['id']}",
                            value=False
                        )
                        if pool_selected:
                            selected_pools.append(pool['id'])
                    
                    submit_batch = st.form_submit_button("Add Selected Pools to Watchlist")
                    
                    if submit_batch:
                        if not selected_pools:
                            st.warning("No pools selected")
                        else:
                            added = add_batch_pools(target_watchlist, selected_pools)
                            st.success(f"Added {added} pools to watchlist")
            
            # Method 3: Smart Suggestions
            with method_tab3:
                st.markdown("Get smart suggestions based on performance metrics")
                
                # Generate suggestions
                suggestions = get_pool_suggestions(pools_df)
                
                if not suggestions:
                    st.warning("Not enough pool data for suggestions")
                else:
                    # Show suggestions with reasons
                    selected_suggestions = []
                    
                    with st.form("suggestions_form"):
                        for suggestion in suggestions:
                            pool_id = suggestion['id']
                            pool_info = pools_df[pools_df["id"] == pool_id].iloc[0].to_dict() if pool_id in pools_df["id"].values else None
                            
                            if pool_info:
                                st.markdown(f"### {suggestion['name']}")
                                st.markdown(f"**Recommendation reason:** {suggestion['reason']}")
                                display_pool_card(pool_info)
                                
                                add_this = st.checkbox(f"Add to watchlist", key=f"sugg_{pool_id}")
                                if add_this:
                                    selected_suggestions.append(pool_id)
                                    
                                st.markdown("---")
                        
                        add_suggested = st.form_submit_button("Add Selected Suggestions")
                        
                        if add_suggested:
                            if not selected_suggestions:
                                st.warning("No suggestions selected")
                            else:
                                added = add_batch_pools(target_watchlist, selected_suggestions)
                                st.success(f"Added {added} suggested pools to watchlist")
            
            # Method 4: Paste Multiple IDs
            with method_tab4:
                st.markdown("Paste multiple pool IDs (one per line)")
                
                # Multi-line text area for pool IDs
                pool_ids_text = st.text_area(
                    "Pool IDs",
                    height=200,
                    help="Enter one pool ID per line"
                )
                
                # Process the input
                add_multi_button = st.button("Add All Pools", key="add_multi")
                
                if add_multi_button and pool_ids_text:
                    # Split text into lines and clean up
                    pool_id_lines = [line.strip() for line in pool_ids_text.split('\n') if line.strip()]
                    
                    if not pool_id_lines:
                        st.warning("No valid pool IDs found")
                    else:
                        # Add pools in batch
                        added = add_batch_pools(target_watchlist, pool_id_lines)
                        
                        # Summary of results
                        if added == len(pool_id_lines):
                            st.success(f"Successfully added all {added} pools to watchlist")
                        elif added > 0:
                            st.warning(f"Added {added} out of {len(pool_id_lines)} pools to watchlist")
                            st.info("Some pools may have been skipped because they were already in the watchlist or not found in the database")
                        else:
                            st.error("Failed to add any pools to watchlist")
    
    # Tab 3: Import/Export
    with tab3:
        # Split into import and export columns
        imp_col, exp_col = st.columns(2)
        
        # Import section
        with imp_col:
            st.subheader("Import Watchlist")
            
            import_options = st.radio(
                "Import Method",
                ["Upload JSON File", "Paste JSON Data"]
            )
            
            if import_options == "Upload JSON File":
                uploaded_file = st.file_uploader(
                    "Choose a watchlist JSON file",
                    type=["json"],
                    help="Upload a JSON file containing watchlist data"
                )
                
                if uploaded_file:
                    try:
                        # Read the file contents
                        json_data = uploaded_file.getvalue().decode('utf-8')
                        
                        # Parse as JSON
                        watchlist_data = json.loads(json_data)
                        
                        # Preview the data
                        st.subheader("Preview")
                        st.json(watchlist_data)
                        
                        # Import button
                        if st.button("Import Watchlist from File"):
                            watchlist_id = db_handler.import_watchlist_from_json(watchlist_data)
                            
                            if watchlist_id:
                                st.success(f"Successfully imported watchlist")
                                st.session_state["selected_watchlist"] = watchlist_id
                                st.rerun()
                            else:
                                st.error("Failed to import watchlist")
                        
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            else:
                # Paste JSON Data
                json_text = st.text_area(
                    "Paste JSON Watchlist Data",
                    height=300,
                    help="Paste a valid JSON structure with watchlist data"
                )
                
                if json_text.strip():
                    try:
                        # Parse as JSON
                        watchlist_data = json.loads(json_text)
                        
                        # Preview the data
                        st.subheader("Preview")
                        st.json(watchlist_data)
                        
                        # Import button
                        if st.button("Import Watchlist from Text"):
                            watchlist_id = db_handler.import_watchlist_from_json(watchlist_data)
                            
                            if watchlist_id:
                                st.success(f"Successfully imported watchlist")
                                st.session_state["selected_watchlist"] = watchlist_id
                                st.rerun()
                            else:
                                st.error("Failed to import watchlist")
                        
                    except json.JSONDecodeError:
                        st.error("Invalid JSON data. Please check the format.")
                    except Exception as e:
                        st.error(f"Error processing data: {e}")
            
            # Show sample format
            with st.expander("Sample JSON Format"):
                st.code("""
{
  "name": "High APR Pools",
  "description": "Selection of pools with high APR",
  "pools": [
    "SOLUSDC",
    "MSOLUSDCM",
    "BTCUSDCM"
  ],
  "notes": {
    "SOLUSDC": "Main Solana pool",
    "MSOLUSDCM": "Good volume"
  }
}
                """)
        
        # Export section
        with exp_col:
            st.subheader("Export Watchlist")
            
            if not watchlists:
                st.info("No watchlists available to export")
            else:
                # Select watchlist to export
                export_watchlist = st.selectbox(
                    "Select Watchlist to Export",
                    options=[w["id"] for w in watchlists],
                    format_func=lambda x: next((w["name"] for w in watchlists if w["id"] == x), ""),
                    key="export_select"
                )
                
                if export_watchlist:
                    # Export as JSON
                    json_data = db_handler.export_watchlist_to_json(export_watchlist)
                    
                    if json_data:
                        # Preview the data
                        st.subheader("Preview")
                        st.json(json.loads(json_data))
                        
                        # Export buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"watchlist_{export_watchlist}.json",
                                mime="application/json",
                            )
                        
                        with col2:
                            if st.button("Copy to Clipboard"):
                                st.code(json_data)
                                st.success("JSON data shown above. Copy it to your clipboard.")
                    else:
                        st.error("Failed to export watchlist")
            
            # Backup all watchlists
            st.subheader("Backup All Watchlists")
            
            if st.button("Backup All Watchlists to File"):
                if db_handler.backup_watchlists_to_file():
                    st.success("Successfully backed up all watchlists to watchlists.json")
                    
                    # Offer download of the backup file
                    if os.path.exists("watchlists.json"):
                        with open("watchlists.json", "r") as f:
                            backup_data = f.read()
                        
                        st.download_button(
                            label="Download Backup",
                            data=backup_data,
                            file_name="watchlists_backup.json",
                            mime="application/json",
                        )
                else:
                    st.error("Failed to backup watchlists")

# Run the app
if __name__ == "__main__":
    main()