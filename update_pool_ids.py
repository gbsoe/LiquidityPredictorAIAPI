#!/usr/bin/env python3

# Simple script to update pool IDs in the three sections where they're missing

with open('pages/2_Predictions.py', 'r') as file:
    content = file.read()

# Update all three sections with Pool IDs
modified_content = content.replace(
    '                            # Basic info about the pool with Pool ID\n                            pool_info = f"**{pool[\'pool_name\']}**  \\n" \\\n                                      f"APR: {pool[\'predicted_apr\']:.2f}%  \\n" \\\n                                      f"Risk: {pool[\'risk_score\']:.2f}"',
    '                            # Basic info about the pool with Pool ID\n                            pool_info = f"**{pool[\'pool_name\']}**  \\n" \\\n                                      f"Pool ID: {pool[\'pool_id\']}  \\n" \\\n                                      f"APR: {pool[\'predicted_apr\']:.2f}%  \\n" \\\n                                      f"Risk: {pool[\'risk_score\']:.2f}"'
)

# Also handle the remaining section that doesn't have the Pool ID header yet
modified_content = modified_content.replace(
    '                            # Basic info about the pool\n                            pool_info = f"**{pool[\'pool_name\']}**  \\n" \\\n                                      f"APR: {pool[\'predicted_apr\']:.2f}%  \\n" \\\n                                      f"Risk: {pool[\'risk_score\']:.2f}"',
    '                            # Basic info about the pool with Pool ID\n                            pool_info = f"**{pool[\'pool_name\']}**  \\n" \\\n                                      f"Pool ID: {pool[\'pool_id\']}  \\n" \\\n                                      f"APR: {pool[\'predicted_apr\']:.2f}%  \\n" \\\n                                      f"Risk: {pool[\'risk_score\']:.2f}"'
)

with open('pages/2_Predictions.py', 'w') as file:
    file.write(modified_content)

print("Updated all pool info sections to include Pool ID")