#!/bin/bash

# Update the Strategic Investment Opportunities section
sed -i '293s/# Basic info about the pool/# Basic info about the pool with Pool ID/' pages/2_Predictions.py
sed -i '293,297s/pool_info = f"\*\*{pool\['\''pool_name'\''\]}\*\*  \\n" \\\n                                      f"Pool ID: {pool\['\''pool_id'\''\]}\\\\n" \\\n                                      f"APR: {pool\['\''predicted_apr'\''\]:.2f}%  \\n" \\\n                                      f"Risk: {pool\['\''risk_score'\''\]:.2f}"/pool_info = f"\*\*{pool\['\''pool_name'\''\]}\*\*  \\n" \\\n                                      f"Pool ID: {pool\['\''pool_id'\''\]}  \\n" \\\n                                      f"APR: {pool\['\''predicted_apr'\''\]:.2f}%  \\n" \\\n                                      f"Risk: {pool\['\''risk_score'\''\]:.2f}"/' pages/2_Predictions.py

# Update the High Yield Opportunities section
sed -i '341s/# Basic info about the pool with Pool ID/# Basic info about the pool with Pool ID/' pages/2_Predictions.py
sed -i '342,344s/pool_info = f"\*\*{pool\['\''pool_name'\''\]}\*\*  \\n" \\\n                                      f"APR: {pool\['\''predicted_apr'\''\]:.2f}%  \\n" \\\n                                      f"Risk: {pool\['\''risk_score'\''\]:.2f}"/pool_info = f"\*\*{pool\['\''pool_name'\''\]}\*\*  \\n" \\\n                                      f"Pool ID: {pool\['\''pool_id'\''\]}  \\n" \\\n                                      f"APR: {pool\['\''predicted_apr'\''\]:.2f}%  \\n" \\\n                                      f"Risk: {pool\['\''risk_score'\''\]:.2f}"/' pages/2_Predictions.py

# Update the Conservative Investment Options section
sed -i '388s/# Basic info about the pool with Pool ID/# Basic info about the pool with Pool ID/' pages/2_Predictions.py
sed -i '389,391s/pool_info = f"\*\*{pool\['\''pool_name'\''\]}\*\*  \\n" \\\n                                      f"APR: {pool\['\''predicted_apr'\''\]:.2f}%  \\n" \\\n                                      f"Risk: {pool\['\''risk_score'\''\]:.2f}"/pool_info = f"\*\*{pool\['\''pool_name'\''\]}\*\*  \\n" \\\n                                      f"Pool ID: {pool\['\''pool_id'\''\]}  \\n" \\\n                                      f"APR: {pool\['\''predicted_apr'\''\]:.2f}%  \\n" \\\n                                      f"Risk: {pool\['\''risk_score'\''\]:.2f}"/' pages/2_Predictions.py