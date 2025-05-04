#!/bin/bash

# Update first pool info in Strategic Investment Opportunities section
sed -i '293,295s/# Basic info about the pool/# Basic info about the pool with Pool ID/' pages/2_Predictions.py
sed -i '294s/f"APR: {pool/f"Pool ID: {pool['\''pool_id'\'']}\\\n" \\\n                                      f"APR: {pool/' pages/2_Predictions.py

# Update second pool info in High Yield Opportunities section
sed -i '340,342s/# Basic info about the pool/# Basic info about the pool with Pool ID/' pages/2_Predictions.py
sed -i '341s/f"APR: {pool/f"Pool ID: {pool['\''pool_id'\'']}\\\n" \\\n                                      f"APR: {pool/' pages/2_Predictions.py

# Update third pool info in Conservative Investment Options section
sed -i '387,389s/# Basic info about the pool/# Basic info about the pool with Pool ID/' pages/2_Predictions.py
sed -i '388s/f"APR: {pool/f"Pool ID: {pool['\''pool_id'\'']}\\\n" \\\n                                      f"APR: {pool/' pages/2_Predictions.py