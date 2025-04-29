"""
Check the structure of the DeFi API response and analyze the data
"""

import json
import pandas as pd
from collections import Counter, defaultdict

def analyze_api_response(file_path="api_response_sample.json"):
    """
    Analyze the structure and content of an API response sample
    to identify patterns, data types, and distributions
    """
    # Load the response data
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading API response: {str(e)}")
        return
    
    if not data:
        print("No data found in the response")
        return
    
    # Print basic info
    print(f"API Response Analysis - {file_path}")
    print("=" * 60)
    
    if isinstance(data, list):
        print(f"Response is a list with {len(data)} items")
        first_item = data[0] if data else None
    else:
        print("Response is a dictionary object")
        first_item = data
    
    if not first_item:
        print("No items to analyze")
        return
    
    # Analyze structure
    print("\nStructure Analysis:")
    print("-" * 60)
    
    def analyze_object(obj, prefix="", max_depth=3, current_depth=0):
        """Recursively analyze object structure"""
        if current_depth >= max_depth:
            return f"{prefix}... (max depth reached)"
        
        if isinstance(obj, dict):
            result = []
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)) and value:
                    result.append(f"{path} ({type(value).__name__})")
                    result.append(analyze_object(value, path, max_depth, current_depth + 1))
                else:
                    type_name = type(value).__name__
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."
                    result.append(f"{path}: {type_name} = {value_str}")
            return "\n".join(result)
        elif isinstance(obj, list) and obj:
            sample = obj[0]
            if isinstance(sample, (dict, list)):
                return analyze_object(sample, f"{prefix}[0]", max_depth, current_depth + 1)
            else:
                return f"{prefix}[]: {type(sample).__name__} (list of {len(obj)} items)"
        else:
            return f"{prefix}: {type(obj).__name__}"
    
    print(analyze_object(first_item))
    
    # Token analysis - check if tokens are available
    token_info = False
    for item in data:
        tokens = item.get('tokens', [])
        if tokens:
            token_info = True
            break
    
    print(f"\nToken info available in response: {'Yes' if token_info else 'No'}")
    
    # Extract token info from names instead
    all_tokens = []
    pair_counts = Counter()
    
    for item in data:
        name = item.get('name', '')
        if name and '-' in name:
            parts = name.split('-')
            if len(parts) >= 2:
                token1 = parts[0].strip()
                token2_parts = parts[1].split(' ')
                token2 = token2_parts[0].strip()
                all_tokens.extend([token1, token2])
                pair_counts[f"{token1}-{token2}"] += 1
    
    token_counts = Counter(all_tokens)
    
    print("\nExtracted Token Info:")
    print("-" * 60)
    print(f"Unique tokens found: {len(token_counts)}")
    
    print("\nTop Tokens:")
    for token, count in token_counts.most_common(10):
        print(f"  {token}: {count} occurrences")
    
    print("\nTop Token Pairs:")
    for pair, count in pair_counts.most_common(5):
        print(f"  {pair}: {count} occurrences")
    
    # Analysis of metrics
    print("\nMetrics Analysis:")
    print("-" * 60)
    
    metrics = defaultdict(list)
    for item in data:
        item_metrics = item.get('metrics', {})
        for key, value in item_metrics.items():
            if isinstance(value, (int, float)) and key not in ('id', 'poolId'):
                metrics[key].append(value)
    
    for metric, values in metrics.items():
        if values:
            print(f"{metric}:")
            print(f"  Min: {min(values)}")
            print(f"  Max: {max(values)}")
            print(f"  Avg: {sum(values) / len(values)}")
    
    # DEX distribution
    dex_counts = Counter()
    for item in data:
        dex = item.get('source', 'Unknown')
        if dex != 'Unknown':
            dex_counts[dex] += 1
    
    print("\nDEX Distribution:")
    print("-" * 60)
    for dex, count in dex_counts.most_common():
        print(f"  {dex}: {count} pools")
    
    # Save as a structured report
    try:
        report = {
            "response_type": "list" if isinstance(data, list) else "object",
            "item_count": len(data) if isinstance(data, list) else 1,
            "token_info_available": token_info,
            "extracted_tokens": dict(token_counts),
            "extracted_pairs": dict(pair_counts),
            "dex_distribution": dict(dex_counts),
            "metrics_summary": {
                metric: {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                } for metric, values in metrics.items() if values
            }
        }
        
        with open("api_structure_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\nDetailed report saved to api_structure_report.json")
    except Exception as e:
        print(f"Error saving report: {str(e)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_api_response()