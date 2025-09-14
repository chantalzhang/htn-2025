#!/usr/bin/env python3
"""
Test script to demonstrate missing data fallback functionality.
"""

import sys
import pandas as pd
from pathlib import Path

# Add scripts directory to path
sys.path.append('scripts')

from process_sport import process_sport_data

def test_missing_data_fallback():
    """Test the missing data fallback logic."""
    
    print("=" * 60)
    print("MISSING DATA FALLBACK TEST")
    print("=" * 60)
    
    # Process the test data with missing values
    result = process_sport_data(
        'basketball_test_missing',
        'data/basketball_test_missing.csv',
        'processed/basketball_test_missing.csv'
    )
    
    print(f"\nOriginal records: {result['original_records']}")
    print(f"Processed records: {result['processed_records']}")
    print(f"Quality score: {result['validation']['quality_score']}%")
    
    # Show what data was filled
    if 'filled_data' in result['validation'] and result['validation']['filled_data']:
        print("\n✅ FILLED MISSING DATA:")
        for col, count in result['validation']['filled_data'].items():
            print(f"   {col}: {count} values filled")
    
    # Show any remaining missing data
    if result['validation']['missing_data']:
        print("\n⚠️  STILL MISSING:")
        for col, count in result['validation']['missing_data'].items():
            print(f"   {col}: {count} values still missing")
    else:
        print("\n🎉 ALL MISSING DATA SUCCESSFULLY FILLED!")
    
    # Show the processed data
    print("\n" + "=" * 60)
    print("PROCESSED DATA SAMPLE:")
    print("=" * 60)
    
    try:
        df = pd.read_csv('processed/basketball_test_missing.csv')
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error reading processed file: {e}")
    
    return result

if __name__ == "__main__":
    test_missing_data_fallback()
