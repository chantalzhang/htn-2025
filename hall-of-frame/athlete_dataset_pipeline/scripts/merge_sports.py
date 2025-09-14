"""
Merge all processed sport data into a unified dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime


def merge_all_sports(processed_dir: str, output_path: str, metadata_path: str = None) -> Dict[str, Any]:
    """
    Merge all processed sport CSV files into a unified dataset.
    
    Args:
        processed_dir: Directory containing processed sport CSV files
        output_path: Path to save the merged dataset
        metadata_path: Optional path to save dataset metadata
    
    Returns:
        Dictionary with merge results and statistics
    """
    print("Starting sport data merge process...")
    
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        return {"error": f"Processed directory not found: {processed_dir}"}
    
    # Find all CSV files in processed directory
    csv_files = list(processed_path.glob("*.csv"))
    if not csv_files:
        return {"error": f"No CSV files found in {processed_dir}"}
    
    print(f"Found {len(csv_files)} sport files to merge")
    
    merged_data = []
    sport_stats = {}
    
    # Process each sport file
    for csv_file in csv_files:
        sport_name = csv_file.stem  # filename without extension
        print(f"Processing {sport_name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Ensure sport column exists
            if 'sport' not in df.columns:
                df['sport'] = sport_name
            
            # Add source file info
            df['source_file'] = csv_file.name
            df['processed_date'] = datetime.now().isoformat()
            
            merged_data.append(df)
            
            # Collect statistics
            sport_stats[sport_name] = {
                'records': len(df),
                'columns': list(df.columns),
                'file_path': str(csv_file)
            }
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            sport_stats[sport_name] = {'error': str(e)}
    
    if not merged_data:
        return {"error": "No data could be loaded from CSV files"}
    
    # Merge all dataframes
    print("Merging all sport data...")
    merged_df = pd.concat(merged_data, ignore_index=True, sort=False)
    
    # Standardize merged dataset
    merged_df = standardize_merged_dataset(merged_df)
    
    # Save merged dataset
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"Merged dataset saved to {output_path}")
    except Exception as e:
        return {"error": f"Failed to save merged dataset: {str(e)}"}
    
    # Generate and save metadata
    metadata = generate_dataset_metadata(merged_df, sport_stats)
    
    if metadata_path:
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Metadata saved to {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save metadata: {str(e)}")
    
    return {
        "success": True,
        "total_records": len(merged_df),
        "sports_included": list(sport_stats.keys()),
        "output_path": output_path,
        "metadata": metadata
    }


def standardize_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the merged dataset for consistency.
    """
    # Ensure consistent column order
    priority_columns = [
        'sport', 'name', 'first_name', 'last_name', 'age', 
        'height_cm', 'weight_kg', 'bmi', 'position', 'team', 'country'
    ]
    
    # Get existing priority columns and other columns
    existing_priority = [col for col in priority_columns if col in df.columns]
    other_columns = [col for col in df.columns if col not in priority_columns]
    
    # Reorder columns
    df = df[existing_priority + other_columns]
    
    # Fill missing values in key columns
    if 'sport' in df.columns:
        df['sport'] = df['sport'].fillna('Unknown')
    
    if 'position' in df.columns:
        df['position'] = df['position'].fillna('Unknown')
    
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
    
    # Sort by sport and name for better organization
    sort_columns = []
    if 'sport' in df.columns:
        sort_columns.append('sport')
    if 'name' in df.columns:
        sort_columns.append('name')
    elif 'last_name' in df.columns:
        sort_columns.append('last_name')
    
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    
    return df


def generate_dataset_metadata(df: pd.DataFrame, sport_stats: Dict) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for the merged dataset.
    """
    metadata = {
        "dataset_info": {
            "name": "Multi-Sport Athlete Dataset",
            "description": "Merged dataset containing athlete statistics from multiple sports",
            "created_date": datetime.now().isoformat(),
            "total_records": len(df),
            "total_columns": len(df.columns)
        },
        "sports": {},
        "columns": {},
        "data_quality": {},
        "statistics": {}
    }
    
    # Sport-level statistics
    for sport in df['sport'].unique():
        sport_df = df[df['sport'] == sport]
        metadata["sports"][sport] = {
            "records": len(sport_df),
            "percentage": round((len(sport_df) / len(df)) * 100, 2)
        }
    
    # Column information
    for col in df.columns:
        col_info = {
            "type": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isna().sum()),
            "null_percentage": round((df[col].isna().sum() / len(df)) * 100, 2)
        }
        
        # Add statistics for numeric columns
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "median": float(df[col].median()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None
            })
        
        # Add unique values for categorical columns
        elif df[col].dtype == 'object':
            unique_values = df[col].dropna().unique()
            col_info.update({
                "unique_count": len(unique_values),
                "unique_values": list(unique_values)[:20]  # Limit to first 20
            })
        
        metadata["columns"][col] = col_info
    
    # Overall data quality metrics
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    
    metadata["data_quality"] = {
        "completeness_percentage": round(((total_cells - missing_cells) / total_cells) * 100, 2),
        "total_missing_values": int(missing_cells),
        "records_with_missing_data": int((df.isna().any(axis=1)).sum()),
        "complete_records": int((~df.isna().any(axis=1)).sum())
    }
    
    # Key statistics
    if 'height_cm' in df.columns:
        metadata["statistics"]["height_cm"] = {
            "mean": float(df['height_cm'].mean()) if not df['height_cm'].isna().all() else None,
            "range": [float(df['height_cm'].min()), float(df['height_cm'].max())] if not df['height_cm'].isna().all() else None
        }
    
    if 'weight_kg' in df.columns:
        metadata["statistics"]["weight_kg"] = {
            "mean": float(df['weight_kg'].mean()) if not df['weight_kg'].isna().all() else None,
            "range": [float(df['weight_kg'].min()), float(df['weight_kg'].max())] if not df['weight_kg'].isna().all() else None
        }
    
    if 'age' in df.columns:
        metadata["statistics"]["age"] = {
            "mean": float(df['age'].mean()) if not df['age'].isna().all() else None,
            "range": [float(df['age'].min()), float(df['age'].max())] if not df['age'].isna().all() else None
        }
    
    # Sport file information
    metadata["source_files"] = sport_stats
    
    return metadata


def create_sport_comparison_report(merged_df: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
    """
    Create a comparison report across sports.
    """
    if 'sport' not in merged_df.columns:
        return {"error": "No sport column found in dataset"}
    
    sports = merged_df['sport'].unique()
    comparison = {}
    
    for sport in sports:
        sport_df = merged_df[merged_df['sport'] == sport]
        
        sport_analysis = {
            "total_athletes": len(sport_df),
            "avg_height_cm": float(sport_df['height_cm'].mean()) if 'height_cm' in sport_df.columns and not sport_df['height_cm'].isna().all() else None,
            "avg_weight_kg": float(sport_df['weight_kg'].mean()) if 'weight_kg' in sport_df.columns and not sport_df['weight_kg'].isna().all() else None,
            "avg_bmi": float(sport_df['bmi'].mean()) if 'bmi' in sport_df.columns and not sport_df['bmi'].isna().all() else None,
            "avg_age": float(sport_df['age'].mean()) if 'age' in sport_df.columns and not sport_df['age'].isna().all() else None,
            "countries": list(sport_df['country'].unique()) if 'country' in sport_df.columns else [],
            "positions": list(sport_df['position'].unique()) if 'position' in sport_df.columns else []
        }
        
        comparison[sport] = sport_analysis
    
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"Sport comparison report saved to {output_path}")
        except Exception as e:
            print(f"Warning: Could not save comparison report: {str(e)}")
    
    return comparison
