"""
Data loading and preprocessing for athlete clustering analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class AthleteDataLoader:
    """Handles loading and initial preprocessing of athlete data."""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the merged athlete dataset CSV
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load athlete dataset from CSV.
        
        Returns:
            DataFrame with raw athlete data
        """
        print("📊 Loading athlete dataset...")
        
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} athletes across {self.df['sport'].nunique()} sports")
        
        # Display basic info
        print(f"📋 Columns: {list(self.df.columns)}")
        print(f"\n🔍 Dataset Overview:")
        print(f"   Sports: {', '.join(self.df['sport'].unique())}")
        print(f"   Gender distribution: {dict(self.df['Sex'].value_counts())}")
        print(f"   Missing data summary:")
        
        missing_summary = self.df.isnull().sum()
        for col, missing_count in missing_summary[missing_summary > 0].items():
            print(f"     {col}: {missing_count} missing ({missing_count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values with intelligent imputation.
        
        Returns:
            DataFrame with missing values handled
        """
        print("\n🔧 Handling missing values...")
        self.df_processed = self.df.copy()
        
        # Clean Sex column - handle missing values
        if self.df_processed['Sex'].isnull().any():
            print("   Warning: Some athletes missing gender info, filling with 'M' as default")
            self.df_processed['Sex'].fillna('M', inplace=True)
        
        # Define numeric columns for imputation
        numeric_cols = ['height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 'Torso Length']
        
        # Filter to existing columns and check for numeric data
        numeric_cols = [col for col in numeric_cols if col in self.df_processed.columns]
        
        for col in numeric_cols:
            # Convert to numeric first, handling any string values
            self.df_processed[col] = pd.to_numeric(self.df_processed[col], errors='coerce')
            
            if self.df_processed[col].isnull().any():
                print(f"   Imputing {col}...")
                
                # Try imputation by sex and sport first
                for sex in ['M', 'F']:
                    for sport in self.df_processed['sport'].unique():
                        mask = (self.df_processed['Sex'] == sex) & (self.df_processed['sport'] == sport)
                        subset = self.df_processed.loc[mask, col]
                        
                        if subset.isnull().any() and subset.notna().any():
                            mean_val = subset.mean()
                            self.df_processed.loc[mask & self.df_processed[col].isnull(), col] = mean_val
                
                # Fallback: impute by sex only
                for sex in ['M', 'F']:
                    mask = self.df_processed['Sex'] == sex
                    subset = self.df_processed.loc[mask, col]
                    
                    if subset.isnull().any() and subset.notna().any():
                        mean_val = subset.mean()
                        self.df_processed.loc[mask & self.df_processed[col].isnull(), col] = mean_val
                
                # Final fallback: overall mean
                if self.df_processed[col].isnull().any():
                    overall_mean = self.df_processed[col].mean()
                    if pd.notna(overall_mean):
                        self.df_processed[col].fillna(overall_mean, inplace=True)
                    else:
                        # If still no valid data, use reasonable defaults
                        defaults = {'height_cm': 180, 'weight_kg': 75, 'Arm Span': 180, 'Leg Length': 90, 'Torso Length': 90}
                        self.df_processed[col].fillna(defaults.get(col, 0), inplace=True)
        
        print("✅ Missing value imputation complete")
        return self.df_processed
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dictionary with data summary
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'total_athletes': len(self.df),
            'sports': list(self.df['sport'].unique()),
            'sport_counts': dict(self.df['sport'].value_counts()),
            'gender_distribution': dict(self.df['Sex'].value_counts()) if 'Sex' in self.df.columns else {},
            'missing_data': dict(self.df.isnull().sum()[self.df.isnull().sum() > 0])
        }
        
        return summary
