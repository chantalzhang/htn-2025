"""
Feature engineering for athlete body type clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class FeatureEngineer:
    """Handles feature creation and scaling for clustering analysis."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.features_scaled = None
        self.feature_names = None
    
    def create_body_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create body ratio features to normalize proportions.
        
        Args:
            df: DataFrame with athlete data
            
        Returns:
            DataFrame with added body ratio features
        """
        print("\n📐 Creating body ratio features...")
        
        df_features = df.copy()
        
        # Basic ratios
        if 'height_cm' in df_features.columns and 'weight_kg' in df_features.columns:
            # BMI (if not already calculated)
            if 'BMI' not in df_features.columns and 'bmi' not in df_features.columns:
                df_features['BMI'] = df_features['weight_kg'] / (df_features['height_cm'] / 100) ** 2
            elif 'bmi' in df_features.columns:
                df_features['BMI'] = df_features['bmi']
        
        # Limb proportions (if limb measurements available)
        if 'Arm Span' in df_features.columns and 'height_cm' in df_features.columns:
            df_features['arm_span_height_ratio'] = df_features['Arm Span'] / df_features['height_cm']
        
        if 'Leg Length' in df_features.columns and 'height_cm' in df_features.columns:
            df_features['leg_height_ratio'] = df_features['Leg Length'] / df_features['height_cm']
        
        if 'Torso Length' in df_features.columns and 'height_cm' in df_features.columns:
            df_features['torso_height_ratio'] = df_features['Torso Length'] / df_features['height_cm']
        
        # Weight-height relationships
        if 'weight_kg' in df_features.columns and 'height_cm' in df_features.columns:
            df_features['weight_height_ratio'] = df_features['weight_kg'] / df_features['height_cm']
        
        print("✅ Body ratio features created")
        return df_features
    
    def select_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare features for clustering.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with selected clustering features
        """
        print("\n🎯 Selecting clustering features...")
        
        # Define potential clustering features
        potential_features = [
            'height_cm', 'weight_kg', 'BMI',
            'arm_span_height_ratio', 'leg_height_ratio', 'torso_height_ratio',
            'weight_height_ratio'
        ]
        
        # Select features that exist in the dataset
        available_features = [f for f in potential_features if f in df.columns]
        
        if not available_features:
            raise ValueError("No suitable clustering features found in dataset")
        
        features_df = df[available_features].copy()
        
        # Remove any remaining NaN values
        features_df = features_df.dropna()
        
        print(f"   Selected features: {available_features}")
        print(f"   Feature matrix shape: {features_df.shape}")
        
        self.feature_names = available_features
        return features_df
    
    def scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize features using z-score normalization.
        
        Args:
            features_df: DataFrame with clustering features
            
        Returns:
            DataFrame with scaled features
        """
        print("\n⚖️ Standardizing features...")
        
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features_df),
            columns=features_df.columns,
            index=features_df.index
        )
        
        print("✅ Feature scaling complete")
        print(f"   Scaled feature statistics:")
        print(f"   Mean: {self.features_scaled.mean().round(3).to_dict()}")
        print(f"   Std:  {self.features_scaled.std().round(3).to_dict()}")
        
        return self.features_scaled
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on variance.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.features_scaled is None:
            raise ValueError("Features not scaled yet. Call scale_features() first.")
        
        importance = pd.DataFrame({
            'feature': self.features_scaled.columns,
            'variance': self.features_scaled.var(),
            'std_dev': self.features_scaled.std()
        }).sort_values('variance', ascending=False)
        
        return importance
