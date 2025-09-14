"""
Utility functions for athlete data processing pipeline.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from sport_averages import get_sport_average, get_sport_std, get_measurement_bounds

def convert_height_to_cm(height: str, unit: str = 'auto') -> float:
    """
    Convert height to centimeters from various formats.
    
    Args:
        height: Height string (e.g., "6'2\"", "1.88m", "188cm")
        unit: Unit type ('auto', 'ft', 'm', 'cm')
    
    Returns:
        Height in centimeters
    """
    if pd.isna(height) or height == '':
        return np.nan
    
    height_str = str(height).strip().lower()
    
    # Handle feet and inches format (6'2", 6ft 2in, etc.)
    if "'" in height_str or 'ft' in height_str:
        import re
        match = re.search(r"(\d+)['ft]\s*(\d+)?", height_str)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2)) if match.group(2) else 0
            return (feet * 12 + inches) * 2.54
    
    # Handle meters
    if 'm' in height_str and 'cm' not in height_str:
        return float(height_str.replace('m', '')) * 100
    
    # Handle centimeters
    if 'cm' in height_str:
        return float(height_str.replace('cm', ''))
    
    # Try to parse as number (assume cm if > 10, else assume meters)
    try:
        num = float(height_str)
        return num * 100 if num < 10 else num
    except ValueError:
        return np.nan


def convert_weight_to_kg(weight: str, unit: str = 'auto') -> float:
    """
    Convert weight to kilograms from various formats.
    
    Args:
        weight: Weight string (e.g., "180lbs", "82kg", "180")
        unit: Unit type ('auto', 'lbs', 'kg')
    
    Returns:
        Weight in kilograms
    """
    if pd.isna(weight) or weight == '':
        return np.nan
    
    weight_str = str(weight).strip().lower()
    
    # Handle pounds
    if 'lb' in weight_str or unit == 'lbs':
        return float(weight_str.replace('lbs', '').replace('lb', '')) * 0.453592
    
    # Handle kilograms
    if 'kg' in weight_str or unit == 'kg':
        return float(weight_str.replace('kg', ''))
    
    # Try to parse as number (assume lbs if > 100, else assume kg)
    try:
        num = float(weight_str)
        return num * 0.453592 if num > 100 else num
    except ValueError:
        return np.nan


def standardize_position(position: str, sport: str) -> str:
    """
    Standardize position names across sports.
    
    Args:
        position: Original position name
        sport: Sport type
    
    Returns:
        Standardized position name
    """
    if pd.isna(position):
        return 'Unknown'
    
    position = str(position).strip().upper()
    
    # Basketball positions
    if sport.lower() == 'basketball':
        position_map = {
            'PG': 'Point Guard', 'POINT GUARD': 'Point Guard',
            'SG': 'Shooting Guard', 'SHOOTING GUARD': 'Shooting Guard',
            'SF': 'Small Forward', 'SMALL FORWARD': 'Small Forward',
            'PF': 'Power Forward', 'POWER FORWARD': 'Power Forward',
            'C': 'Center', 'CENTER': 'Center'
        }
        return position_map.get(position, position)
    
    # Add more sport-specific mappings as needed
    return position


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """
    Calculate BMI from height and weight.
    
    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
    
    Returns:
        BMI value
    """
    if pd.isna(height_cm) or pd.isna(weight_kg) or height_cm <= 0 or weight_kg <= 0:
        return np.nan
    
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean and convert a series to numeric, handling various formats.
    
    Args:
        series: Pandas series to clean
    
    Returns:
        Cleaned numeric series
    """
    # Remove common non-numeric characters
    cleaned = series.astype(str).str.replace(r'[,$%]', '', regex=True)
    
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(cleaned, errors='coerce')


def convert_measurements_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert height, weight, and wingspan to numeric.
    
    Args:
        df: DataFrame with athlete data
    
    Returns:
        DataFrame with converted measurements
    """
    df_converted = df.copy()
    
    # Convert height, weight, and wingspan to numeric
    for col in ['height_cm', 'weight_kg', 'wingspan_cm']:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].apply(
                lambda x: convert_height_to_cm(str(x)) if col == 'height_cm' or col == 'wingspan_cm' 
                else convert_weight_to_kg(str(x)) if col == 'weight_kg' 
                else np.nan
            )
    
    return df_converted


def fill_missing_measurements(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Fill missing measurements using gender-aware sport-specific averages and statistical sampling.
    
    Args:
        df: DataFrame with athlete data
        sport: Sport name for getting appropriate averages
    
    Returns:
        DataFrame with missing measurements filled
    """
    df_filled = df.copy()
    
    # Convert height, weight, wingspan to numeric first
    df_filled = convert_measurements_to_numeric(df_filled)
    
    # Measurements that can be filled with fallback logic
    fillable_measurements = [
        'height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 
        'Torso Length', 'Spike Reach', 'Block Reach',
        'arm_span_cm', 'leg_length_cm', 'torso_length_cm', 
        'spike_reach_cm', 'block_reach_cm'
    ]
    
    filled_count = 0
    
    for measurement in fillable_measurements:
        if measurement in df_filled.columns:
            missing_mask = df_filled[measurement].isna()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                # Process by gender for more accurate averages
                for gender in ['M', 'F']:
                    gender_mask = (df_filled['Sex'] == gender) & missing_mask
                    gender_missing_count = gender_mask.sum()
                    
                    if gender_missing_count > 0:
                        # Map column names to standard measurement names for sport averages
                        measurement_mapping = {
                            'Arm Span': 'arm_span_cm',
                            'Leg Length': 'leg_length_cm', 
                            'Torso Length': 'torso_length_cm',
                            'Spike Reach': 'spike_reach_cm',
                            'Block Reach': 'block_reach_cm'
                        }
                        
                        # Get the standard measurement name for sport averages
                        standard_measurement = measurement_mapping.get(measurement, measurement)
                        
                        # Get gender-specific sport average and std
                        avg = get_sport_average(sport, standard_measurement, gender)
                        std = get_sport_std(sport, standard_measurement)
                        
                        # If we have existing data for this gender, blend with it
                        existing_data = df_filled[(df_filled['Sex'] == gender) & df_filled[measurement].notna()][measurement]
                        if len(existing_data) > 0:
                            existing_mean = existing_data.mean()
                            existing_std = existing_data.std() if len(existing_data) > 1 else std
                            
                            # Blend professional averages with existing data (70% professional, 30% existing)
                            blended_mean = 0.7 * avg + 0.3 * existing_mean
                            blended_std = 0.7 * std + 0.3 * existing_std
                        else:
                            blended_mean = avg
                            blended_std = std
                        
                        # Generate realistic values using normal distribution
                        filled_values = np.random.normal(blended_mean, blended_std, gender_missing_count)
                        
                        # Apply realistic bounds to prevent unrealistic values
                        bounds = get_measurement_bounds(standard_measurement)
                        filled_values = np.clip(filled_values, bounds[0], bounds[1])
                        
                        # Fill missing values for this gender
                        df_filled.loc[gender_mask, measurement] = filled_values
                        filled_count += gender_missing_count
                        
                        if gender_missing_count > 0:
                            print(f"  Filled {gender_missing_count} missing {measurement} values for {gender} athletes using sport averages")
    
    print(f"  Total measurements filled: {filled_count}")
    return df_filled


def validate_athlete_data(df: pd.DataFrame, sport: str) -> Dict:
    """
    Validate athlete data and calculate quality metrics.
    Note: This function now assumes missing values have already been filled.
    
    Args:
        df: DataFrame with athlete data (should already have missing values filled)
        sport: Sport name
    
    Returns:
        Dictionary with validation results and quality metrics
    """
    results = {
        'total_records': len(df),
        'valid_records': 0,
        'missing_data': {},
        'outliers': {},
        'quality_score': 0.0,
        'warnings': []
    }
    
    # Check for remaining missing data
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            results['missing_data'][col] = missing_count
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns and df[col].notna().any():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                results['outliers'][col] = len(outliers)
    
    # Calculate quality score
    total_cells = len(df) * len(df.columns)
    missing_cells = sum(results['missing_data'].values())
    quality_score = ((total_cells - missing_cells) / total_cells) * 100
    results['quality_score'] = round(quality_score, 2)
    
    # Count valid records (records with no missing critical data)
    critical_cols = ['height_cm', 'weight_kg'] if 'height_cm' in df.columns else []
    if critical_cols:
        valid_mask = df[critical_cols].notnull().all(axis=1)
        results['valid_records'] = valid_mask.sum()
    else:
        results['valid_records'] = len(df)
    
    return results
