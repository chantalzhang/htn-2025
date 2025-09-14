"""
Process individual sport data files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from utils import (
    convert_height_to_cm, 
    convert_weight_to_kg, 
    standardize_position,
    calculate_bmi,
    clean_numeric_column,
    validate_athlete_data
)


def process_sport_data(sport_name: str, raw_data_path: str, output_path: str) -> Dict:
    """
    Process raw sport data into standardized format.
    
    Args:
        sport_name: Name of the sport (e.g., 'basketball', 'swimming')
        raw_data_path: Path to raw CSV file
        output_path: Path to save processed CSV
    
    Returns:
        Dictionary with processing results and validation metrics
    """
    print(f"Processing {sport_name} data...")
    
    # Load raw data
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(df)} records from {raw_data_path}")
    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}
    
    # Create a copy for processing
    processed_df = df.copy()
    
    # Add sport column
    processed_df['sport'] = sport_name
    
    # Standardize common columns
    processed_df = standardize_common_columns(processed_df, sport_name)
    
    # Apply sport-specific processing
    processed_df = apply_sport_specific_processing(processed_df, sport_name)
    
    # Calculate derived metrics
    processed_df = calculate_derived_metrics(processed_df)
    
    # Clean and validate data
    processed_df = clean_final_data(processed_df)
    
    # Fill missing measurements using random sampling from distributions
    from utils import fill_missing_measurements
    processed_df = fill_missing_measurements(processed_df, sport_name)
    
    # Save processed data
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    except Exception as e:
        return {"error": f"Failed to save data: {str(e)}"}
    
    # Validate data and get quality metrics
    validation_results = validate_athlete_data(processed_df, sport_name)
    
    return {
        "sport": sport_name,
        "original_records": len(df),
        "processed_records": len(processed_df),
        "output_path": output_path,
        "validation": validation_results
    }


def standardize_common_columns(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Standardize common columns across all sports.
    """
    # Common column mappings (flexible to handle different naming conventions)
    column_mappings = {
        # Name variations
        'name': ['name', 'player_name', 'athlete_name', 'full_name'],
        'first_name': ['first_name', 'fname', 'first'],
        'last_name': ['last_name', 'lname', 'last', 'surname'],
        
        # Physical attributes
        'height': ['height', 'ht', 'height_in', 'height_cm'],
        'weight': ['weight', 'wt', 'weight_lbs', 'weight_kg'],
        'age': ['age', 'years_old'],
        
        # Position/Role
        'position': ['position', 'pos', 'role', 'event'],
        
        # Team/Country
        'team': ['team', 'club', 'organization'],
        'country': ['country', 'nation', 'nationality'],
        
        # Performance metrics (sport-specific)
        'points': ['points', 'pts', 'score'],
        'games': ['games', 'matches', 'competitions']
    }
    
    # Find and rename columns
    for standard_name, variations in column_mappings.items():
        for col in df.columns:
            if col.lower().strip() in [v.lower() for v in variations]:
                if standard_name not in df.columns:  # Only rename if standard name doesn't exist
                    df = df.rename(columns={col: standard_name})
                break
    
    # Convert height and weight to standard units
    if 'height' in df.columns:
        df['height_cm'] = df['height'].apply(convert_height_to_cm)
    
    if 'weight' in df.columns:
        df['weight_kg'] = df['weight'].apply(convert_weight_to_kg)
    
    # Standardize position names
    if 'position' in df.columns:
        df['position'] = df['position'].apply(lambda x: standardize_position(x, sport))
    
    # Clean age column
    if 'age' in df.columns:
        df['age'] = clean_numeric_column(df['age'])
    
    return df


def apply_sport_specific_processing(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Apply sport-specific data processing.
    """
    sport = sport.lower()
    
    if sport == 'basketball':
        return process_basketball_data(df)
    elif sport == 'swimming':
        return process_swimming_data(df)
    elif sport == 'soccer':
        return process_soccer_data(df)
    elif sport == 'tennis':
        return process_tennis_data(df)
    elif sport == 'track_field':
        return process_track_field_data(df)
    elif sport == 'gymnastics':
        return process_gymnastics_data(df)
    elif sport == 'volleyball':
        return process_volleyball_data(df)
    elif sport == 'baseball':
        return process_baseball_data(df)
    elif sport == 'hockey':
        return process_hockey_data(df)
    elif sport == 'cycling':
        return process_cycling_data(df)
    
    return df


def process_basketball_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process basketball-specific data."""
    # Clean common basketball stats
    stat_columns = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']
    for col in stat_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_swimming_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process swimming-specific data."""
    # Convert times to seconds if needed
    time_columns = ['time', 'best_time', 'personal_best']
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_time_to_seconds)
    
    return df


def process_soccer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process soccer-specific data."""
    stat_columns = ['goals', 'assists', 'yellow_cards', 'red_cards', 'minutes_played']
    for col in stat_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_tennis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process tennis-specific data."""
    stat_columns = ['ranking', 'wins', 'losses', 'prize_money']
    for col in stat_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_track_field_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process track and field-specific data."""
    # Convert times and distances
    time_columns = ['time', 'personal_best', 'season_best']
    distance_columns = ['distance', 'throw_distance', 'jump_distance']
    
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_time_to_seconds)
    
    for col in distance_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_gymnastics_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process gymnastics-specific data."""
    score_columns = ['total_score', 'vault', 'uneven_bars', 'balance_beam', 'floor_exercise']
    for col in score_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_volleyball_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process volleyball-specific data."""
    stat_columns = ['kills', 'blocks', 'digs', 'aces', 'errors']
    for col in stat_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_baseball_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process baseball-specific data."""
    stat_columns = ['batting_average', 'home_runs', 'rbi', 'era', 'strikeouts']
    for col in stat_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_hockey_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process hockey-specific data."""
    stat_columns = ['goals', 'assists', 'penalty_minutes', 'plus_minus', 'shots']
    for col in stat_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    return df


def process_cycling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process cycling-specific data."""
    stat_columns = ['time', 'speed', 'distance', 'watts', 'heart_rate']
    for col in stat_columns:
        if col in df.columns:
            if col == 'time':
                df[col] = df[col].apply(convert_time_to_seconds)
            else:
                df[col] = clean_numeric_column(df[col])
    
    return df


def convert_time_to_seconds(time_str) -> float:
    """
    Convert time string to seconds.
    Handles formats like: "1:23.45", "23.45", "1h 23m 45s"
    """
    if pd.isna(time_str):
        return np.nan
    
    time_str = str(time_str).strip()
    
    try:
        # Handle "MM:SS.ms" format
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:  # "HH:MM:SS" format
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        
        # Handle plain seconds
        return float(time_str)
    
    except ValueError:
        return np.nan


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics like BMI, ratios, etc.
    """
    # Calculate BMI
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
        df['bmi'] = df.apply(lambda row: calculate_bmi(row['height_cm'], row['weight_kg']), axis=1)
    
    # Add more derived metrics as needed
    
    return df


def clean_final_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final data cleaning and organization.
    """
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Ensure consistent column order
    priority_columns = ['sport', 'name', 'first_name', 'last_name', 'age', 'height_cm', 'weight_kg', 'bmi', 'position', 'team', 'country']
    
    # Reorder columns with priority columns first
    existing_priority = [col for col in priority_columns if col in df.columns]
    other_columns = [col for col in df.columns if col not in priority_columns]
    df = df[existing_priority + other_columns]
    
    return df
