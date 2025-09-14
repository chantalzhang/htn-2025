"""
Professional athlete average measurements by sport.
These are fallback values for missing data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

# Professional sport averages based on research data - gender-specific
SPORT_AVERAGES_MALE = {
    'basketball': {
        'height_cm': 201.5,  # NBA average: 200-203 cm
        'weight_kg': 98.5,   # NBA average: 97-100 kg
        'arm_span_cm': 213.4,  # NBA average wingspan
        'leg_length_cm': 105.8,  # ~0.525 ratio (long legs)
        'torso_length_cm': 95.7,  # ~0.475 ratio
        'age': 26.2,
        'bmi': 24.2
    },
    'swimming': {
        'height_cm': 190.0,  # Male swimmers: 185-195 cm
        'weight_kg': 84.0,   # Male swimmers: 80-88 kg
        'arm_span_cm': 198.8,  # ~1.045 ratio (longer arms for stroke)
        'leg_length_cm': 90.8,  # ~0.478 ratio (shorter legs, longer torso)
        'torso_length_cm': 99.2,  # ~0.522 ratio (longer torso for buoyancy)
        'age': 24.7,
        'bmi': 23.3
    },
    'soccer': {
        'height_cm': 179.5,  # Male soccer: 176-183 cm
        'weight_kg': 75.5,   # Male soccer: 73-78 kg
        'arm_span_cm': 179.5,  # 1.00 ratio of height
        'leg_length_cm': 94.2,  # ~0.525 ratio (longer legs for running/kicking)
        'torso_length_cm': 85.3,  # ~0.475 ratio (athletic build)
        'age': 25.8,
        'bmi': 23.4
    },
    'tennis': {
        'height_cm': 187.5,  # Male tennis: 185-190 cm
        'weight_kg': 82.5,   # Male tennis: 80-85 kg
        'arm_span_cm': 187.5,  # 1.00 ratio of height
        'leg_length_cm': 98.4,  # ~0.525 ratio (longer legs for court coverage)
        'torso_length_cm': 89.1,  # ~0.475 ratio (athletic build)
        'age': 26.1,
        'bmi': 23.5
    },
    'track_field': {
        'height_cm': 181.5,  # Male sprinters: 178-185 cm
        'weight_kg': 78.5,   # Male sprinters: 74-83 kg
        'arm_span_cm': 181.5,  # 1.00 ratio of height
        'leg_length_cm': 95.3,  # ~0.525 ratio (longer legs for sprinters)
        'torso_length_cm': 86.2,  # ~0.475 ratio
        'age': 25.4,
        'bmi': 23.8
    },
    'gymnastics': {
        'height_cm': 168.5,  # Male gymnasts: 165-172 cm
        'weight_kg': 66.0,   # Male gymnasts: 62-70 kg
        'arm_span_cm': 168.5,  # 1.00 ratio of height
        'leg_length_cm': 84.3,  # ~0.50 ratio (balanced proportions)
        'torso_length_cm': 84.3,  # ~0.50 ratio (strong core)
        'age': 23.8,
        'bmi': 23.3
    },
    'volleyball': {
        'height_cm': 197.5,  # Male volleyball: 195-200 cm
        'weight_kg': 91.5,   # Male volleyball: 88-95 kg
        'arm_span_cm': 207.4,  # ~1.05 ratio (longer reach for spiking/blocking)
        'leg_length_cm': 103.7,  # ~0.525 ratio (longer legs for jumping)
        'torso_length_cm': 93.8,  # ~0.475 ratio (athletic build)
        'spike_reach_cm': 355.0,  # Average spike reach for elite males
        'block_reach_cm': 340.0,  # Average block reach for elite males
        'age': 26.8,
        'bmi': 23.5
    },
    'baseball': {
        'height_cm': 185.4,  # ~6'1"
        'weight_kg': 90.7,   # ~200 lbs
        'age': 27.9,
        'bmi': 26.4
    },
    'hockey': {
        'height_cm': 185.4,  # ~6'1"
        'weight_kg': 90.7,   # ~200 lbs
        'age': 27.1,
        'bmi': 26.4
    },
    'cycling': {
        'height_cm': 180.3,  # ~5'11"
        'weight_kg': 68.0,   # ~150 lbs
        'age': 28.3,
        'bmi': 20.9
    },
    'distance_running': {
        'height_cm': 171.5,  # Male distance runners: 168-175 cm
        'weight_kg': 57.0,   # Male distance runners: 54-60 kg
        'arm_span_cm': 171.5,  # 1.00 ratio of height
        'leg_length_cm': 90.0,  # ~0.525 ratio (longer legs for distance)
        'torso_length_cm': 81.5,  # ~0.475 ratio (ectomorphic build)
        'age': 26.5,
        'bmi': 19.4  # Lower BMI for ectomorphic distance runners
    },
    'weightlifting': {
        'height_cm': 173.0,  # Male weightlifting: 168-178 cm
        'weight_kg': 89.5,   # Male weightlifting: 85-94 kg
        'arm_span_cm': 173.0,  # 1.00 ratio of height
        'leg_length_cm': 86.5,  # ~0.50 ratio (balanced for power generation)
        'torso_length_cm': 86.5,  # ~0.50 ratio (strong core for lifting)
        'age': 27.5,
        'bmi': 29.9  # Higher BMI due to muscle mass
    },
    'rowing': {
        'height_cm': 192.0,  # Male rowing: 189-195 cm
        'weight_kg': 95.0,   # Male rowing: 90-100 kg
        'arm_span_cm': 201.6,  # ~1.05 ratio (longer reach for rowing stroke)
        'leg_length_cm': 99.8,  # ~0.52 ratio (longer legs for drive phase)
        'torso_length_cm': 92.2,  # ~0.48 ratio (strong core for power transfer)
        'age': 28.0,
        'bmi': 25.7
    }
}

# Female sport averages
SPORT_AVERAGES_FEMALE = {
    'basketball': {
        'height_cm': 183.5,  # WNBA average: 182-185 cm
        'weight_kg': 78.5,   # WNBA average: 77-80 kg
        'arm_span_cm': 188.1,  # ~1.025 ratio
        'leg_length_cm': 96.3,  # ~0.525 ratio
        'torso_length_cm': 87.2,  # ~0.475 ratio
        'age': 26.2,
        'bmi': 23.3
    },
    'gymnastics': {
        'height_cm': 155.0,  # Female gymnasts: 150-160 cm
        'weight_kg': 51.0,   # Female gymnasts: 47-55 kg
        'arm_span_cm': 155.0,  # 1.00 ratio of height
        'leg_length_cm': 77.5,  # ~0.50 ratio
        'torso_length_cm': 77.5,  # ~0.50 ratio
        'age': 20.5,
        'bmi': 21.2
    },
    'swimming': {
        'height_cm': 178.5,  # Female swimmers: 175-182 cm
        'weight_kg': 68.5,   # Female swimmers: 65-72 kg
        'arm_span_cm': 186.5,  # ~1.045 ratio
        'leg_length_cm': 85.3,  # ~0.478 ratio
        'torso_length_cm': 93.2,  # ~0.522 ratio
        'age': 23.8,
        'bmi': 21.5
    },
    'track_field': {
        'height_cm': 170.0,  # Female sprinters: 165-175 cm
        'weight_kg': 61.0,   # Female sprinters: 57-65 kg
        'arm_span_cm': 170.0,  # 1.00 ratio
        'leg_length_cm': 89.3,  # ~0.525 ratio
        'torso_length_cm': 80.7,  # ~0.475 ratio
        'age': 24.2,
        'bmi': 21.1
    },
    'distance_running': {
        'height_cm': 162.5,  # Female distance runners: 158-167 cm
        'weight_kg': 48.5,   # Female distance runners: 45-52 kg
        'arm_span_cm': 162.5,  # 1.00 ratio
        'leg_length_cm': 85.3,  # ~0.525 ratio
        'torso_length_cm': 77.2,  # ~0.475 ratio
        'age': 25.8,
        'bmi': 18.4
    },
    'soccer': {
        'height_cm': 167.0,  # Female soccer: 164-170 cm
        'weight_kg': 61.0,   # Female soccer: 58-64 kg
        'arm_span_cm': 167.0,  # 1.00 ratio
        'leg_length_cm': 87.7,  # ~0.525 ratio
        'torso_length_cm': 79.3,  # ~0.475 ratio
        'age': 24.5,
        'bmi': 21.9
    },
    'volleyball': {
        'height_cm': 185.0,  # Female volleyball: 182-188 cm
        'weight_kg': 73.0,   # Female volleyball: 70-76 kg
        'arm_span_cm': 194.3,  # ~1.05 ratio
        'leg_length_cm': 97.1,  # ~0.525 ratio
        'torso_length_cm': 87.9,  # ~0.475 ratio
        'spike_reach_cm': 310.0,  # Average spike reach for elite females
        'block_reach_cm': 295.0,  # Average block reach for elite females
        'age': 25.2,
        'bmi': 21.3
    },
    'weightlifting': {
        'height_cm': 163.0,  # Female weightlifting: 158-168 cm
        'weight_kg': 71.5,   # Female weightlifting: 68-75 kg
        'arm_span_cm': 163.0,  # 1.00 ratio
        'leg_length_cm': 81.5,  # ~0.50 ratio
        'torso_length_cm': 81.5,  # ~0.50 ratio
        'age': 26.8,
        'bmi': 26.9
    },
    'tennis': {
        'height_cm': 176.0,  # Female tennis: 174-178 cm
        'weight_kg': 67.5,   # Female tennis: 65-70 kg
        'arm_span_cm': 176.0,  # 1.00 ratio
        'leg_length_cm': 92.4,  # ~0.525 ratio
        'torso_length_cm': 83.6,  # ~0.475 ratio
        'age': 25.3,
        'bmi': 21.8
    },
    'rowing': {
        'height_cm': 180.0,  # Female rowing: 177-183 cm
        'weight_kg': 75.0,   # Female rowing: 70-80 kg
        'arm_span_cm': 189.0,  # ~1.05 ratio
        'leg_length_cm': 93.6,  # ~0.52 ratio
        'torso_length_cm': 86.4,  # ~0.48 ratio
        'age': 26.5,
        'bmi': 23.1
    }
}

# Standard deviations for generating realistic variations
SPORT_STANDARD_DEVIATIONS = {
    'basketball': {
        'height_cm': 7.6,
        'weight_kg': 12.7,
        'wingspan_cm': 8.9,
        'hand_length_cm': 1.3,
        'hand_width_cm': 1.0,
        'age': 4.2,
        'bmi': 2.1
    },
    'swimming': {
        'height_cm': 6.4,
        'weight_kg': 8.6,
        'wingspan_cm': 7.1,
        'torso_length_cm': 5.7,  # Standard deviation for torso length
        'age': 3.8,
        'bmi': 1.8
    },
    'soccer': {
        'height_cm': 8.2,  # Based on data range (168-195 cm)
        'weight_kg': 9.1,  # Based on data range (64-95 kg)
        'arm_span_cm': 8.2,  # Same as height std
        'leg_length_cm': 4.3,  # Standard deviation for leg length
        'torso_length_cm': 3.9,  # Standard deviation for torso length
        'age': 4.1,
        'bmi': 1.5
    },
    'tennis': {
        'height_cm': 12.0,  # Based on data range (167-211 cm) and gender differences
        'weight_kg': 13.5,  # Based on data range (59-109 kg) and gender differences
        'arm_span_cm': 12.0,  # Same as height std
        'leg_length_cm': 6.3,  # Standard deviation for leg length
        'torso_length_cm': 5.7,  # Standard deviation for torso length
        'age': 4.5,
        'bmi': 1.9
    },
    'track_field': {
        'height_cm': 6.9,  # From study data (180.45 ± 6.9 cm)
        'weight_kg': 7.7,  # From study data (78.8 ± 7.7 kg)
        'arm_span_cm': 6.9,  # Same as height std
        'leg_length_cm': 4.5,  # Standard deviation for leg length
        'torso_length_cm': 4.5,  # Standard deviation for torso length
        'age': 4.7,
        'bmi': 2.3
    },
    'gymnastics': {
        'height_cm': 5.1,
        'weight_kg': 6.8,
        'arm_span_cm': 5.1,  # Same as height std
        'leg_length_cm': 3.8,  # Standard deviation for leg length
        'torso_length_cm': 4.1,  # Standard deviation for torso length
        'age': 2.9,
        'bmi': 1.4
    },
    'volleyball': {
        'height_cm': 8.9,  # Based on data range (186-218 cm)
        'weight_kg': 10.9,  # Based on data range (76-110 kg)
        'wingspan_cm': 10.2,
        'spike_reach_cm': 10.0,  # Standard deviation for spike reach
        'block_reach_cm': 7.5,   # Standard deviation for block reach
        'leg_length_cm': 4.7,    # Standard deviation for leg length
        'torso_length_cm': 4.2,  # Standard deviation for torso length
        'age': 4.3,
        'bmi': 1.7
    },
    'baseball': {
        'height_cm': 6.4,
        'weight_kg': 11.3,
        'age': 5.1,
        'bmi': 2.4
    },
    'hockey': {
        'height_cm': 5.1,
        'weight_kg': 9.1,
        'age': 4.8,
        'bmi': 2.0
    },
    'cycling': {
        'height_cm': 5.1,
        'weight_kg': 6.8,
        'age': 5.4,
        'bmi': 1.6
    },
    'distance_running': {
        'height_cm': 6.0,  # Standard deviation for distance runners
        'weight_kg': 4.0,  # Lower weight variation (lean builds)
        'arm_span_cm': 6.0,  # Same as height std
        'leg_length_cm': 4.7,  # Standard deviation for leg length
        'torso_length_cm': 4.0,  # Standard deviation for torso length
        'age': 4.8,
        'bmi': 1.2  # Lower BMI variation
    },
    'weightlifting': {
        'height_cm': 12.5,  # Based on data range (147-197 cm)
        'weight_kg': 35.0,  # High variation due to weight classes (52-183 kg)
        'arm_span_cm': 12.5,  # Same as height std
        'leg_length_cm': 6.3,  # Standard deviation for leg length
        'torso_length_cm': 6.3,  # Standard deviation for torso length
        'age': 5.2,
        'bmi': 4.5  # Higher BMI variation due to muscle mass differences
    },
    'rowing': {
        'height_cm': 8.5,  # Based on data range (177-200 cm) and gender differences
        'weight_kg': 12.0,  # Based on data range (69-103 kg) and gender differences
        'arm_span_cm': 9.0,  # Slightly higher for reach variation
        'leg_length_cm': 4.4,  # Standard deviation for leg length
        'torso_length_cm': 4.0,  # Standard deviation for torso length
        'age': 5.0,
        'bmi': 2.1
    }
}


# Clean structure - remove all the corrupted content above


def get_sport_average(sport: str, measurement: str, gender: str = 'M') -> float:
    """
    Get the professional average for a specific measurement in a sport, gender-aware.
    
    Args:
        sport: Sport name
        measurement: Measurement name (e.g., 'height_cm', 'weight_kg')
        gender: 'M' for male, 'F' for female
    
    Returns:
        Average value for the measurement
    """
    # Choose the appropriate averages based on gender
    if gender.upper() == 'F':
        sport_averages = SPORT_AVERAGES_FEMALE
    else:
        sport_averages = SPORT_AVERAGES_MALE
    
    if sport in sport_averages and measurement in sport_averages[sport]:
        return sport_averages[sport][measurement]
    
    # Default fallback values if sport/measurement not found
    if gender.upper() == 'F':
        defaults = {
            'height_cm': 168.0,
            'weight_kg': 62.0,
            'age': 25.0,
            'bmi': 22.0
        }
    else:
        defaults = {
            'height_cm': 180.0,
            'weight_kg': 78.0,
            'age': 26.0,
            'bmi': 24.0
        }
    
    return defaults.get(measurement, 0.0)


def get_sport_std(sport: str, measurement: str) -> float:
    """
    Get the standard deviation for a specific measurement in a sport.
    
    Args:
        sport: Sport name
        measurement: Measurement name
    
    Returns:
        Standard deviation for that measurement
    """
    sport = sport.lower()
    if sport not in SPORT_STANDARD_DEVIATIONS:
        sport = 'basketball'
    
    return SPORT_STANDARD_DEVIATIONS[sport].get(measurement, 1.0)


def generate_realistic_measurement(sport: str, measurement: str, 
                                 existing_data: list = None, 
                                 use_data_distribution: bool = True) -> float:
    """
    Generate a realistic measurement value using professional averages
    and optionally incorporating existing data distribution.
    
    Args:
        sport: Sport name
        measurement: Measurement to generate
        existing_data: List of existing values for this measurement
        use_data_distribution: Whether to blend with existing data stats
    
    Returns:
        Generated measurement value
    """
    sport_avg = get_sport_average(sport, measurement)
    sport_std = get_sport_std(sport, measurement)
    
    if existing_data and use_data_distribution and len(existing_data) > 2:
        # Blend professional averages with existing data distribution
        existing_data = [float(x) for x in existing_data if pd.notna(x)]
        if len(existing_data) > 0:
            data_avg = np.mean(existing_data)
            data_std = np.std(existing_data) if len(existing_data) > 1 else sport_std
            
            # Weight the averages (70% existing data, 30% professional average)
            blended_avg = 0.7 * data_avg + 0.3 * sport_avg
            blended_std = 0.7 * data_std + 0.3 * sport_std
            
            # Generate value from blended distribution
            value = np.random.normal(blended_avg, blended_std)
        else:
            # Use professional averages only
            value = np.random.normal(sport_avg, sport_std)
    else:
        # Use professional averages only
        value = np.random.normal(sport_avg, sport_std)
    
    # Apply reasonable bounds based on measurement type
    if measurement == 'height_cm':
        value = max(140, min(240, value))  # 4'7" to 7'10"
    elif measurement == 'weight_kg':
        value = max(40, min(200, value))   # 88 to 440 lbs
    elif measurement == 'wingspan_cm':
        value = max(140, min(260, value))  # 4'7" to 8'6"
    elif measurement == 'age':
        value = max(16, min(45, int(value)))  # 16 to 45 years
    elif measurement == 'bmi':
        value = max(15, min(35, value))    # Reasonable BMI range
    elif 'hand' in measurement:
        value = max(15, min(35, value))    # Hand measurements in cm
    elif measurement == 'torso_length_cm':
        value = max(80, min(130, value))   # Torso length in cm
    elif measurement == 'arm_span_cm':
        value = max(140, min(260, value))  # Arm span in cm
    elif measurement == 'leg_length_cm':
        value = max(60, min(120, value))   # Leg length in cm
    elif measurement == 'spike_reach_cm':
        value = max(300, min(400, value))  # Spike reach in cm
    elif measurement == 'block_reach_cm':
        value = max(280, min(380, value))  # Block reach in cm
    
    return value


def get_all_sport_averages() -> Dict[str, Dict[str, float]]:
    """Return all sport averages."""
    return SPORT_AVERAGES.copy()


def get_all_sport_stds() -> Dict[str, Dict[str, float]]:
    """Return all sport standard deviations."""
    return SPORT_STANDARD_DEVIATIONS.copy()
