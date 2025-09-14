"""
Professional athlete average measurements by sport - gender-specific.
These are fallback values for missing data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

# Male professional sport averages
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
    'gymnastics': {
        'height_cm': 168.5,  # Male gymnasts: 165-172 cm
        'weight_kg': 66.0,   # Male gymnasts: 62-70 kg
        'arm_span_cm': 168.5,  # 1.00 ratio of height
        'leg_length_cm': 84.3,  # ~0.50 ratio (balanced proportions)
        'torso_length_cm': 84.3,  # ~0.50 ratio (strong core)
        'age': 23.8,
        'bmi': 23.3
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
    'distance_running': {
        'height_cm': 171.5,  # Male distance runners: 168-175 cm
        'weight_kg': 57.0,   # Male distance runners: 54-60 kg
        'arm_span_cm': 171.5,  # 1.00 ratio of height
        'leg_length_cm': 90.0,  # ~0.525 ratio (longer legs for distance)
        'torso_length_cm': 81.5,  # ~0.475 ratio (ectomorphic build)
        'age': 26.5,
        'bmi': 19.4
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
    'weightlifting': {
        'height_cm': 173.0,  # Male weightlifting: 168-178 cm
        'weight_kg': 89.5,   # Male weightlifting: 85-94 kg
        'arm_span_cm': 173.0,  # 1.00 ratio of height
        'leg_length_cm': 86.5,  # ~0.50 ratio (balanced for power generation)
        'torso_length_cm': 86.5,  # ~0.50 ratio (strong core for lifting)
        'age': 27.5,
        'bmi': 29.9
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

# Female professional sport averages
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
    'swimming': {
        'height_cm': 178.5,  # Female swimmers: 175-182 cm
        'weight_kg': 68.5,   # Female swimmers: 65-72 kg
        'arm_span_cm': 186.5,  # ~1.045 ratio
        'leg_length_cm': 85.3,  # ~0.478 ratio
        'torso_length_cm': 93.2,  # ~0.522 ratio
        'age': 23.8,
        'bmi': 21.5
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
        'arm_span_cm': 8.9,
        'leg_length_cm': 4.0,
        'torso_length_cm': 3.6,
        'age': 4.2,
        'bmi': 1.8
    },
    'swimming': {
        'height_cm': 6.8,
        'weight_kg': 8.5,
        'arm_span_cm': 7.1,
        'leg_length_cm': 3.2,
        'torso_length_cm': 3.7,
        'age': 4.1,
        'bmi': 1.4
    },
    'gymnastics': {
        'height_cm': 5.2,
        'weight_kg': 6.8,
        'arm_span_cm': 5.2,
        'leg_length_cm': 2.6,
        'torso_length_cm': 2.6,
        'age': 3.8,
        'bmi': 1.9
    },
    'soccer': {
        'height_cm': 8.2,
        'weight_kg': 9.1,
        'arm_span_cm': 8.2,
        'leg_length_cm': 4.3,
        'torso_length_cm': 3.9,
        'age': 4.1,
        'bmi': 1.5
    },
    'tennis': {
        'height_cm': 12.0,
        'weight_kg': 13.5,
        'arm_span_cm': 12.0,
        'leg_length_cm': 6.3,
        'torso_length_cm': 5.7,
        'age': 4.5,
        'bmi': 1.9
    },
    'track_field': {
        'height_cm': 6.9,
        'weight_kg': 7.7,
        'arm_span_cm': 6.9,
        'leg_length_cm': 4.5,
        'torso_length_cm': 4.5,
        'age': 4.7,
        'bmi': 1.8
    },
    'volleyball': {
        'height_cm': 7.8,
        'weight_kg': 10.2,
        'arm_span_cm': 8.2,
        'spike_reach_cm': 12.5,
        'block_reach_cm': 7.5,
        'leg_length_cm': 4.7,
        'torso_length_cm': 4.2,
        'age': 4.3,
        'bmi': 1.7
    },
    'distance_running': {
        'height_cm': 6.0,
        'weight_kg': 4.0,
        'arm_span_cm': 6.0,
        'leg_length_cm': 4.7,
        'torso_length_cm': 4.0,
        'age': 4.8,
        'bmi': 1.2
    },
    'weightlifting': {
        'height_cm': 12.5,
        'weight_kg': 35.0,
        'arm_span_cm': 12.5,
        'leg_length_cm': 6.3,
        'torso_length_cm': 6.3,
        'age': 5.2,
        'bmi': 4.5
    },
    'rowing': {
        'height_cm': 8.5,
        'weight_kg': 12.0,
        'arm_span_cm': 9.0,
        'leg_length_cm': 4.4,
        'torso_length_cm': 4.0,
        'age': 5.0,
        'bmi': 2.1
    }
}


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
        Standard deviation for the measurement
    """
    if sport in SPORT_STANDARD_DEVIATIONS and measurement in SPORT_STANDARD_DEVIATIONS[sport]:
        return SPORT_STANDARD_DEVIATIONS[sport][measurement]
    
    # Default standard deviations if not found
    defaults = {
        'height_cm': 8.0,
        'weight_kg': 10.0,
        'age': 4.0,
        'bmi': 2.0
    }
    
    return defaults.get(measurement, 1.0)


def get_measurement_bounds(measurement: str) -> tuple:
    """
    Get realistic bounds for measurements to prevent unrealistic values.
    
    Args:
        measurement: Measurement name
    
    Returns:
        Tuple of (min_value, max_value)
    """
    bounds = {
        'height_cm': (140, 230),
        'weight_kg': (40, 200),
        'arm_span_cm': (140, 250),
        'leg_length_cm': (60, 130),
        'torso_length_cm': (60, 120),
        'spike_reach_cm': (250, 400),
        'block_reach_cm': (240, 380),
        'age': (16, 45),
        'bmi': (15, 40)
    }
    
    return bounds.get(measurement, (0, 1000))
