#!/usr/bin/env python3
"""
Sport Database
Contains descriptions and athletic stats for each sport.
"""

SPORT_DATABASE = {
    'basketball': {
        'name': 'Basketball',
        'description': 'Height and wingspan give players a major advantage in rebounding, blocking, and shooting. Lean muscle mass and agility allow for explosive moves and quick changes of direction. Leg length relative to torso helps with stride efficiency and vertical jumping ability.',
        'stats': {
            'strength': 75,
            'agility': 85,
            'endurance': 70,
            'power': 80,
            'speed': 80,
            'flexibility': 65,
            'coordination': 85,
            'balance': 80
        }
    },
    'swimming': {
        'name': 'Swimming',
        'description': 'A long torso improves buoyancy and stroke efficiency, while an arm span greater than height increases pulling power. Larger hands and feet act like natural paddles and fins. Having low but stable body fat helps with hydrodynamics and staying afloat efficiently.',
        'stats': {
            'strength': 70,
            'agility': 65,
            'endurance': 85,
            'power': 80,
            'speed': 80,
            'flexibility': 75,
            'coordination': 85,
            'balance': 70
        }
    },
    'gymnastics': {
        'name': 'Gymnastics',
        'description': 'A compact body makes flips and spins easier to control. Strong shoulders and core provide the strength to hold bodyweight in difficult positions. Flexibility in the hips, spine, and shoulders is crucial for extreme ranges of motion.',
        'stats': {
            'strength': 80,
            'agility': 95,
            'endurance': 65,
            'power': 85,
            'speed': 75,
            'flexibility': 95,
            'coordination': 95,
            'balance': 95
        }
    },
    'track_field': {
        'name': 'Sprint Running (100m/200m)',
        'description': 'Fast-twitch muscle fibers drive acceleration and top-end speed. Optimal leg length provides long but quick strides. A lean body with well-distributed muscle mass maximizes the power-to-weight ratio for explosive sprinting.',
        'stats': {
            'strength': 75,
            'agility': 70,
            'endurance': 50,
            'power': 95,
            'speed': 100,
            'flexibility': 65,
            'coordination': 85,
            'balance': 75
        }
    },
    'distance_running': {
        'name': 'Distance Running (Marathon)',
        'description': 'A slim, lightweight frame increases running efficiency and oxygen use. Long legs relative to the torso create economical strides. High lung capacity and VO2 max are biological advantages. Thin calves and ankles reduce energy loss per stride.',
        'stats': {
            'strength': 55,
            'agility': 65,
            'endurance': 100,
            'power': 55,
            'speed': 65,
            'flexibility': 60,
            'coordination': 70,
            'balance': 70
        }
    },
    'soccer': {
        'name': 'Soccer',
        'description': 'Balanced height and leg strength provide kicking power while supporting quick changes of direction. A strong core and hips improve stability during tackles and dribbles. High aerobic capacity supports covering large distances throughout a match.',
        'stats': {
            'strength': 70,
            'agility': 85,
            'endurance': 85,
            'power': 80,
            'speed': 85,
            'flexibility': 70,
            'coordination': 90,
            'balance': 85
        }
    },
    'volleyball': {
        'name': 'Volleyball',
        'description': 'Height and long arms are major advantages for spiking and blocking at the net. Explosive leg strength is critical for vertical jumping. Quick lateral movement is necessary for defensive plays and serve reception.',
        'stats': {
            'strength': 70,
            'agility': 80,
            'endurance': 70,
            'power': 85,
            'speed': 75,
            'flexibility': 65,
            'coordination': 85,
            'balance': 80
        }
    },
    'weightlifting': {
        'name': 'Weightlifting',
        'description': 'Shorter limbs relative to torso reduce the bar path distance and improve leverage. Strong wrists and ankles allow better load support. High muscle mass and density maximize absolute strength and force output.',
        'stats': {
            'strength': 100,
            'agility': 60,
            'endurance': 50,
            'power': 95,
            'speed': 65,
            'flexibility': 75,
            'coordination': 80,
            'balance': 85
        }
    },
    'tennis': {
        'name': 'Tennis',
        'description': 'Longer arms provide better reach and serve angles. Strong legs and core deliver explosive strokes and stability. Agility and fast reaction time allow quick adjustments to unpredictable ball trajectories.',
        'stats': {
            'strength': 70,
            'agility': 90,
            'endurance': 80,
            'power': 75,
            'speed': 85,
            'flexibility': 70,
            'coordination': 90,
            'balance': 85
        }
    },
    'rowing': {
        'name': 'Rowing',
        'description': 'Long arms and legs allow greater stroke length and leverage. A tall torso improves pulling power on each stroke. High aerobic capacity and muscular endurance are essential for sustained power output.',
        'stats': {
            'strength': 85,
            'agility': 65,
            'endurance': 90,
            'power': 85,
            'speed': 70,
            'flexibility': 70,
            'coordination': 80,
            'balance': 75
        }
    }
}

def get_sport_info(sport_key: str) -> dict:
    """
    Get sport information including name, description, and stats.
    
    Args:
        sport_key: The sport key (e.g., 'basketball', 'swimming')
        
    Returns:
        Dictionary with sport information or None if not found
    """
    return SPORT_DATABASE.get(sport_key.lower())

def get_sport_stats(sport_key: str) -> dict:
    """
    Get just the athletic stats for a sport.
    
    Args:
        sport_key: The sport key (e.g., 'basketball', 'swimming')
        
    Returns:
        Dictionary with athletic stats or None if not found
    """
    sport_info = get_sport_info(sport_key)
    return sport_info['stats'] if sport_info else None

def get_sport_description(sport_key: str) -> str:
    """
    Get just the description for a sport.
    
    Args:
        sport_key: The sport key (e.g., 'basketball', 'swimming')
        
    Returns:
        String description or None if not found
    """
    sport_info = get_sport_info(sport_key)
    return sport_info['description'] if sport_info else None

def get_sport_name(sport_key: str) -> str:
    """
    Get the display name for a sport.
    
    Args:
        sport_key: The sport key (e.g., 'basketball', 'swimming')
        
    Returns:
        String name or None if not found
    """
    sport_info = get_sport_info(sport_key)
    return sport_info['name'] if sport_info else None

def list_all_sports() -> list:
    """
    Get a list of all available sports.
    
    Returns:
        List of sport keys
    """
    return list(SPORT_DATABASE.keys())

def main():
    """Test the sport database."""
    print("🏆 SPORT DATABASE TEST")
    print("=" * 40)
    
    # Test getting sport info
    basketball_info = get_sport_info('basketball')
    print(f"Basketball Info: {basketball_info['name']}")
    print(f"Description: {basketball_info['description'][:100]}...")
    print(f"Stats: {basketball_info['stats']}")
    
    print("\n" + "=" * 40)
    
    # Test getting just stats
    swimming_stats = get_sport_stats('swimming')
    print(f"Swimming Stats: {swimming_stats}")
    
    print("\n" + "=" * 40)
    
    # Test getting just description
    gymnastics_desc = get_sport_description('gymnastics')
    print(f"Gymnastics Description: {gymnastics_desc[:100]}...")
    
    print("\n" + "=" * 40)
    
    # List all sports
    all_sports = list_all_sports()
    print(f"All available sports: {all_sports}")
    
    print("\n✅ Sport database test complete!")

if __name__ == "__main__":
    main()
