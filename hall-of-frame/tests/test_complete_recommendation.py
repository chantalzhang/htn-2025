#!/usr/bin/env python3
"""
Test Complete Recommendation System
Shows sport recommendations with descriptions and stats.
"""

from sport_recommendation_engine import SportRecommendationEngine

def main():
    """Test the complete recommendation system."""
    print("🎯 COMPLETE RECOMMENDATION SYSTEM TEST")
    print("=" * 60)
    
    # Initialize the engine
    engine = SportRecommendationEngine()
    engine.initialize()
    
    # Test cases with different body types
    test_cases = [
        {
            'name': 'Tall Male Basketball Player',
            'gender': 'M',
            'height_cm': 200,
            'weight_kg': 95,
            'arm_span': 210
        },
        {
            'name': 'Compact Female Gymnast',
            'gender': 'F',
            'height_cm': 160,
            'weight_kg': 50,
            'arm_span': 158
        },
        {
            'name': 'Lean Female Runner',
            'gender': 'F',
            'height_cm': 170,
            'weight_kg': 55,
            'arm_span': 172
        },
        {
            'name': 'Powerful Male Weightlifter',
            'gender': 'M',
            'height_cm': 175,
            'weight_kg': 100,
            'arm_span': 180
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            # Get complete recommendation
            recommendation = engine.get_full_recommendation(
                gender=test_case['gender'],
                height_cm=test_case['height_cm'],
                weight_kg=test_case['weight_kg'],
                arm_span=test_case.get('arm_span')
            )
            
            sport_rec = recommendation['sport_recommendation']
            similar_athlete = recommendation['similar_athlete']
            
            # Display sport recommendation
            print(f"🏆 RECOMMENDED SPORT: {sport_rec['sport_name']}")
            print(f"📝 Description: {sport_rec['sport_description']}")
            print(f"📊 Athletic Stats:")
            if sport_rec['sport_stats']:
                for stat, value in sport_rec['sport_stats'].items():
                    print(f"   {stat.capitalize()}: {value}")
            else:
                print("   No stats available")
            
            print(f"\n📈 Cluster Info:")
            print(f"   Cluster: {sport_rec['cluster_id']} ({sport_rec['cluster_size']} athletes)")
            print(f"   All top sports: {sport_rec['all_top_sports']}")
            
            # Display similar athlete
            print(f"\n👤 SIMILAR ATHLETE:")
            if similar_athlete['similar_athlete']:
                athlete = similar_athlete['similar_athlete']
                print(f"   Name: {athlete['Player']}")
                print(f"   Sport: {athlete['sport']}")
                print(f"   Height: {athlete['height_cm']:.1f}cm")
                print(f"   Weight: {athlete['weight_kg']:.1f}kg")
                print(f"   BMI: {athlete['bmi']:.1f}")
                print(f"   Similarity Distance: {similar_athlete['similarity_distance']:.3f}")
            else:
                print("   No similar athlete found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Complete recommendation system test finished!")
    print(f"\n📋 Summary of Features:")
    print(f"   • Sport recommendation with name and description")
    print(f"   • Athletic stats (0-100 scale) for SpiderChart")
    print(f"   • Most similar athlete with measurements")
    print(f"   • Cluster information and analysis")
    print(f"   • Ready for frontend integration")

if __name__ == "__main__":
    main()
