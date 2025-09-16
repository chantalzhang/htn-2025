#!/usr/bin/env python3
"""
Example usage of the Sport Recommendation Engine
Shows how to get sport recommendations and find similar athletes.
"""

from sport_recommendation_engine import SportRecommendationEngine

def main():
    """Example usage of the recommendation engine."""
    print("🎯 SPORT RECOMMENDATION ENGINE - EXAMPLE USAGE")
    print("=" * 60)
    
    # Initialize the engine
    engine = SportRecommendationEngine()
    engine.initialize()
    
    # Example 1: Get sport recommendation only
    print("\n📋 Example 1: Sport Recommendation Only")
    print("-" * 40)
    
    sport_rec = engine.recommend_sport(
        gender='M',
        height_cm=185,
        weight_kg=80,
        arm_span=190
    )
    
    print(f"🏆 Recommended Sport: {sport_rec['recommended_sport']}")
    print(f"📊 Cluster: {sport_rec['cluster_id']} ({sport_rec['cluster_size']} athletes)")
    print(f"📈 All top sports in cluster: {sport_rec['all_top_sports']}")
    
    # Example 2: Find similar athlete only
    print("\n👤 Example 2: Find Similar Athlete Only")
    print("-" * 40)
    
    similar_athlete = engine.find_similar_athlete(
        gender='F',
        height_cm=165,
        weight_kg=55,
        preferred_sport='gymnastics'
    )
    
    if similar_athlete['similar_athlete']:
        athlete = similar_athlete['similar_athlete']
        print(f"👤 Most Similar Athlete: {athlete['Player']}")
        print(f"   Sport: {athlete['sport']}")
        print(f"   Height: {athlete['height_cm']:.1f}cm, Weight: {athlete['weight_kg']:.1f}kg")
        print(f"   Similarity Distance: {similar_athlete['similarity_distance']:.3f}")
    else:
        print("No similar athlete found")
    
    # Example 3: Get complete recommendation
    print("\n🎯 Example 3: Complete Recommendation")
    print("-" * 40)
    
    full_rec = engine.get_full_recommendation(
        gender='F',
        height_cm=170,
        weight_kg=60,
        arm_span=172,
        leg_length=76,
        torso_length=51
    )
    
    sport_rec = full_rec['sport_recommendation']
    similar_athlete = full_rec['similar_athlete']
    
    print(f"🏆 Recommended Sport: {sport_rec['recommended_sport']}")
    print(f"📊 Cluster: {sport_rec['cluster_id']} ({sport_rec['cluster_size']} athletes)")
    
    if similar_athlete['similar_athlete']:
        athlete = similar_athlete['similar_athlete']
        print(f"👤 Similar Athlete: {athlete['Player']} ({athlete['sport']})")
        print(f"   Height: {athlete['height_cm']:.1f}cm, Weight: {athlete['weight_kg']:.1f}kg")
        print(f"   Similarity Distance: {similar_athlete['similarity_distance']:.3f}")
    
    # Example 4: Interactive input (commented out for demo)
    print("\n💡 Example 4: How to use with user input")
    print("-" * 40)
    print("""
    # Get user input
    gender = input("Enter gender (M/F): ")
    height = float(input("Enter height in cm: "))
    weight = float(input("Enter weight in kg: "))
    
    # Get recommendation
    recommendation = engine.get_full_recommendation(
        gender=gender,
        height_cm=height,
        weight_kg=weight
    )
    
    # Display results
    print(f"Recommended Sport: {recommendation['sport_recommendation']['recommended_sport']}")
    print(f"Similar Athlete: {recommendation['similar_athlete']['similar_athlete']['Player']}")
    """)
    
    print("\n✅ Example usage complete!")
    print("\n📝 Key Features:")
    print("   • Single sport recommendation (randomly selected from top sports)")
    print("   • Most similar athlete (prioritizes recommended sport when possible)")
    print("   • Handles missing measurements with intelligent defaults")
    print("   • Works with male, female, or combined models")
    print("   • Returns detailed cluster information")

if __name__ == "__main__":
    main()
