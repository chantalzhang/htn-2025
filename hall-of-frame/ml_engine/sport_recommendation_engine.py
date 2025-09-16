#!/usr/bin/env python3
"""
Sport Recommendation Engine
Takes user body measurements and recommends the best sport and similar athlete.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import random
import json
from typing import Dict, List, Tuple, Optional
from sport_database import get_sport_info, get_sport_stats, get_sport_description, get_sport_name, SPORT_DATABASE

class SportRecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine."""
        self.df = None
        self.df_weighted = None
        self.feature_cols = None
        self.cluster_models = {}
        self.cluster_stats = {}
        self.scalers = {}
        self.is_initialized = False
        
    def load_and_preprocess_data(self):
        """Load and preprocess athlete data with enhanced feature engineering."""
        print("📊 Loading athlete dataset...")
        # Try different paths for the dataset
        import os
        dataset_paths = [
            'athlete_dataset_pipeline/athlete_dataset_merged.csv',
            '../athlete_dataset_pipeline/athlete_dataset_merged.csv',
            'hall-of-frame/athlete_dataset_pipeline/athlete_dataset_merged.csv'
        ]
        
        for path in dataset_paths:
            if os.path.exists(path):
                self.df = pd.read_csv(path)
                print(f"📊 Loaded dataset from: {path}")
                break
        else:
            raise FileNotFoundError("Could not find athlete_dataset_merged.csv in any expected location")
        
        # Enhanced feature engineering for better body-type separation
        self.df['bmi'] = self.df['weight_kg'] / (self.df['height_cm'] / 100) ** 2
        self.df['weight_height_ratio'] = self.df['weight_kg'] / self.df['height_cm']
        self.df['height_weight_ratio'] = self.df['height_cm'] / self.df['weight_kg']
        self.df['arm_span_ratio'] = self.df['Arm Span'] / self.df['height_cm']
        self.df['leg_length_ratio'] = self.df['Leg Length'] / self.df['height_cm']
        self.df['torso_length_ratio'] = self.df['Torso Length'] / self.df['height_cm']
        
        # Advanced body-type features
        self.df['power_index'] = (self.df['weight_kg'] * self.df['bmi']) / 1000
        self.df['endurance_index'] = self.df['height_cm'] / (self.df['weight_kg'] * self.df['bmi']) * 1000
        self.df['reach_advantage'] = (self.df['arm_span_ratio'] - 1.0) * 100
        self.df['build_compactness'] = self.df['weight_kg'] / (self.df['height_cm'] * self.df['height_cm']) * 10000
        
        # Handle missing values
        for col in ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                    'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio',
                    'power_index', 'endurance_index', 'reach_advantage', 'build_compactness']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        print(f"✅ Loaded {len(self.df)} athletes")
        
    def apply_balanced_weighting(self):
        """Apply weighting optimized for balanced clusters."""
        print("⚖️ Applying balanced clustering optimization...")
        
        core_features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                        'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
        advanced_features = ['power_index', 'endurance_index', 'reach_advantage', 'build_compactness']
        
        # Balanced weighting that prevents mega-clusters
        balanced_weights = {
            'height_cm': 1.6,
            'power_index': 1.8,
            'endurance_index': 1.7,
            'reach_advantage': 1.5,
            'build_compactness': 1.6,
            'arm_span_ratio': 1.4,
            'torso_length_ratio': 1.3,
            'bmi': 1.3,
            'weight_height_ratio': 1.2,
            'leg_length_ratio': 1.2,
            'height_weight_ratio': 1.1,
            'weight_kg': 1.1,
        }
        
        self.df_weighted = self.df.copy()
        self.feature_cols = core_features + advanced_features
        
        for feature in self.feature_cols:
            if feature in self.df_weighted.columns:
                weight = balanced_weights.get(feature, 1.0)
                self.df_weighted[f'weighted_{feature}'] = self.df_weighted[feature] * weight
        
    def train_cluster_models(self):
        """Train clustering models for different groups."""
        print("🎯 Training clustering models...")
        
        # Prepare weighted features
        weighted_feature_cols = [f'weighted_{col}' for col in self.feature_cols]
        X = self.df_weighted[weighted_feature_cols].copy()
        X = X.fillna(0)
        
        # Train models for different groups
        groups = [
            ('combined', self.df_weighted, 6),
            ('male', self.df_weighted[self.df_weighted['Sex'] == 'M'], 5),
            ('female', self.df_weighted[self.df_weighted['Sex'] == 'F'], 4)
        ]
        
        for group_name, group_df, k in groups:
            if len(group_df) < k:
                print(f"⚠️ Skipping {group_name}: Not enough data ({len(group_df)} < {k})")
                continue
                
            print(f"   Training {group_name} model...")
            
            # Prepare data
            X_group = X.loc[group_df.index]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_group)
            
            # Train K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Store models and data
            self.cluster_models[group_name] = {
                'kmeans': kmeans,
                'scaler': scaler,
                'cluster_labels': cluster_labels,
                'df': group_df,
                'X_scaled': X_scaled
            }
            
            # Analyze clusters
            self.cluster_stats[group_name] = self._analyze_clusters(group_df, cluster_labels)
            
        print("✅ All models trained successfully")
        
    def _analyze_clusters(self, df, cluster_labels):
        """Analyze clusters to get sport recommendations."""
        cluster_stats = {}
        
        for cluster_id in sorted(set(cluster_labels)):
            cluster_data = df[cluster_labels == cluster_id]
            
            # Top sports in this cluster
            sport_counts = cluster_data['sport'].value_counts()
            top_sports = sport_counts.head(3)
            
            # Calculate cluster center characteristics
            avg_height = cluster_data['height_cm'].mean()
            avg_weight = cluster_data['weight_kg'].mean()
            avg_bmi = cluster_data['bmi'].mean()
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_data),
                'top_sports': top_sports.to_dict(),
                'avg_height': avg_height,
                'avg_weight': avg_weight,
                'avg_bmi': avg_bmi,
                'athletes': cluster_data[['Player', 'sport', 'height_cm', 'weight_kg', 'bmi']].to_dict('records')
            }
            
        return cluster_stats
        
    def initialize(self):
        """Initialize the recommendation engine."""
        if self.is_initialized:
            return
            
        self.load_and_preprocess_data()
        self.apply_balanced_weighting()
        self.train_cluster_models()
        self.is_initialized = True
        print("🚀 Sport Recommendation Engine initialized!")
        
    def get_user_features(self, gender: str, height_cm: float, weight_kg: float, 
                         arm_span: Optional[float] = None, leg_length: Optional[float] = None,
                         torso_length: Optional[float] = None) -> np.ndarray:
        """Convert user input to feature vector."""
        # Calculate derived features
        bmi = weight_kg / (height_cm / 100) ** 2
        weight_height_ratio = weight_kg / height_cm
        height_weight_ratio = height_cm / weight_kg
        
        # Use provided measurements or estimate from height
        if arm_span is None:
            arm_span = height_cm * 1.03  # Average arm span is ~103% of height
        if leg_length is None:
            leg_length = height_cm * 0.45  # Average leg length is ~45% of height
        if torso_length is None:
            torso_length = height_cm * 0.30  # Average torso length is ~30% of height
            
        arm_span_ratio = arm_span / height_cm
        leg_length_ratio = leg_length / height_cm
        torso_length_ratio = torso_length / height_cm
        
        # Advanced features
        power_index = (weight_kg * bmi) / 1000
        endurance_index = height_cm / (weight_kg * bmi) * 1000
        reach_advantage = (arm_span_ratio - 1.0) * 100
        build_compactness = weight_kg / (height_cm * height_cm) * 10000
        
        # Create feature vector in same order as training
        features = np.array([
            height_cm, weight_kg, bmi, weight_height_ratio, height_weight_ratio,
            arm_span_ratio, leg_length_ratio, torso_length_ratio,
            power_index, endurance_index, reach_advantage, build_compactness
        ])
        
        # Apply same weighting as training
        weights = np.array([1.6, 1.1, 1.3, 1.2, 1.1, 1.4, 1.2, 1.3, 1.8, 1.7, 1.5, 1.6])
        weighted_features = features * weights
        
        return weighted_features
        
    def recommend_sport(self, gender: str, height_cm: float, weight_kg: float,
                       arm_span: Optional[float] = None, leg_length: Optional[float] = None,
                       torso_length: Optional[float] = None) -> Dict:
        """Get sport recommendation for user."""
        if not self.is_initialized:
            self.initialize()
            
        # Determine which model to use
        if gender.lower() in ['m', 'male']:
            model_key = 'male'
        elif gender.lower() in ['f', 'female']:
            model_key = 'female'
        else:
            model_key = 'combined'
            
        if model_key not in self.cluster_models:
            raise ValueError(f"No model available for gender: {gender}")
            
        # Get user features
        user_features = self.get_user_features(gender, height_cm, weight_kg, arm_span, leg_length, torso_length)
        
        # Scale features
        model = self.cluster_models[model_key]
        user_scaled = model['scaler'].transform(user_features.reshape(1, -1))
        
        # Predict cluster
        cluster_id = model['kmeans'].predict(user_scaled)[0]
        
        # Get top sports for this cluster
        cluster_info = self.cluster_stats[model_key][cluster_id]
        top_sports = cluster_info['top_sports']
        
        # Randomly select one of the top sports
        if top_sports:
            selected_sport = random.choice(list(top_sports.keys()))
        else:
            selected_sport = "Unknown"
            
        # Get sport information
        sport_info = get_sport_info(selected_sport)
        sport_stats = get_sport_stats(selected_sport)
        sport_description = get_sport_description(selected_sport)
        sport_name = get_sport_name(selected_sport)
        
        return {
            'recommended_sport': selected_sport,
            'sport_name': sport_name,
            'sport_description': sport_description,
            'sport_stats': sport_stats,
            'cluster_id': cluster_id,
            'cluster_size': cluster_info['size'],
            'all_top_sports': top_sports,
            'cluster_avg_height': cluster_info['avg_height'],
            'cluster_avg_weight': cluster_info['avg_weight'],
            'cluster_avg_bmi': cluster_info['avg_bmi']
        }
        
    def find_similar_athlete(self, gender: str, height_cm: float, weight_kg: float,
                           arm_span: Optional[float] = None, leg_length: Optional[float] = None,
                           torso_length: Optional[float] = None, 
                           preferred_sport: Optional[str] = None) -> Dict:
        """Find the most similar athlete to the user."""
        if not self.is_initialized:
            self.initialize()
            
        # Get sport recommendation first
        sport_rec = self.recommend_sport(gender, height_cm, weight_kg, arm_span, leg_length, torso_length)
        cluster_id = sport_rec['cluster_id']
        
        # Determine which model to use
        if gender.lower() in ['m', 'male']:
            model_key = 'male'
        elif gender.lower() in ['f', 'female']:
            model_key = 'female'
        else:
            model_key = 'combined'
            
        model = self.cluster_models[model_key]
        cluster_info = self.cluster_stats[model_key][cluster_id]
        
        # Get user features
        user_features = self.get_user_features(gender, height_cm, weight_kg, arm_span, leg_length, torso_length)
        user_scaled = model['scaler'].transform(user_features.reshape(1, -1))
        
        # Get athletes in the same cluster
        cluster_athletes = cluster_info['athletes']
        
        # Determine which sport to filter by
        target_sport = preferred_sport if preferred_sport else sport_rec['recommended_sport']
        
        # Filter athletes by the target sport (recommended sport or preferred sport)
        sport_athletes = [athlete for athlete in cluster_athletes if athlete['sport'] == target_sport]
        
        # If we found athletes in the target sport, use them
        if sport_athletes:
            cluster_athletes = sport_athletes
        else:
            # If no athletes found in target sport, try to find athletes in similar sports
            # or fall back to all athletes in cluster
            print(f"Warning: No athletes found in sport '{target_sport}' in cluster {cluster_id}")
            print(f"Available sports in cluster: {set(athlete['sport'] for athlete in cluster_athletes)}")
            # Keep all athletes in cluster as fallback
            
        # Find most similar athlete by calculating distances
        min_distance = float('inf')
        most_similar = None
        
        for athlete in cluster_athletes:
            # Get athlete features
            athlete_features = self.get_user_features(
                gender, athlete['height_cm'], athlete['weight_kg']
            )
            athlete_scaled = model['scaler'].transform(athlete_features.reshape(1, -1))
            
            # Calculate distance
            distance = euclidean_distances(user_scaled, athlete_scaled)[0][0]
            
            if distance < min_distance:
                min_distance = distance
                most_similar = athlete
                
        return {
            'similar_athlete': most_similar,
            'similarity_distance': min_distance,
            'recommended_sport': sport_rec['recommended_sport'],
            'cluster_id': cluster_id
        }
        
    def get_full_recommendation(self, gender: str, height_cm: float, weight_kg: float,
                              arm_span: Optional[float] = None, leg_length: Optional[float] = None,
                              torso_length: Optional[float] = None) -> Dict:
        """Get complete recommendation including sport and similar athlete."""
        sport_rec = self.recommend_sport(gender, height_cm, weight_kg, arm_span, leg_length, torso_length)
        similar_athlete = self.find_similar_athlete(gender, height_cm, weight_kg, arm_span, leg_length, torso_length)
        
        return {
            'sport_recommendation': sport_rec,
            'similar_athlete': similar_athlete,
            'user_measurements': {
                'gender': gender,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'arm_span': arm_span,
                'leg_length': leg_length,
                'torso_length': torso_length
            }
        }
    
    def find_similar_athletes(self, gender: str, height_cm: float, weight_kg: float,
                            arm_span: Optional[float] = None, leg_length: Optional[float] = None,
                            torso_length: Optional[float] = None, 
                            top_sports: Optional[list] = None, num_athletes: int = 3) -> list:
        """Find multiple similar athletes from the top sports."""
        try:
            # Get sport recommendation first
            sport_rec = self.recommend_sport(gender, height_cm, weight_kg, arm_span, leg_length, torso_length)
            
            # Get user features
            user_features = self.get_user_features(gender, height_cm, weight_kg)
            
            # Get the appropriate model
            if gender.lower() in ['m', 'male']:
                model_key = 'male'
            elif gender.lower() in ['f', 'female']:
                model_key = 'female'
            else:
                model_key = 'combined'
            
            model = self.cluster_models[model_key]
            user_scaled = model['scaler'].transform(user_features.reshape(1, -1))
            
            # Get cluster assignment
            cluster_id = model['kmeans'].predict(user_scaled)[0]
            
            # Get cluster info and athletes
            cluster_info = self.cluster_stats[model_key][cluster_id]
            cluster_athletes = cluster_info['athletes']
            
            # Filter athletes by top sports if provided
            if top_sports:
                # Convert sport names to sport keys for comparison
                sport_keys = []
                for sport_name in top_sports:
                    for key, info in SPORT_DATABASE.items():
                        if info['name'] == sport_name:
                            sport_keys.append(key)
                            break
                
                # Filter athletes by these sports
                filtered_athletes = []
                for athlete in cluster_athletes:
                    if athlete['sport'] in sport_keys:
                        filtered_athletes.append(athlete)
                
                if filtered_athletes:
                    cluster_athletes = filtered_athletes
            
            # Calculate distances for all athletes
            athlete_distances = []
            for athlete in cluster_athletes:
                athlete_features = self.get_user_features(
                    gender, athlete['height_cm'], athlete['weight_kg']
                )
                athlete_scaled = model['scaler'].transform(athlete_features.reshape(1, -1))
                distance = euclidean_distances(user_scaled, athlete_scaled)[0][0]
                athlete_distances.append((athlete, distance))
            
            # Sort by distance and get top athletes
            athlete_distances.sort(key=lambda x: x[1])
            top_athletes = athlete_distances[:num_athletes]
            
            # Format results
            similar_athletes = []
            for athlete, distance in top_athletes:
                similarity_score = max(0, 100 - (distance * 10))  # Convert distance to percentage
                
                similar_athletes.append({
                    'athlete': athlete,
                    'similarity_score': similarity_score,
                    'distance': distance
                })
            
            return {
                'similar_athletes': similar_athletes,
                'cluster_id': cluster_id,
                'total_athletes_considered': len(cluster_athletes)
            }
            
        except Exception as e:
            print(f"Error finding similar athletes: {e}")
            return {'similar_athletes': [], 'error': str(e)}

def main():
    """Test the recommendation engine."""
    print("🎯 SPORT RECOMMENDATION ENGINE TEST")
    print("=" * 50)
    
    # Initialize engine
    engine = SportRecommendationEngine()
    engine.initialize()
    
    # Test cases
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
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 30)
        
        try:
            recommendation = engine.get_full_recommendation(
                gender=test_case['gender'],
                height_cm=test_case['height_cm'],
                weight_kg=test_case['weight_kg'],
                arm_span=test_case.get('arm_span')
            )
            
            sport_rec = recommendation['sport_recommendation']
            similar_athlete = recommendation['similar_athlete']
            
            print(f"🏆 Recommended Sport: {sport_rec['recommended_sport']}")
            print(f"📊 Cluster: {sport_rec['cluster_id']} ({sport_rec['cluster_size']} athletes)")
            print(f"📈 All top sports: {sport_rec['all_top_sports']}")
            
            if similar_athlete['similar_athlete']:
                athlete = similar_athlete['similar_athlete']
                print(f"👤 Similar Athlete: {athlete['Player']} ({athlete['sport']})")
                print(f"   Height: {athlete['height_cm']:.1f}cm, Weight: {athlete['weight_kg']:.1f}kg")
                print(f"   Similarity Distance: {similar_athlete['similarity_distance']:.3f}")
            else:
                print("👤 No similar athlete found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Recommendation engine test complete!")

if __name__ == "__main__":
    main()
