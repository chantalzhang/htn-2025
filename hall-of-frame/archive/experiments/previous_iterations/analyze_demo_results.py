#!/usr/bin/env python3
"""
Analyze demo-optimized clustering results against ideal archetypes.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_and_preprocess_data():
    """Load and preprocess athlete data with enhanced feature engineering."""
    df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    
    # Enhanced feature engineering for better body-type separation
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
    df['height_weight_ratio'] = df['height_cm'] / df['weight_kg']
    df['arm_span_ratio'] = df['Arm Span'] / df['height_cm']
    df['leg_length_ratio'] = df['Leg Length'] / df['height_cm']
    df['torso_length_ratio'] = df['Torso Length'] / df['height_cm']
    
    # Advanced body-type features
    df['power_index'] = (df['weight_kg'] * df['bmi']) / 1000
    df['endurance_index'] = df['height_cm'] / (df['weight_kg'] * df['bmi']) * 1000
    df['reach_advantage'] = (df['arm_span_ratio'] - 1.0) * 100
    df['build_compactness'] = df['weight_kg'] / (df['height_cm'] * df['height_cm']) * 10000
    
    # Handle missing values
    for col in ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio',
                'power_index', 'endurance_index', 'reach_advantage', 'build_compactness']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    return df

def apply_sophisticated_weighting(df):
    """Apply sophisticated weighting."""
    core_features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                    'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
    advanced_features = ['power_index', 'endurance_index', 'reach_advantage', 'build_compactness']
    
    sophisticated_weights = {
        'height_cm': 1.8,
        'power_index': 2.2,
        'endurance_index': 2.0,
        'reach_advantage': 1.9,
        'arm_span_ratio': 1.7,
        'torso_length_ratio': 1.6,
        'build_compactness': 1.8,
        'bmi': 1.5,
        'weight_height_ratio': 1.4,
        'leg_length_ratio': 1.3,
        'height_weight_ratio': 1.2,
        'weight_kg': 1.1,
    }
    
    df_weighted = df.copy()
    all_features = core_features + advanced_features
    
    for feature in all_features:
        if feature in df_weighted.columns:
            weight = sophisticated_weights.get(feature, 1.0)
            df_weighted[f'weighted_{feature}'] = df_weighted[feature] * weight
    
    return df_weighted, all_features

def determine_archetype(height, weight, bmi, power, endurance, reach, compactness):
    """Determine body-type archetype."""
    if height > 185 and reach > 5 and endurance > 0.8:
        return "Tall/Lean/Long Arms"
    elif height < 170 and compactness > 0.4 and power > 0.3:
        return "Compact/Short/Powerful"
    elif reach > 3 and endurance > 0.6:
        return "Tall Torso/Long Arms"
    elif endurance > 1.0 and bmi < 22:
        return "Light/Lean Endurance"
    elif power > 0.4 and bmi > 25:
        return "Heavy/Powerful"
    else:
        return "Medium Build/Agile"

def analyze_demo_results(df, cluster_labels, group_name):
    """Analyze demo results against ideal archetypes."""
    print(f"\n🎯 {group_name.upper()} - DEMO RESULTS ANALYSIS")
    print("=" * 60)
    
    # Cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    total_athletes = len(cluster_labels)
    
    print(f"Total athletes: {total_athletes}")
    print(f"Number of clusters: {len(cluster_sizes)}")
    
    # Check cluster balance (ideal: 10-25% each)
    print(f"\n📊 Cluster Balance Analysis:")
    balanced_clusters = 0
    for cluster_id, size in cluster_sizes.items():
        percentage = size / total_athletes * 100
        status = "✅ BALANCED" if 10 <= percentage <= 25 else "⚠️ IMBALANCED"
        if 10 <= percentage <= 25:
            balanced_clusters += 1
        print(f"  Cluster {cluster_id}: {size} athletes ({percentage:.1f}%) {status}")
    
    print(f"\nBalanced clusters: {balanced_clusters}/{len(cluster_sizes)} ({balanced_clusters/len(cluster_sizes)*100:.1f}%)")
    
    # Analyze archetypes and sports
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    print(f"\n🏆 Archetype Analysis:")
    ideal_archetypes = [
        "Tall/Lean/Long Arms → basketball + volleyball",
        "Compact/Short/Powerful → gymnastics, wrestling", 
        "Tall Torso/Long Arms → rowing + swimming",
        "Light/Lean Endurance → distance running, soccer",
        "Heavy/Powerful → weightlifting, wrestling/judo",
        "Medium Build/Agile → tennis, sprint running"
    ]
    
    archetype_matches = 0
    sport_redundancy = {}
    
    for cluster_id in sorted(cluster_sizes.index):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Calculate archetype
        archetype = determine_archetype(
            cluster_data['height_cm'].mean(),
            cluster_data['weight_kg'].mean(),
            cluster_data['bmi'].mean(),
            cluster_data['power_index'].mean(),
            cluster_data['endurance_index'].mean(),
            cluster_data['reach_advantage'].mean(),
            cluster_data['build_compactness'].mean()
        )
        
        # Top sports
        sport_counts = cluster_data['sport'].value_counts()
        top_3_sports = sport_counts.head(3)
        
        print(f"\n  Cluster {cluster_id} ({len(cluster_data)} athletes):")
        print(f"    Archetype: {archetype}")
        print(f"    Top sports: {', '.join([f'{sport} ({count})' for sport, count in top_3_sports.items()])}")
        
        # Check if archetype matches ideal
        for ideal in ideal_archetypes:
            if any(word in archetype.lower() for word in ideal.lower().split()[:2]):
                archetype_matches += 1
                break
        
        # Track sport redundancy
        for sport in top_3_sports.index:
            sport_redundancy[sport] = sport_redundancy.get(sport, 0) + 1
    
    print(f"\n✅ Archetype matches: {archetype_matches}/{len(cluster_sizes)} ({archetype_matches/len(cluster_sizes)*100:.1f}%)")
    
    # Check sport redundancy (ideal: no sport in 3+ clusters)
    print(f"\n📈 Sport Redundancy Check:")
    high_redundancy = 0
    for sport, count in sport_redundancy.items():
        status = "⚠️ HIGH" if count >= 3 else "✅ OK"
        if count >= 3:
            high_redundancy += 1
        print(f"  {sport}: appears in {count} clusters {status}")
    
    print(f"\nSports with high redundancy: {high_redundancy}/{len(sport_redundancy)} ({high_redundancy/len(sport_redundancy)*100:.1f}%)")
    
    # Overall score
    balance_score = balanced_clusters / len(cluster_sizes) * 100
    archetype_score = archetype_matches / len(cluster_sizes) * 100
    redundancy_score = (len(sport_redundancy) - high_redundancy) / len(sport_redundancy) * 100
    
    overall_score = (balance_score + archetype_score + redundancy_score) / 3
    
    print(f"\n🏆 OVERALL DEMO QUALITY SCORE: {overall_score:.1f}%")
    print(f"  Balance: {balance_score:.1f}%")
    print(f"  Archetypes: {archetype_score:.1f}%") 
    print(f"  Redundancy: {redundancy_score:.1f}%")

def main():
    """Main analysis function."""
    print("🎯 DEMO-OPTIMIZED CLUSTERING ANALYSIS")
    print("=" * 60)
    print("Analyzing results against ideal archetype criteria")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    df_weighted, feature_cols = apply_sophisticated_weighting(df)
    
    # Prepare weighted features
    weighted_feature_cols = [f'weighted_{col}' for col in feature_cols]
    X = df_weighted[weighted_feature_cols].copy()
    X = X.fillna(0)
    
    # Analyze different groups
    groups = [
        ('Combined', df_weighted, 6),
        ('Male', df_weighted[df_weighted['Sex'] == 'M'], 5),
        ('Female', df_weighted[df_weighted['Sex'] == 'F'], 4)
    ]
    
    for group_name, group_df, k in groups:
        if len(group_df) < k:
            print(f"\n⚠️  Skipping {group_name}: Not enough data ({len(group_df)} < {k})")
            continue
            
        # Perform clustering (same as demo script)
        X_group = X.loc[group_df.index]
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_group)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Analyze results
        analyze_demo_results(group_df, cluster_labels, group_name)

if __name__ == "__main__":
    main()
