#!/usr/bin/env python3
"""
Quick analysis to verify cluster balance and sport diversity.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def load_and_preprocess_data():
    """Load and preprocess the athlete data."""
    df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    
    # Create ratio features
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
    df['height_weight_ratio'] = df['height_cm'] / df['weight_kg']
    df['arm_span_ratio'] = df['Arm Span'] / df['height_cm']
    df['leg_length_ratio'] = df['Leg Length'] / df['height_cm']
    df['torso_length_ratio'] = df['Torso Length'] / df['height_cm']
    
    # Handle missing values
    for col in ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    return df

def apply_strategic_weighting(df):
    """Apply the strategic body type weighting."""
    feature_cols = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                   'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
    
    body_type_weights = {
        'height_cm': 2.5,
        'weight_kg': 2.0,
        'bmi': 2.8,
        'weight_height_ratio': 2.5,
        'arm_span_ratio': 2.2,
        'leg_length_ratio': 1.8,
        'torso_length_ratio': 2.0,
        'height_weight_ratio': 1.5,
    }
    
    df_weighted = df.copy()
    for feature in feature_cols:
        if feature in df_weighted.columns:
            weight = body_type_weights.get(feature, 1.0)
            df_weighted[f'weighted_{feature}'] = df_weighted[feature] * weight
    
    return df_weighted, feature_cols

def analyze_cluster_balance(df, cluster_labels, group_name):
    """Analyze cluster balance and sport diversity."""
    print(f"\n📊 {group_name.upper()} ANALYSIS:")
    print("=" * 50)
    
    # Cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    total_athletes = len(cluster_labels)
    
    print(f"Total athletes: {total_athletes}")
    print(f"Number of clusters: {len(cluster_sizes)}")
    print("\nCluster sizes:")
    for cluster_id, size in cluster_sizes.items():
        percentage = size / total_athletes * 100
        print(f"  Cluster {cluster_id}: {size} athletes ({percentage:.1f}%)")
    
    # Sport distribution per cluster
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    print(f"\nSport distribution per cluster:")
    sport_cluster_matrix = pd.crosstab(df_with_clusters['sport'], df_with_clusters['cluster'])
    
    # Show top 3 sports per cluster
    for cluster_id in sorted(cluster_sizes.index):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        sport_counts = cluster_data['sport'].value_counts()
        top_3_sports = sport_counts.head(3)
        
        print(f"\n  Cluster {cluster_id} ({len(cluster_data)} athletes):")
        for sport, count in top_3_sports.items():
            percentage = count / len(cluster_data) * 100
            print(f"    {sport}: {count} ({percentage:.1f}%)")
    
    # Check sport diversity
    all_sports = set(df['sport'].unique())
    sports_in_clusters = set()
    for cluster_id in sorted(cluster_sizes.index):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        sports_in_clusters.update(cluster_data['sport'].unique())
    
    missing_sports = all_sports - sports_in_clusters
    print(f"\nSport diversity check:")
    print(f"  Total sports in dataset: {len(all_sports)}")
    print(f"  Sports appearing in clusters: {len(sports_in_clusters)}")
    if missing_sports:
        print(f"  Missing sports: {missing_sports}")
    else:
        print(f"  ✅ All sports appear in clusters!")

def main():
    """Main analysis function."""
    print("🔍 CLUSTER BALANCE ANALYSIS")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    df_weighted, feature_cols = apply_strategic_weighting(df)
    
    # Prepare weighted features
    weighted_feature_cols = [f'weighted_{col}' for col in feature_cols]
    X = df_weighted[weighted_feature_cols].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
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
            
        # Perform clustering
        X_group = X_scaled[group_df.index]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_group)
        
        # Analyze results
        analyze_cluster_balance(group_df, cluster_labels, group_name)

if __name__ == "__main__":
    main()
