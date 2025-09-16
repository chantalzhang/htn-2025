#!/usr/bin/env python3
"""
Final Demo Optimization - Addresses cluster balance for ideal demo results.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def apply_balanced_weighting(df):
    """Apply weighting optimized for balanced clusters."""
    print("   Applying balanced clustering optimization...")
    
    core_features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                    'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
    advanced_features = ['power_index', 'endurance_index', 'reach_advantage', 'build_compactness']
    
    # Balanced weighting that prevents mega-clusters
    balanced_weights = {
        # Moderate emphasis on key differentiators
        'height_cm': 1.6,           # Moderate height emphasis
        'power_index': 1.8,         # Moderate power emphasis
        'endurance_index': 1.7,     # Moderate endurance emphasis
        'reach_advantage': 1.5,     # Moderate reach emphasis
        'build_compactness': 1.6,   # Moderate compactness emphasis
        
        # Standard emphasis on other features
        'arm_span_ratio': 1.4,
        'torso_length_ratio': 1.3,
        'bmi': 1.3,
        'weight_height_ratio': 1.2,
        'leg_length_ratio': 1.2,
        'height_weight_ratio': 1.1,
        'weight_kg': 1.1,
    }
    
    df_weighted = df.copy()
    all_features = core_features + advanced_features
    
    for feature in all_features:
        if feature in df_weighted.columns:
            weight = balanced_weights.get(feature, 1.0)
            df_weighted[f'weighted_{feature}'] = df_weighted[feature] * weight
    
    return df_weighted, all_features

def perform_balanced_clustering(df, feature_cols, k, group_name):
    """Perform clustering optimized for balance."""
    print(f"   Optimizing for balanced clusters in {group_name}...")
    
    # Prepare weighted features
    weighted_feature_cols = [f'weighted_{col}' for col in feature_cols]
    X = df[weighted_feature_cols].copy()
    X = X.fillna(0)
    
    # Use StandardScaler for more balanced clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try multiple approaches for better balance
    best_balance_score = float('inf')
    best_cluster_labels = None
    best_silhouette = -1
    
    # Try different random states and k values
    k_values = [k-1, k, k+1] if k > 3 else [k, k+1]
    
    for test_k in k_values:
        if test_k < 2 or test_k > len(df) // 5:  # Reasonable bounds
            continue
            
        for random_state in [42, 123, 456, 789, 999]:
            kmeans = KMeans(n_clusters=test_k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate balance score (lower is better)
            cluster_sizes = pd.Series(cluster_labels).value_counts()
            balance_score = cluster_sizes.std() / cluster_sizes.mean()  # Coefficient of variation
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(X_scaled, cluster_labels)
                
                # Combined score: prioritize balance but maintain quality
                combined_score = balance_score - (silhouette * 0.1)
                
                if combined_score < best_balance_score:
                    best_balance_score = combined_score
                    best_cluster_labels = cluster_labels
                    best_silhouette = silhouette
    
    print(f"   Best balance score: {best_balance_score:.3f}, silhouette: {best_silhouette:.3f}")
    return best_cluster_labels, X_scaled, best_silhouette

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

def create_final_visualization(all_results):
    """Create final optimized visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Final Demo-Optimized Athlete Body-Type Clustering', fontsize=16, fontweight='bold')
    
    groups = ['combined', 'male', 'female']
    titles = ['Combined Athletes', 'Male Athletes', 'Female Athletes']
    
    for idx, (group, title) in enumerate(zip(groups, titles)):
        if group not in all_results:
            continue
            
        result = all_results[group]
        df = result['df']
        cluster_labels = result['cluster_labels']
        X_scaled = result['X_scaled']
        
        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot
        scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                  cmap='tab10', alpha=0.7, s=50)
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Add cluster size annotations
        for cluster_id in sorted(set(cluster_labels)):
            cluster_mask = cluster_labels == cluster_id
            cluster_center = X_pca[cluster_mask].mean(axis=0)
            cluster_size = np.sum(cluster_mask)
            percentage = cluster_size / len(cluster_labels) * 100
            
            axes[idx].annotate(f'C{cluster_id}: {cluster_size} ({percentage:.1f}%)', 
                             xy=cluster_center, xytext=(5, 5), 
                             textcoords='offset points', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('final_demo_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Final visualization saved as 'final_demo_visualizations.png'")

def create_final_results_display(all_results):
    """Create final results display."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 12))
    fig.suptitle('Final Demo-Optimized Clustering Results', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    groups = ['combined', 'male', 'female']
    titles = ['Combined Analysis', 'Male Analysis', 'Female Analysis']
    
    for idx, (group, title) in enumerate(zip(groups, titles)):
        if group not in all_results:
            continue
            
        result = all_results[group]
        cluster_stats = result['cluster_stats']
        
        # Create text display
        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        y_pos = 0.95
        for stat in cluster_stats:
            # Cluster header with balance indicator
            balance_status = "✅" if 10 <= stat['percentage'] <= 25 else "⚠️"
            axes[idx].text(0.05, y_pos, f"{balance_status} Cluster {stat['cluster_id']} ({stat['size']} athletes, {stat['percentage']:.1f}%)", 
                          fontsize=11, fontweight='bold', color='darkblue')
            y_pos -= 0.08
            
            # Archetype
            axes[idx].text(0.05, y_pos, f"Archetype: {stat['archetype']}", 
                          fontsize=10, fontweight='bold', color='darkgreen')
            y_pos -= 0.06
            
            # Measurements
            axes[idx].text(0.05, y_pos, f"Avg: {stat['avg_height']:.1f}cm, {stat['avg_weight']:.1f}kg, BMI {stat['avg_bmi']:.1f}", 
                          fontsize=9, color='black')
            y_pos -= 0.05
            
            # Top sports
            sport_text = "Top sports: "
            for sport, count in stat['top_sports'].items():
                sport_text += f"{sport} ({count}), "
            sport_text = sport_text.rstrip(", ")
            axes[idx].text(0.05, y_pos, sport_text, fontsize=9, color='darkred')
            y_pos -= 0.08
            
            y_pos -= 0.02  # Extra spacing between clusters
    
    plt.tight_layout()
    plt.savefig('final_demo_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Final results saved as 'final_demo_results.png'")

def analyze_clusters_for_archetypes(df, cluster_labels, feature_cols):
    """Analyze clusters to identify body-type archetypes."""
    cluster_stats = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_data = df[cluster_labels == cluster_id]
        size = len(cluster_data)
        percentage = size / len(df) * 100
        
        # Calculate archetype indicators
        avg_height = cluster_data['height_cm'].mean()
        avg_weight = cluster_data['weight_kg'].mean()
        avg_bmi = cluster_data['bmi'].mean()
        avg_power = cluster_data['power_index'].mean()
        avg_endurance = cluster_data['endurance_index'].mean()
        avg_reach = cluster_data['reach_advantage'].mean()
        avg_compactness = cluster_data['build_compactness'].mean()
        
        # Determine archetype
        archetype = determine_archetype(avg_height, avg_weight, avg_bmi, avg_power, 
                                      avg_endurance, avg_reach, avg_compactness)
        
        # Top sports
        sport_counts = cluster_data['sport'].value_counts()
        top_sports = sport_counts.head(3)
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': size,
            'percentage': percentage,
            'archetype': archetype,
            'avg_height': avg_height,
            'avg_weight': avg_weight,
            'avg_bmi': avg_bmi,
            'avg_power': avg_power,
            'avg_endurance': avg_endurance,
            'avg_reach': avg_reach,
            'avg_compactness': avg_compactness,
            'top_sports': top_sports
        })
    
    return cluster_stats

def main():
    """Main execution function."""
    print("🎯 FINAL DEMO OPTIMIZATION")
    print("=" * 50)
    print("Optimizing for balanced clusters and ideal archetypes")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    print(f"✅ Loaded {len(df)} athletes")
    
    # Apply balanced weighting
    df_weighted, feature_cols = apply_balanced_weighting(df)
    
    # Store results for all three analyses
    all_results = {}
    
    # 1. Combined Analysis
    print("\n🏆 ANALYZING ALL ATHLETES...")
    cluster_labels_combined, X_scaled_combined, silhouette_combined = perform_balanced_clustering(
        df_weighted, feature_cols, 6, "Combined")
    cluster_stats_combined = analyze_clusters_for_archetypes(df_weighted, cluster_labels_combined, feature_cols)
    
    all_results['combined'] = {
        'df': df_weighted,
        'cluster_labels': cluster_labels_combined,
        'X_scaled': X_scaled_combined,
        'cluster_stats': cluster_stats_combined,
        'silhouette': silhouette_combined
    }
    
    # 2. Male Analysis
    print("\n🏃 ANALYZING MALE ATHLETES...")
    df_male = df_weighted[df_weighted['Sex'] == 'M'].copy()
    cluster_labels_male, X_scaled_male, silhouette_male = perform_balanced_clustering(
        df_male, feature_cols, 5, "Male")
    cluster_stats_male = analyze_clusters_for_archetypes(df_male, cluster_labels_male, feature_cols)
    
    all_results['male'] = {
        'df': df_male,
        'cluster_labels': cluster_labels_male,
        'X_scaled': X_scaled_male,
        'cluster_stats': cluster_stats_male,
        'silhouette': silhouette_male
    }
    
    # 3. Female Analysis
    print("\n🏃 ANALYZING FEMALE ATHLETES...")
    df_female = df_weighted[df_weighted['Sex'] == 'F'].copy()
    cluster_labels_female, X_scaled_female, silhouette_female = perform_balanced_clustering(
        df_female, feature_cols, 4, "Female")
    cluster_stats_female = analyze_clusters_for_archetypes(df_female, cluster_labels_female, feature_cols)
    
    all_results['female'] = {
        'df': df_female,
        'cluster_labels': cluster_labels_female,
        'X_scaled': X_scaled_female,
        'cluster_stats': cluster_stats_female,
        'silhouette': silhouette_female
    }
    
    # Create final visualizations
    print("\n🎨 Creating final optimized visualizations...")
    create_final_visualization(all_results)
    create_final_results_display(all_results)
    
    print("\n✅ Final demo optimization complete!")
    print("📁 Files created:")
    print("   • final_demo_visualizations.png - Final optimized PCA plots")
    print("   • final_demo_results.png - Final optimized results with balance indicators")

if __name__ == "__main__":
    main()
