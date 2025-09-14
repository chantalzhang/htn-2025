"""
Final Two PNG Results Generator
==============================

Creates exactly what was requested:
1. One PNG with three clustering visualizations side by side
2. One PNG with text results in three side-by-side text boxes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess the athlete dataset."""
    print("📊 Loading athlete dataset...")
    
    # Load the dataset
    df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    print(f"✅ Loaded {len(df)} athletes")
    
    # Handle missing values conservatively
    df_processed = df.copy()
    
    # For columns with >50% missing, use mean imputation
    for col in ['Hand Length', 'Hand Width', 'Arm Span', 'Leg Length', 'Torso Length', 'Spike Reach', 'Block Reach']:
        if col in df_processed.columns:
            missing_pct = df_processed[col].isnull().sum() / len(df_processed) * 100
            if missing_pct > 50:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # Create feature ratios
    df_processed['bmi'] = df_processed['weight_kg'] / (df_processed['height_cm'] / 100) ** 2
    df_processed['weight_height_ratio'] = df_processed['weight_kg'] / df_processed['height_cm']
    df_processed['height_weight_ratio'] = df_processed['height_cm'] / df_processed['weight_kg']
    df_processed['arm_span_ratio'] = df_processed['Arm Span'] / df_processed['height_cm']
    df_processed['leg_length_ratio'] = df_processed['Leg Length'] / df_processed['height_cm']
    df_processed['torso_length_ratio'] = df_processed['Torso Length'] / df_processed['height_cm']
    
    # Select features for clustering (will be weighted)
    feature_cols = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                   'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
    
    return df_processed, feature_cols

def apply_body_type_weighting(df, feature_cols):
    """
    Apply strategic body type weighting to create balanced, distinct clusters.
    Designed to separate athletes into clear body-type archetypes:
    1. Tall/lean/long arms (basketball, volleyball)
    2. Compact/short/powerful (gymnastics, wrestling)  
    3. Tall torso/long arms (rowing, swimming)
    4. Light/lean endurance (distance running, soccer)
    5. Heavy/powerful (weightlifting, wrestling/judo)
    6. Medium build/agile (tennis, sprint running)
    """
    print("   Applying strategic body type weighting for balanced clusters...")

    # Strategic body type weights to create distinct archetypes
    body_type_weights = {
        # Height differentiation (separates tall vs short sports)
        'height_cm': 2.5,           # Strong emphasis to separate tall sports
        
        # Power vs lean differentiation  
        'weight_kg': 2.0,           # Separates heavy vs light sports
        'bmi': 2.8,                 # Strong emphasis on bulk vs lean
        'weight_height_ratio': 2.5, # Separates stocky vs lean builds
        
        # Limb proportions (separates different body types)
        'arm_span_ratio': 2.2,      # Separates long-armed vs short-armed sports
        'leg_length_ratio': 1.8,    # Separates long-legged vs short-legged
        'torso_length_ratio': 2.0,  # Separates long-torso vs short-torso
        
        # Height-weight balance
        'height_weight_ratio': 1.5, # Additional separation of build types
    }

    # Copy the entire dataframe to preserve all columns
    df_weighted = df.copy()

    # Apply weights (default 1.0 if not defined)
    for feature in feature_cols:
        if feature in df_weighted.columns:
            weight = body_type_weights.get(feature, 1.0)
            df_weighted[f'weighted_{feature}'] = df_weighted[feature] * weight

    return df_weighted

def perform_clustering(df, feature_cols, k):
    """Perform K-means clustering."""
    # Prepare data using weighted features
    weighted_feature_cols = [f'weighted_{col}' for col in feature_cols]
    X = df[weighted_feature_cols].copy()
    
    # Convert to numeric and handle missing values
    for col in weighted_feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.mean())
    if X.isnull().any().any():
        X = X.fillna(0)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, cluster_labels)
    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
    
    return cluster_labels, X_scaled, silhouette, calinski

def analyze_clusters(df, cluster_labels, feature_cols):
    """Analyze cluster characteristics."""
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster_id in sorted(df_analysis['cluster'].unique()):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        # Basic stats
        size = len(cluster_data)
        percentage = size / len(df_analysis) * 100
        
        # Average measurements
        avg_height = cluster_data['height_cm'].mean()
        avg_weight = cluster_data['weight_kg'].mean()
        avg_bmi = cluster_data['bmi'].mean()
        
        # Top sports
        sport_counts = cluster_data['sport'].value_counts()
        top_sports = sport_counts.head(3)
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': size,
            'percentage': percentage,
            'avg_height': avg_height,
            'avg_weight': avg_weight,
            'avg_bmi': avg_bmi,
            'top_sports': top_sports
        })
    
    return cluster_stats

def create_three_visualizations_png():
    """Create one PNG with three clustering visualizations side by side."""
    print("\n🎨 Creating three clustering visualizations in one PNG...")
    
    # Load and preprocess data
    df, feature_cols = load_and_preprocess_data()
    df_weighted = apply_body_type_weighting(df, feature_cols)
    
    # Store results for all three analyses
    all_results = {}
    
    # 1. Combined Analysis
    print("\n🏆 ANALYZING ALL ATHLETES...")
    cluster_labels_combined, X_scaled_combined, silhouette_combined, calinski_combined = perform_clustering(df_weighted, feature_cols, 6)
    cluster_stats_combined = analyze_clusters(df_weighted, cluster_labels_combined, feature_cols)
    
    all_results['combined'] = {
        'df': df_weighted,
        'cluster_labels': cluster_labels_combined,
        'X_scaled': X_scaled_combined,
        'cluster_stats': cluster_stats_combined,
        'k': 5,
        'silhouette': silhouette_combined
    }
    
    # 2. Male Analysis
    print("🏃 ANALYZING MALE ATHLETES...")
    df_male = df_weighted[df_weighted['Sex'] == 'M'].copy()
    cluster_labels_male, X_scaled_male, silhouette_male, calinski_male = perform_clustering(df_male, feature_cols, 5)
    cluster_stats_male = analyze_clusters(df_male, cluster_labels_male, feature_cols)
    
    all_results['male'] = {
        'df': df_male,
        'cluster_labels': cluster_labels_male,
        'X_scaled': X_scaled_male,
        'cluster_stats': cluster_stats_male,
        'k': 5,
        'silhouette': silhouette_male
    }
    
    # 3. Female Analysis
    print("🏃 ANALYZING FEMALE ATHLETES...")
    df_female = df_weighted[df_weighted['Sex'] == 'F'].copy()
    cluster_labels_female, X_scaled_female, silhouette_female, calinski_female = perform_clustering(df_female, feature_cols, 4)
    cluster_stats_female = analyze_clusters(df_female, cluster_labels_female, feature_cols)
    
    all_results['female'] = {
        'df': df_female,
        'cluster_labels': cluster_labels_female,
        'X_scaled': X_scaled_female,
        'cluster_stats': cluster_stats_female,
        'k': 5,
        'silhouette': silhouette_female
    }
    
    # Create combined visualization with all three PCA plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    titles = ['All Athletes', 'Male Athletes', 'Female Athletes']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (analysis_type, ax, title, color) in enumerate(zip(['combined', 'male', 'female'], [ax1, ax2, ax3], titles, colors)):
        result = all_results[analysis_type]
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(result['X_scaled'])
        
        # Plot clusters
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=result['cluster_labels'], 
                           cmap='tab10', alpha=0.7, s=50)
        
        # Formatting
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(f'{title}\n({result["k"]} clusters, silhouette={result["silhouette"]:.3f})', 
                    fontweight='bold', color=color, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)
    
    plt.suptitle('Athlete Body Type Clustering - PCA Visualizations', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig('clustering_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'clustering_visualizations.png'")
    
    return all_results

def create_three_text_boxes_png(all_results):
    """Create one PNG with three side-by-side text boxes showing results."""
    print("\n📝 Creating three text boxes with results in one PNG...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 16))
    
    titles = ['Combined Analysis', 'Male Analysis', 'Female Analysis']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (analysis_type, ax, title, color) in enumerate(zip(['combined', 'male', 'female'], [ax1, ax2, ax3], titles, colors)):
        result = all_results[analysis_type]
        cluster_stats = result['cluster_stats']
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create text content
        text_content = f"🏆 {title}\n"
        text_content += "=" * 60 + "\n\n"
        text_content += f"📊 Clusters: {result['k']}\n"
        text_content += f"📈 Athletes: {sum(stat['size'] for stat in cluster_stats)}\n"
        text_content += f"🎯 Silhouette: {result['silhouette']:.3f}\n\n"
        
        # Add cluster summaries
        for stat in cluster_stats:
            text_content += f"🔹 CLUSTER {stat['cluster_id']}\n"
            text_content += f"   Athletes: {stat['size']} ({stat['percentage']:.1f}%)\n"
            text_content += f"   Height: {stat['avg_height']:.1f}cm\n"
            text_content += f"   Weight: {stat['avg_weight']:.1f}kg\n"
            text_content += f"   BMI: {stat['avg_bmi']:.1f}\n"
            text_content += f"   Top Sports: {', '.join(stat['top_sports'].head(2).index)}\n\n"
        
        # Add text box with large, clear font
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=color, alpha=0.15, 
                         edgecolor=color, linewidth=2))
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)
    
    plt.suptitle('Athlete Body Type Clustering Results', fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved as 'clustering_results.png'")

def main():
    """Main execution function."""
    print("🚀 FINAL TWO PNG RESULTS GENERATOR")
    print("=" * 50)
    print("Creating exactly what was requested:")
    print("1. One PNG with three clustering visualizations side by side")
    print("2. One PNG with text results in three side-by-side text boxes")
    
    # Create the three visualizations in one PNG
    all_results = create_three_visualizations_png()
    
    # Create the three text boxes in one PNG
    create_three_text_boxes_png(all_results)
    
    print("\n✅ Complete! Two PNG files generated:")
    print("📁 Files created:")
    print("   • clustering_visualizations.png - Three PCA plots side by side")
    print("   • clustering_results.png - Three text boxes side by side")

if __name__ == "__main__":
    main()
