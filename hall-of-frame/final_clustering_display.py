import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    print(f"   Sports: {sorted(df['sport'].unique())}")
    print(f"   Gender distribution: {df['Sex'].value_counts().to_dict()}")
    
    # Handle missing values conservatively
    print("\n🔄 Preprocessing data...")
    print("   Handling missing values conservatively...")
    
    # Keep original data intact, only impute where absolutely necessary
    df_processed = df.copy()
    
    # For columns with >50% missing, use mean imputation
    for col in ['Hand Length', 'Hand Width', 'Arm Span', 'Leg Length', 'Torso Length', 'Spike Reach', 'Block Reach']:
        if col in df_processed.columns:
            missing_pct = df_processed[col].isnull().sum() / len(df_processed) * 100
            if missing_pct > 50:
                print(f"   Warning: {col} has {missing_pct:.1f}% missing values - using mean imputation")
                # Convert to numeric first, then fill missing values
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            else:
                print(f"   Note: {col} has {missing_pct:.1f}% missing values - preserving original data")
    
    # Create feature ratios
    df_processed['bmi'] = df_processed['weight_kg'] / (df_processed['height_cm'] / 100) ** 2
    df_processed['weight_height_ratio'] = df_processed['weight_kg'] / df_processed['height_cm']
    df_processed['height_weight_ratio'] = df_processed['height_cm'] / df_processed['weight_kg']
    df_processed['arm_span_ratio'] = df_processed['Arm Span'] / df_processed['height_cm']
    df_processed['leg_length_ratio'] = df_processed['Leg Length'] / df_processed['height_cm']
    df_processed['torso_length_ratio'] = df_processed['Torso Length'] / df_processed['height_cm']
    
    # Select features for clustering
    feature_cols = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                   'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
    
    # Remove rows with too many missing values
    df_processed = df_processed.dropna(subset=feature_cols, how='all')
    
    print(f"✅ Preprocessing complete. Using {len(feature_cols)} features:")
    print(f"   {', '.join(feature_cols[:5])}... (+{len(feature_cols)-5} more)")
    
    return df_processed, feature_cols

def apply_sport_specific_weighting(df, feature_cols):
    """Apply sport-specific feature weighting to emphasize relevant body traits."""
    print("   Applying sport-specific feature weighting...")
    
    df_weighted = df.copy()
    
    # Define sport-specific weights
    sport_weights = {
        'basketball': {
            'height_cm': 1.5, 'weight_kg': 1.2, 'bmi': 1.1,
            'arm_span_ratio': 1.4, 'leg_length_ratio': 1.3, 'torso_length_ratio': 1.1
        },
        'gymnastics': {
            'height_cm': 0.8, 'weight_kg': 1.3, 'bmi': 1.4,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.1, 'torso_length_ratio': 1.2
        },
        'swimming': {
            'height_cm': 1.2, 'weight_kg': 1.1, 'bmi': 1.0,
            'arm_span_ratio': 1.5, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.4
        },
        'weightlifting': {
            'height_cm': 1.0, 'weight_kg': 1.6, 'bmi': 1.7,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.1
        },
        'distance_running': {
            'height_cm': 1.0, 'weight_kg': 0.7, 'bmi': 0.6,
            'arm_span_ratio': 1.0, 'leg_length_ratio': 1.1, 'torso_length_ratio': 1.0
        },
        'track_field': {
            'height_cm': 1.1, 'weight_kg': 1.2, 'bmi': 1.1,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.0
        },
        'soccer': {
            'height_cm': 1.1, 'weight_kg': 1.0, 'bmi': 1.0,
            'arm_span_ratio': 1.0, 'leg_length_ratio': 1.1, 'torso_length_ratio': 1.0
        },
        'tennis': {
            'height_cm': 1.1, 'weight_kg': 1.0, 'bmi': 1.0,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.0, 'torso_length_ratio': 1.0
        },
        'rowing': {
            'height_cm': 1.3, 'weight_kg': 1.2, 'bmi': 1.1,
            'arm_span_ratio': 1.2, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.2
        },
        'volleyball': {
            'height_cm': 1.4, 'weight_kg': 1.1, 'bmi': 1.0,
            'arm_span_ratio': 1.3, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.1
        }
    }
    
    # Apply weights
    for sport, weights in sport_weights.items():
        sport_mask = df_weighted['sport'] == sport
        for feature, weight in weights.items():
            if feature in df_weighted.columns:
                df_weighted.loc[sport_mask, feature] *= weight
    
    return df_weighted

def find_optimal_clusters(df, feature_cols, max_k=10):
    """Find optimal number of clusters using multiple metrics."""
    print(f"🔍 Finding optimal number of clusters...")
    
    # Prepare data - handle missing values
    X = df[feature_cols].copy()
    
    # Convert all columns to numeric and handle missing values
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values with column means
    X = X.fillna(X.mean())
    
    # Check for any remaining NaN values and fill with 0 if needed
    if X.isnull().any().any():
        print(f"   Warning: Still have NaN values after mean imputation, filling with 0")
        X = X.fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different k values
    k_range = range(2, min(max_k + 1, len(df) // 10))
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    
    print(f"💡 Optimal number of clusters: {optimal_k}")
    print(f"   Silhouette score: {best_silhouette:.3f}")
    
    return optimal_k, best_silhouette

def perform_clustering(df, feature_cols, k, gender=None):
    """Perform K-means clustering."""
    if gender:
        print(f"🔍 Finding optimal number of clusters for {gender} athletes...")
    else:
        print(f"🔍 Finding optimal number of clusters...")
    
    # Prepare data - handle missing values
    X = df[feature_cols].copy()
    
    # Convert all columns to numeric and handle missing values
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values with column means
    X = X.fillna(X.mean())
    
    # Check for any remaining NaN values and fill with 0 if needed
    if X.isnull().any().any():
        print(f"   Warning: Still have NaN values after mean imputation, filling with 0")
        X = X.fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    print(f"🔍 Performing K-means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
    
    print(f"✅ Clustering complete")
    print(f"   Silhouette score: {silhouette:.3f}")
    print(f"   Calinski-Harabasz score: {calinski:.1f}")
    
    return cluster_labels, X_scaled, silhouette, calinski

def analyze_clusters(df, cluster_labels, feature_cols):
    """Analyze cluster characteristics."""
    print("\n📊 Analyzing clusters...")
    
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

def create_clustering_visualizations(df, cluster_labels, X_scaled, title_suffix=""):
    """Create PCA visualization of clusters."""
    print("\n🎨 Creating visualizations...")
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot clusters
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
    
    # Add cluster centers
    cluster_centers_pca = pca.transform(
        StandardScaler().fit_transform(
            df.groupby('cluster')[['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                                 'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']].mean()
        )
    )
    ax.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
              c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
    
    # Formatting
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(f'Athlete Body Type Clustering{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')
    
    plt.tight_layout()
    return fig

def create_results_text_boxes(cluster_stats, title_suffix=""):
    """Create text boxes with clustering results."""
    print("\n📝 Creating results text boxes...")
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 12))
    
    # Define colors for each analysis
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    titles = ['Combined Analysis', 'Male Analysis', 'Female Analysis']
    
    # For now, we'll use the same data for all three (this will be updated when we have separate analyses)
    for i, (ax, color, title) in enumerate(zip([ax1, ax2, ax3], colors, titles)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create text box
        text_content = f"🏆 {title}\n"
        text_content += "=" * 50 + "\n\n"
        
        if i == 0:  # Combined
            text_content += f"📊 Total Clusters: {len(cluster_stats)}\n"
            text_content += f"📈 Total Athletes: {sum(stat['size'] for stat in cluster_stats)}\n\n"
        elif i == 1:  # Male
            text_content += f"📊 Male Clusters: {len(cluster_stats)}\n"
            text_content += f"📈 Male Athletes: {sum(stat['size'] for stat in cluster_stats)}\n\n"
        else:  # Female
            text_content += f"📊 Female Clusters: {len(cluster_stats)}\n"
            text_content += f"📈 Female Athletes: {sum(stat['size'] for stat in cluster_stats)}\n\n"
        
        # Add cluster summaries
        for stat in cluster_stats:
            text_content += f"🔹 Cluster {stat['cluster_id']}: {stat['size']} athletes ({stat['percentage']:.1f}%)\n"
            text_content += f"   Height: {stat['avg_height']:.1f}cm, Weight: {stat['avg_weight']:.1f}kg\n"
            text_content += f"   BMI: {stat['avg_bmi']:.1f}\n"
            text_content += f"   Top Sports: {', '.join(stat['top_sports'].head(2).index)}\n\n"
        
        # Add text box
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.1))
        
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    
    plt.suptitle(f'Athlete Body Type Clustering Results{title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    """Main execution function."""
    print("🚀 SPORT-SPECIFIC ATHLETE BODY TYPE CLUSTERING PIPELINE")
    print("=" * 70)
    
    # Load and preprocess data
    df, feature_cols = load_and_preprocess_data()
    
    # Apply sport-specific weighting
    df_weighted = apply_sport_specific_weighting(df, feature_cols)
    
    # Store results for all three analyses
    all_results = {}
    
    # 1. Combined Analysis
    print("\n" + "=" * 60)
    print("🏆 ANALYZING ALL ATHLETES")
    print("=" * 60)
    
    optimal_k_combined, best_silhouette_combined = find_optimal_clusters(df_weighted, feature_cols)
    cluster_labels_combined, X_scaled_combined, silhouette_combined, calinski_combined = perform_clustering(df_weighted, feature_cols, optimal_k_combined)
    cluster_stats_combined = analyze_clusters(df_weighted, cluster_labels_combined, feature_cols)
    
    all_results['combined'] = {
        'df': df_weighted,
        'cluster_labels': cluster_labels_combined,
        'X_scaled': X_scaled_combined,
        'cluster_stats': cluster_stats_combined,
        'k': optimal_k_combined,
        'silhouette': silhouette_combined
    }
    
    # 2. Male Analysis
    print("\n" + "=" * 60)
    print("🏃 ANALYZING MALE ATHLETES")
    print("=" * 60)
    
    df_male = df_weighted[df_weighted['Sex'] == 'M'].copy()
    print(f"   Filtering to M athletes only: {len(df_male)} athletes")
    
    optimal_k_male, best_silhouette_male = find_optimal_clusters(df_male, feature_cols)
    cluster_labels_male, X_scaled_male, silhouette_male, calinski_male = perform_clustering(df_male, feature_cols, optimal_k_male)
    cluster_stats_male = analyze_clusters(df_male, cluster_labels_male, feature_cols)
    
    all_results['male'] = {
        'df': df_male,
        'cluster_labels': cluster_labels_male,
        'X_scaled': X_scaled_male,
        'cluster_stats': cluster_stats_male,
        'k': optimal_k_male,
        'silhouette': silhouette_male
    }
    
    # 3. Female Analysis
    print("\n" + "=" * 60)
    print("🏃 ANALYZING FEMALE ATHLETES")
    print("=" * 60)
    
    df_female = df_weighted[df_weighted['Sex'] == 'F'].copy()
    print(f"   Filtering to F athletes only: {len(df_female)} athletes")
    
    optimal_k_female, best_silhouette_female = find_optimal_clusters(df_female, feature_cols)
    cluster_labels_female, X_scaled_female, silhouette_female, calinski_female = perform_clustering(df_female, feature_cols, optimal_k_female)
    cluster_stats_female = analyze_clusters(df_female, cluster_labels_female, feature_cols)
    
    all_results['female'] = {
        'df': df_female,
        'cluster_labels': cluster_labels_female,
        'X_scaled': X_scaled_female,
        'cluster_stats': cluster_stats_female,
        'k': optimal_k_female,
        'silhouette': silhouette_female
    }
    
    # Create combined visualization with all three PCA plots
    print("\n🎨 Creating combined visualizations...")
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
                    fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')
    
    plt.suptitle('Athlete Body Type Clustering - PCA Visualizations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig('clustering_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'clustering_visualizations.png'")
    
    # Create results text boxes
    print("\n📝 Creating results text boxes...")
    fig_results, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))
    
    for i, (analysis_type, ax, title, color) in enumerate(zip(['combined', 'male', 'female'], [ax1, ax2, ax3], titles, colors)):
        result = all_results[analysis_type]
        cluster_stats = result['cluster_stats']
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create text box
        text_content = f"🏆 {title}\n"
        text_content += "=" * 50 + "\n\n"
        text_content += f"📊 Clusters: {result['k']}\n"
        text_content += f"📈 Athletes: {sum(stat['size'] for stat in cluster_stats)}\n"
        text_content += f"🎯 Silhouette: {result['silhouette']:.3f}\n\n"
        
        # Add cluster summaries
        for stat in cluster_stats:
            text_content += f"🔹 Cluster {stat['cluster_id']}: {stat['size']} athletes ({stat['percentage']:.1f}%)\n"
            if not pd.isna(stat['avg_height']):
                text_content += f"   Height: {stat['avg_height']:.1f}cm"
            if not pd.isna(stat['avg_weight']):
                text_content += f", Weight: {stat['avg_weight']:.1f}kg"
            if not pd.isna(stat['avg_bmi']):
                text_content += f"\n   BMI: {stat['avg_bmi']:.1f}"
            text_content += f"\n   Top Sports: {', '.join(stat['top_sports'].head(2).index)}\n\n"
        
        # Add text box
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.1))
        
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    
    plt.suptitle('Athlete Body Type Clustering Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_results.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved as 'clustering_results.png'")
    
    # Save cluster assignments
    for analysis_type, result in all_results.items():
        df_output = result['df'].copy()
        df_output['cluster'] = result['cluster_labels']
        df_output.to_csv(f'final_clustering_results_{analysis_type}.csv', index=False)
        print(f"💾 Results saved to 'final_clustering_results_{analysis_type}.csv'")
    
    print("\n✅ Clustering analysis complete!")
    print("📁 Files generated:")
    print("   • clustering_visualizations.png")
    print("   • clustering_results.png")
    print("   • final_clustering_results_combined.csv")
    print("   • final_clustering_results_male.csv")
    print("   • final_clustering_results_female.csv")

if __name__ == "__main__":
    main()
