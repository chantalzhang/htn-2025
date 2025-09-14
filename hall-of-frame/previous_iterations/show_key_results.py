"""
Show Key Clustering Results
==========================

A focused visualization showing the most important clustering results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_cluster_data():
    """Load data and perform clustering."""
    print("📊 Loading and clustering athlete data...")
    
    # Load data
    df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
    df = df[~df['sport'].isin(invalid_sports)]
    df['Sex'] = df['Sex'].str.upper().map({'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'})
    df = df.dropna(subset=['Sex'])
    
    # Create features
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio']
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].mean())
    
    # Cluster all athletes
    X = df[features]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['cluster'] = labels
    
    return df, X_scaled

def create_key_results_visualization(df, X_scaled):
    """Create a focused visualization of key results."""
    print("🎨 Creating key results visualization...")
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🏃 ATHLETE BODY TYPE CLUSTERING - KEY RESULTS', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. PCA Plot with clusters
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=df, palette='viridis', alpha=0.7, s=60, ax=axes[0,0])
    axes[0,0].set_title('Athlete Clusters (PCA Visualization)', fontweight='bold')
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0,0].legend(title='Cluster')
    
    # 2. Sport distribution heatmap
    sport_cluster = pd.crosstab(df['sport'], df['cluster'], normalize='columns')
    sns.heatmap(sport_cluster, cmap='YlOrRd', annot=True, fmt='.2f', ax=axes[0,1])
    axes[0,1].set_title('Sport Distribution by Cluster', fontweight='bold')
    axes[0,1].set_xlabel('Cluster')
    axes[0,1].set_ylabel('Sport')
    
    # 3. Cluster sizes
    cluster_sizes = df['cluster'].value_counts().sort_index()
    bars = sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, 
                      palette='viridis', ax=axes[0,2])
    axes[0,2].set_title('Athletes per Cluster', fontweight='bold')
    axes[0,2].set_xlabel('Cluster')
    axes[0,2].set_ylabel('Number of Athletes')
    
    # Add percentage labels
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        pct = (height / len(df)) * 100
        axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Body measurements by cluster
    features = ['height_cm', 'weight_kg', 'bmi']
    cluster_means = df.groupby('cluster')[features].mean()
    
    x = np.arange(len(cluster_means))
    width = 0.25
    
    for i, feature in enumerate(features):
        axes[1,0].bar(x + i*width, cluster_means[feature], width, 
                     label=feature, alpha=0.8)
    
    axes[1,0].set_title('Body Measurements by Cluster', fontweight='bold')
    axes[1,0].set_xlabel('Cluster')
    axes[1,0].set_ylabel('Value')
    axes[1,0].set_xticks(x + width)
    axes[1,0].set_xticklabels(['0', '1', '2'])
    axes[1,0].legend()
    
    # 5. Top sports per cluster
    for i in range(3):
        cluster_data = df[df['cluster'] == i]
        sport_counts = cluster_data['sport'].value_counts().head(5)
        
        if len(sport_counts) > 0:
            sns.barplot(x=sport_counts.values, y=sport_counts.index, 
                       palette='viridis', ax=axes[1,1])
            axes[1,1].set_title(f'Top Sports in Cluster {i}', fontweight='bold')
            axes[1,1].set_xlabel('Number of Athletes')
            axes[1,1].set_ylabel('')
            break
    
    # 6. Body type archetype summary
    axes[1,2].axis('off')
    
    # Calculate cluster summaries
    summaries = []
    for i in range(3):
        cluster_data = df[df['cluster'] == i]
        cluster_means = cluster_data[['height_cm', 'weight_kg', 'bmi']].mean()
        top_sports = cluster_data['sport'].value_counts().head(2)
        
        height = cluster_means['height_cm']
        weight = cluster_means['weight_kg']
        bmi = cluster_means['bmi']
        
        if height > 190:
            height_desc = "Tall"
        elif height > 180:
            height_desc = "Above average"
        elif height > 170:
            height_desc = "Average"
        else:
            height_desc = "Shorter"
        
        if bmi < 20:
            build_desc = "lean"
        elif bmi < 25:
            build_desc = "athletic"
        elif bmi < 30:
            build_desc = "muscular"
        else:
            build_desc = "very muscular"
        
        sport_list = ', '.join([f"{sport} ({count})" for sport, count in top_sports.items()])
        
        summaries.append(f"""
CLUSTER {i} ({len(cluster_data)} athletes)
{height_desc}, {build_desc}
{height:.0f}cm, {weight:.0f}kg, BMI {bmi:.1f}
Top sports: {sport_list}
        """)
    
    summary_text = "BODY TYPE ARCHETYPES:\n" + "\n".join(summaries)
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('key_clustering_results.png', dpi=300, bbox_inches='tight')
    print("✅ Key results saved as 'key_clustering_results.png'")
    
    # Show the plot
    plt.show()
    
    return fig

def print_summary(df):
    """Print a text summary of the results."""
    print("\n" + "="*60)
    print("📊 CLUSTERING RESULTS SUMMARY")
    print("="*60)
    
    for i in range(3):
        cluster_data = df[df['cluster'] == i]
        cluster_means = cluster_data[['height_cm', 'weight_kg', 'bmi']].mean()
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        print(f"\n🏆 CLUSTER {i} - {len(cluster_data)} athletes ({(len(cluster_data)/len(df)*100):.1f}%)")
        print(f"   Height: {cluster_means['height_cm']:.1f} cm")
        print(f"   Weight: {cluster_means['weight_kg']:.1f} kg")
        print(f"   BMI: {cluster_means['bmi']:.1f}")
        print("   Top Sports:")
        for sport, count in top_sports.items():
            pct = (count / len(cluster_data)) * 100
            print(f"     • {sport}: {count} athletes ({pct:.1f}%)")

def main():
    """Run the key results visualization."""
    print("🚀 SHOWING KEY CLUSTERING RESULTS")
    print("="*50)
    
    # Load and cluster data
    df, X_scaled = load_and_cluster_data()
    
    # Create visualization
    fig = create_key_results_visualization(df, X_scaled)
    
    # Print summary
    print_summary(df)
    
    print("\n✅ Key results displayed and saved!")

if __name__ == "__main__":
    main()
