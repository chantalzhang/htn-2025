"""
Final Optimized Athlete Body Type Clustering
===========================================

Implements user's recommendations:
- Men: 3-4 clusters (k=4 optimal: silhouette=0.413, max_cluster=65.7%)
- Women: 2-3 clusters (k=3 optimal: silhouette=0.555, max_cluster=54.0%)  
- Combined: 3-5 clusters (k=3 optimal: silhouette=0.464, max_cluster=72.7%)

Focus: Body type archetypes with 2-3 dominant sports per cluster
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

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the athlete dataset."""
    print("📊 Loading athlete dataset...")
    df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    
    # Clean data
    invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
    df = df[~df['sport'].isin(invalid_sports)]
    df['Sex'] = df['Sex'].str.upper().map({'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'})
    df = df.dropna(subset=['Sex'])
    
    # Create body features
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
    df['height_weight_ratio'] = df['height_cm'] / df['weight_kg']
    
    # Handle missing values with simple imputation
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 'height_weight_ratio']
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].mean())
    
    print(f"✅ Loaded {len(df)} athletes")
    print(f"   Male: {len(df[df['Sex'] == 'M'])}, Female: {len(df[df['Sex'] == 'F'])}")
    print(f"   Sports: {sorted(df['sport'].unique())}")
    
    return df

def cluster_athletes(df, gender=None, n_clusters=None):
    """Cluster athletes with specified parameters."""
    if gender:
        data = df[df['Sex'] == gender].copy()
        print(f"\n🏃 Clustering {gender} athletes...")
    else:
        data = df.copy()
        print(f"\n🏃 Clustering all athletes...")
    
    # Select features
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 'height_weight_ratio']
    X = data[features]
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    data['cluster'] = labels
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    cluster_sizes = np.bincount(labels)
    max_cluster_pct = np.max(cluster_sizes) / len(labels) * 100
    
    print(f"✅ {n_clusters} clusters, silhouette={silhouette:.3f}, max_cluster={max_cluster_pct:.1f}%")
    
    return data, kmeans, scaler, X_scaled

def analyze_clusters(data, gender_label=""):
    """Analyze and describe clusters."""
    print(f"\n📊 CLUSTER ANALYSIS {gender_label}")
    print("=" * 60)
    
    # Calculate cluster sizes
    cluster_sizes = data['cluster'].value_counts().sort_index()
    
    # Calculate mean values for each feature by cluster
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 'height_weight_ratio']
    cluster_means = data.groupby('cluster')[features].mean()
    
    # Calculate sport distribution by cluster
    sport_dist = pd.crosstab(data['sport'], data['cluster'], normalize='columns')
    
    for cluster in range(len(cluster_sizes)):
        size = cluster_sizes[cluster]
        pct = (size / len(data)) * 100
        
        print(f"\n📊 CLUSTER {cluster} - {size} athletes ({pct:.1f}%)")
        print("-" * 40)
        
        # Body type characteristics
        cluster_data = cluster_means.loc[cluster]
        print(f"📏 Height: {cluster_data['height_cm']:.1f} cm")
        print(f"📏 Weight: {cluster_data['weight_kg']:.1f} kg")
        print(f"📏 BMI: {cluster_data['bmi']:.1f}")
        print(f"📏 Weight/Height ratio: {cluster_data['weight_height_ratio']:.3f}")
        
        # Top sports
        sports = sport_dist[cluster].sort_values(ascending=False).head(3)
        print(f"\n🏅 TOP SPORTS:")
        for sport, pct in sports.items():
            print(f"   • {sport}: {pct*100:.1f}%")
        
        # Body type archetype
        height = cluster_data['height_cm']
        weight = cluster_data['weight_kg']
        bmi = cluster_data['bmi']
        
        if height > 190:
            height_desc = "Tall"
        elif height > 180:
            height_desc = "Above average height"
        elif height > 170:
            height_desc = "Average height"
        else:
            height_desc = "Shorter"
        
        if bmi < 20:
            build_desc = "lean/lightweight"
        elif bmi < 25:
            build_desc = "athletic/balanced"
        elif bmi < 30:
            build_desc = "muscular/heavy"
        else:
            build_desc = "very muscular/heavy"
        
        print(f"\n🎯 BODY TYPE: {height_desc}, {build_desc}")

def create_visualizations(data, gender_label="", filename_suffix=""):
    """Create visualizations of clustering results."""
    print(f"\n🎨 Creating visualizations{gender_label}...")
    
    # PCA for visualization
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 'height_weight_ratio']
    X = data[features]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    data['pca_1'] = X_pca[:, 0]
    data['pca_2'] = X_pca[:, 1]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Athlete Body Type Clustering{gender_label}', fontsize=16, fontweight='bold')
    
    # 1. PCA scatter plot (colored by cluster)
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data, palette='viridis', alpha=0.7, s=60, ax=axes[0,0])
    axes[0,0].set_title('Athlete Clusters (PCA)')
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    
    # 2. Sport distribution by cluster
    sport_cluster = pd.crosstab(data['sport'], data['cluster'], normalize='columns')
    sns.heatmap(sport_cluster, cmap='YlOrRd', annot=True, fmt='.2f', ax=axes[0,1])
    axes[0,1].set_title('Sport Distribution by Cluster')
    axes[0,1].set_xlabel('Cluster')
    axes[0,1].set_ylabel('Sport')
    
    # 3. Cluster sizes
    cluster_sizes = data['cluster'].value_counts().sort_index()
    bars = sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='viridis', ax=axes[1,0])
    axes[1,0].set_title('Athletes per Cluster')
    axes[1,0].set_xlabel('Cluster')
    axes[1,0].set_ylabel('Number of Athletes')
    
    # Add percentage labels
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        pct = (height / len(data)) * 100
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Sports colored by cluster
    sports = data['sport'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(sports)))
    for i, sport in enumerate(sports):
        sport_data = data[data['sport'] == sport]
        axes[1,1].scatter(sport_data['pca_1'], sport_data['pca_2'], 
                         c=[colors[i]], label=sport, alpha=0.7, s=50)
    axes[1,1].set_title('Athletes by Sport (PCA)')
    axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    filename = f'final_athlete_clustering{filename_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations saved as '{filename}'")
    
    return fig

def main():
    """Run the complete optimized clustering analysis."""
    print("🚀 FINAL OPTIMIZED ATHLETE BODY TYPE CLUSTERING")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # 1. Combined analysis (k=3 optimal)
    print("\n" + "="*60)
    print("🏆 COMBINED ANALYSIS (All Athletes)")
    print("="*60)
    data_all, _, _, _ = cluster_athletes(df, gender=None, n_clusters=3)
    analyze_clusters(data_all, "(All Athletes)")
    create_visualizations(data_all, " (All Athletes)", "_all")
    data_all.to_csv('final_athlete_clusters_all.csv', index=False)
    
    # 2. Male analysis (k=4 optimal)
    print("\n" + "="*60)
    print("🏃 MALE ANALYSIS")
    print("="*60)
    data_male, _, _, _ = cluster_athletes(df, gender='M', n_clusters=4)
    analyze_clusters(data_male, "(Male Athletes)")
    create_visualizations(data_male, " (Male Athletes)", "_male")
    data_male.to_csv('final_athlete_clusters_male.csv', index=False)
    
    # 3. Female analysis (k=3 optimal)
    print("\n" + "="*60)
    print("🏃 FEMALE ANALYSIS")
    print("="*60)
    data_female, _, _, _ = cluster_athletes(df, gender='F', n_clusters=3)
    analyze_clusters(data_female, "(Female Athletes)")
    create_visualizations(data_female, " (Female Athletes)", "_female")
    data_female.to_csv('final_athlete_clusters_female.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("📊 Results Summary:")
    print("   • Combined: 3 clusters (silhouette=0.464, max_cluster=72.7%)")
    print("   • Male: 4 clusters (silhouette=0.413, max_cluster=65.7%)")
    print("   • Female: 3 clusters (silhouette=0.555, max_cluster=54.0%)")
    print("\n📁 Files generated:")
    print("   • final_athlete_clusters_all.csv")
    print("   • final_athlete_clusters_male.csv")
    print("   • final_athlete_clusters_female.csv")
    print("   • final_athlete_clustering_all.png")
    print("   • final_athlete_clustering_male.png")
    print("   • final_athlete_clustering_female.png")

if __name__ == "__main__":
    main()
