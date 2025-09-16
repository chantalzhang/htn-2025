"""
Simple Clustering Results Display
===============================

Shows only the essential information:
- Clusters for men, women, and combined
- Silhouette scores
- Top 3 sports per cluster with athlete counts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style
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
    
    # Create features
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio']
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].mean())
    
    print(f"✅ Loaded {len(df)} athletes")
    print(f"   Male: {len(df[df['Sex'] == 'M'])}, Female: {len(df[df['Sex'] == 'F'])}")
    
    return df

def cluster_athletes(df, gender=None, n_clusters=None):
    """Cluster athletes and return results."""
    if gender:
        data = df[df['Sex'] == gender].copy()
    else:
        data = df.copy()
    
    # Select features
    features = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio']
    X = data[features]
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    data['cluster'] = labels
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, labels)
    
    return data, silhouette

def create_simple_visualization(df):
    """Create a simple visualization showing only the essential results."""
    print("🎨 Creating simple clustering visualization...")
    
    # Perform clustering for all three analyses
    data_all, sil_all = cluster_athletes(df, gender=None, n_clusters=3)
    data_male, sil_male = cluster_athletes(df, gender='M', n_clusters=4)
    data_female, sil_female = cluster_athletes(df, gender='F', n_clusters=3)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('🏃 ATHLETE BODY TYPE CLUSTERING RESULTS', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Combined Analysis
    ax1 = axes[0]
    ax1.set_title(f'COMBINED ANALYSIS\n3 clusters, silhouette={sil_all:.3f}', 
                  fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    combined_text = ""
    for i in range(3):
        cluster_data = data_all[data_all['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        combined_text += f"\nCLUSTER {i} ({len(cluster_data)} athletes):\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            combined_text += f"  {j}. {sport}: {count} athletes\n"
    
    ax1.text(0.05, 0.95, combined_text, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Male Analysis
    ax2 = axes[1]
    ax2.set_title(f'MALE ANALYSIS\n4 clusters, silhouette={sil_male:.3f}', 
                  fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    male_text = ""
    for i in range(4):
        cluster_data = data_male[data_male['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        male_text += f"\nCLUSTER {i} ({len(cluster_data)} athletes):\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            male_text += f"  {j}. {sport}: {count} athletes\n"
    
    ax2.text(0.05, 0.95, male_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Female Analysis
    ax3 = axes[2]
    ax3.set_title(f'FEMALE ANALYSIS\n3 clusters, silhouette={sil_female:.3f}', 
                  fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    female_text = ""
    for i in range(3):
        cluster_data = data_female[data_female['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        female_text += f"\nCLUSTER {i} ({len(cluster_data)} athletes):\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            female_text += f"  {j}. {sport}: {count} athletes\n"
    
    ax3.text(0.05, 0.95, female_text, transform=ax3.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('simple_clustering_results.png', dpi=300, bbox_inches='tight')
    print("✅ Simple results saved as 'simple_clustering_results.png'")
    
    # Show the plot
    plt.show()
    
    return fig

def print_text_summary(df):
    """Print a simple text summary."""
    print("\n" + "="*80)
    print("📊 CLUSTERING RESULTS SUMMARY")
    print("="*80)
    
    # Perform clustering
    data_all, sil_all = cluster_athletes(df, gender=None, n_clusters=3)
    data_male, sil_male = cluster_athletes(df, gender='M', n_clusters=4)
    data_female, sil_female = cluster_athletes(df, gender='F', n_clusters=3)
    
    print(f"\n🏆 COMBINED ANALYSIS (3 clusters, silhouette={sil_all:.3f})")
    print("-" * 50)
    for i in range(3):
        cluster_data = data_all[data_all['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        print(f"\nCluster {i} ({len(cluster_data)} athletes):")
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            print(f"  {j}. {sport}: {count} athletes")
    
    print(f"\n🏃 MALE ANALYSIS (4 clusters, silhouette={sil_male:.3f})")
    print("-" * 50)
    for i in range(4):
        cluster_data = data_male[data_male['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        print(f"\nCluster {i} ({len(cluster_data)} athletes):")
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            print(f"  {j}. {sport}: {count} athletes")
    
    print(f"\n🏃 FEMALE ANALYSIS (3 clusters, silhouette={sil_female:.3f})")
    print("-" * 50)
    for i in range(3):
        cluster_data = data_female[data_female['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        print(f"\nCluster {i} ({len(cluster_data)} athletes):")
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            print(f"  {j}. {sport}: {count} athletes")

def main():
    """Run the simple clustering visualization."""
    print("🚀 SIMPLE CLUSTERING RESULTS")
    print("="*50)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create simple visualization
    fig = create_simple_visualization(df)
    
    # Print text summary
    print_text_summary(df)
    
    print("\n✅ Simple results displayed and saved!")

if __name__ == "__main__":
    main()
