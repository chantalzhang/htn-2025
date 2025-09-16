"""
Separate Clustering Results
==========================

Creates two separate PNG files:
1. Visualizations only (PCA plots)
2. Results only (text summaries)

No overlapping, clean and simple.
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
    
    return data, X_scaled, silhouette

def create_visualizations_only(df):
    """Create visualizations only - PCA plots."""
    print("🎨 Creating visualizations...")
    
    # Perform clustering
    data_all, X_scaled_all, sil_all = cluster_athletes(df, gender=None, n_clusters=6)
    data_male, X_scaled_male, sil_male = cluster_athletes(df, gender='M', n_clusters=7)
    data_female, X_scaled_female, sil_female = cluster_athletes(df, gender='F', n_clusters=5)
    
    # Create PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    
    # Apply PCA to each dataset
    X_pca_all = pca.fit_transform(X_scaled_all)
    X_pca_male = pca.fit_transform(X_scaled_male)
    X_pca_female = pca.fit_transform(X_scaled_female)
    
    data_all['pca_1'] = X_pca_all[:, 0]
    data_all['pca_2'] = X_pca_all[:, 1]
    data_male['pca_1'] = X_pca_male[:, 0]
    data_male['pca_2'] = X_pca_male[:, 1]
    data_female['pca_1'] = X_pca_female[:, 0]
    data_female['pca_2'] = X_pca_female[:, 1]
    
    # Create figure for visualizations only
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🏃 ATHLETE BODY TYPE CLUSTERING - VISUALIZATIONS', 
                 fontsize=16, fontweight='bold')
    
    # Combined PCA
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_all, palette='viridis', alpha=0.7, s=60, ax=axes[0])
    axes[0].set_title(f'Combined Analysis\n6 clusters, silhouette={sil_all:.3f}', 
                      fontweight='bold', fontsize=12)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Male PCA
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_male, palette='Blues', alpha=0.7, s=60, ax=axes[1])
    axes[1].set_title(f'Male Analysis\n7 clusters, silhouette={sil_male:.3f}', 
                      fontweight='bold', fontsize=12)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Female PCA
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_female, palette='Reds', alpha=0.7, s=60, ax=axes[2])
    axes[2].set_title(f'Female Analysis\n5 clusters, silhouette={sil_female:.3f}', 
                      fontweight='bold', fontsize=12)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[2].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('clustering_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'clustering_visualizations.png'")
    
    plt.show()
    
    return fig

def create_results_only(df):
    """Create results only - three text boxes side by side."""
    print("📊 Creating results summary...")
    
    # Perform clustering
    data_all, _, sil_all = cluster_athletes(df, gender=None, n_clusters=6)
    data_male, _, sil_male = cluster_athletes(df, gender='M', n_clusters=7)
    data_female, _, sil_female = cluster_athletes(df, gender='F', n_clusters=5)
    
    # Create figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, 16))
    fig.suptitle('🏃 ATHLETE BODY TYPE CLUSTERING - RESULTS', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Combined Analysis Text
    combined_text = f"COMBINED ANALYSIS\n6 clusters, silhouette={sil_all:.3f}\n"
    combined_text += "=" * 40 + "\n\n"
    
    for i in range(6):
        cluster_data = data_all[data_all['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        combined_text += f"CLUSTER {i} ({len(cluster_data)} athletes)\n"
        combined_text += f"  {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}\n"
        combined_text += "  Top Sports:\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            combined_text += f"    {j}. {sport}: {count}\n"
        combined_text += "\n"
    
    # Male Analysis Text
    male_text = f"MALE ANALYSIS\n7 clusters, silhouette={sil_male:.3f}\n"
    male_text += "=" * 40 + "\n\n"
    
    for i in range(7):
        cluster_data = data_male[data_male['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        male_text += f"CLUSTER {i} ({len(cluster_data)} athletes)\n"
        male_text += f"  {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}\n"
        male_text += "  Top Sports:\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            male_text += f"    {j}. {sport}: {count}\n"
        male_text += "\n"
    
    # Female Analysis Text
    female_text = f"FEMALE ANALYSIS\n5 clusters, silhouette={sil_female:.3f}\n"
    female_text += "=" * 40 + "\n\n"
    
    for i in range(5):
        cluster_data = data_female[data_female['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        female_text += f"CLUSTER {i} ({len(cluster_data)} athletes)\n"
        female_text += f"  {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}\n"
        female_text += "  Top Sports:\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            female_text += f"    {j}. {sport}: {count}\n"
        female_text += "\n"
    
    # Display results in three separate text boxes
    axes[0].axis('off')
    axes[0].text(0.02, 0.98, combined_text, transform=axes[0].transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    axes[1].axis('off')
    axes[1].text(0.02, 0.98, male_text, transform=axes[1].transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    axes[2].axis('off')
    axes[2].text(0.02, 0.98, female_text, transform=axes[2].transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved as 'clustering_results.png'")
    
    plt.show()
    
    return fig

def print_console_summary(df):
    """Print summary to console."""
    print("\n" + "="*80)
    print("📊 CLUSTERING RESULTS SUMMARY")
    print("="*80)
    
    # Perform clustering
    data_all, _, sil_all = cluster_athletes(df, gender=None, n_clusters=6)
    data_male, _, sil_male = cluster_athletes(df, gender='M', n_clusters=7)
    data_female, _, sil_female = cluster_athletes(df, gender='F', n_clusters=5)
    
    print(f"\n🏆 COMBINED ANALYSIS (6 clusters, silhouette={sil_all:.3f})")
    print("-" * 70)
    for i in range(6):
        cluster_data = data_all[data_all['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        print(f"\nCluster {i} ({len(cluster_data)} athletes) - {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}:")
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            print(f"  {j}. {sport}: {count} athletes")
    
    print(f"\n🏃 MALE ANALYSIS (7 clusters, silhouette={sil_male:.3f})")
    print("-" * 70)
    for i in range(7):
        cluster_data = data_male[data_male['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        print(f"\nCluster {i} ({len(cluster_data)} athletes) - {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}:")
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            print(f"  {j}. {sport}: {count} athletes")
    
    print(f"\n🏃 FEMALE ANALYSIS (5 clusters, silhouette={sil_female:.3f})")
    print("-" * 70)
    for i in range(5):
        cluster_data = data_female[data_female['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        print(f"\nCluster {i} ({len(cluster_data)} athletes) - {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}:")
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            print(f"  {j}. {sport}: {count} athletes")

def main():
    """Run the separate clustering visualization."""
    print("🚀 SEPARATE CLUSTERING RESULTS")
    print("="*50)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create visualizations only
    fig1 = create_visualizations_only(df)
    
    # Create results only
    fig2 = create_results_only(df)
    
    # Print console summary
    print_console_summary(df)
    
    print("\n✅ Two separate files created:")
    print("   📊 clustering_visualizations.png - PCA plots only")
    print("   📋 clustering_results.png - Text results only")

if __name__ == "__main__":
    main()
