"""
Improved K Clustering with Higher K Values
========================================

Uses higher k values for better body type separation:
- Combined: k=6
- Male: k=7  
- Female: k=5

Shows visual clusters and results.
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

def create_visualization_with_results(df):
    """Create visualization showing both clusters and results."""
    print("🎨 Creating improved clustering visualization...")
    
    # Perform clustering with higher k values
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
    
    # Create figure with 2 rows: visual clusters and results
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle('🏃 IMPROVED ATHLETE BODY TYPE CLUSTERING\nHigher K Values for Better Body Type Separation', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # ROW 1: Visual Clusters (PCA plots)
    # Combined PCA
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_all, palette='viridis', alpha=0.7, s=50, ax=ax1)
    ax1.set_title(f'Combined Analysis\n6 clusters, silhouette={sil_all:.3f}', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Male PCA
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_male, palette='Blues', alpha=0.7, s=50, ax=ax2)
    ax2.set_title(f'Male Analysis\n7 clusters, silhouette={sil_male:.3f}', 
                  fontweight='bold', fontsize=12)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Female PCA
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_female, palette='Reds', alpha=0.7, s=50, ax=ax3)
    ax3.set_title(f'Female Analysis\n5 clusters, silhouette={sil_female:.3f}', 
                  fontweight='bold', fontsize=12)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax3.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # ROW 2: Results - Combined Analysis
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('COMBINED ANALYSIS RESULTS (6 clusters)', fontweight='bold', fontsize=14)
    ax4.axis('off')
    
    combined_text = ""
    for i in range(6):
        cluster_data = data_all[data_all['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        # Get body measurements
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        combined_text += f"CLUSTER {i} ({len(cluster_data)} athletes) - Height: {height:.1f}cm, Weight: {weight:.1f}kg, BMI: {bmi:.1f}\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            combined_text += f"  {j}. {sport}: {count} athletes\n"
        combined_text += "\n"
    
    ax4.text(0.02, 0.98, combined_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ROW 3: Results - Male and Female Analysis
    # Male Results
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title('MALE ANALYSIS RESULTS (7 clusters)', fontweight='bold', fontsize=12)
    ax5.axis('off')
    
    male_text = ""
    for i in range(7):
        cluster_data = data_male[data_male['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        male_text += f"CLUSTER {i} ({len(cluster_data)} athletes)\n"
        male_text += f"  {height:.1f}cm, {weight:.1f}kg, BMI {bmi:.1f}\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            male_text += f"  {j}. {sport}: {count}\n"
        male_text += "\n"
    
    ax5.text(0.02, 0.98, male_text, transform=ax5.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Female Results
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.set_title('FEMALE ANALYSIS RESULTS (5 clusters)', fontweight='bold', fontsize=12)
    ax6.axis('off')
    
    female_text = ""
    for i in range(5):
        cluster_data = data_female[data_female['cluster'] == i]
        top_sports = cluster_data['sport'].value_counts().head(3)
        
        height = cluster_data['height_cm'].mean()
        weight = cluster_data['weight_kg'].mean()
        bmi = cluster_data['bmi'].mean()
        
        female_text += f"CLUSTER {i} ({len(cluster_data)} athletes) - Height: {height:.1f}cm, Weight: {weight:.1f}kg, BMI: {bmi:.1f}\n"
        for j, (sport, count) in enumerate(top_sports.items(), 1):
            female_text += f"  {j}. {sport}: {count} athletes\n"
        female_text += "\n"
    
    ax6.text(0.02, 0.98, female_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('improved_k_clustering_results.png', dpi=300, bbox_inches='tight')
    print("✅ Improved clustering results saved as 'improved_k_clustering_results.png'")
    
    # Show the plot
    plt.show()
    
    return fig

def print_summary(df):
    """Print a summary of the improved clustering results."""
    print("\n" + "="*80)
    print("📊 IMPROVED CLUSTERING RESULTS SUMMARY")
    print("="*80)
    
    # Perform clustering
    data_all, _, sil_all = cluster_athletes(df, gender=None, n_clusters=6)
    data_male, _, sil_male = cluster_athletes(df, gender='M', n_clusters=7)
    data_female, _, sil_female = cluster_athletes(df, gender='F', n_clusters=5)
    
    print(f"\n🏆 COMBINED ANALYSIS (6 clusters, silhouette={sil_all:.3f})")
    print("-" * 60)
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
    print("-" * 60)
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
    print("-" * 60)
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
    """Run the improved clustering visualization."""
    print("🚀 IMPROVED CLUSTERING WITH HIGHER K VALUES")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create visualization
    fig = create_visualization_with_results(df)
    
    # Print summary
    print_summary(df)
    
    print("\n✅ Improved clustering results displayed and saved!")

if __name__ == "__main__":
    main()
