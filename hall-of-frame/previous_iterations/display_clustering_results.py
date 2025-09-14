"""
Display Clustering Results with Interactive Plots
===============================================

This script displays the clustering results with matplotlib plots
and creates a comprehensive summary visualization.
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

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

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

def cluster_and_analyze(df, gender=None, n_clusters=None):
    """Cluster athletes and return results."""
    if gender:
        data = df[df['Sex'] == gender].copy()
    else:
        data = df.copy()
    
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
    
    return data, kmeans, scaler, X_scaled, silhouette, max_cluster_pct

def create_comprehensive_visualization(df):
    """Create a comprehensive visualization showing all clustering results."""
    print("\n🎨 Creating comprehensive visualization...")
    
    # Perform all three clustering analyses
    data_all, _, _, X_scaled_all, sil_all, max_pct_all = cluster_and_analyze(df, gender=None, n_clusters=3)
    data_male, _, _, X_scaled_male, sil_male, max_pct_male = cluster_and_analyze(df, gender='M', n_clusters=4)
    data_female, _, _, X_scaled_female, sil_female, max_pct_female = cluster_and_analyze(df, gender='F', n_clusters=3)
    
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
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle('🏃 ATHLETE BODY TYPE CLUSTERING ANALYSIS\nComprehensive Results Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Combined Analysis - PCA Plot
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_all, palette='viridis', alpha=0.7, s=60, ax=ax1)
    ax1.set_title(f'Combined Analysis (All Athletes)\n3 clusters, silhouette={sil_all:.3f}', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Male Analysis - PCA Plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_male, palette='Blues', alpha=0.7, s=60, ax=ax2)
    ax2.set_title(f'Male Athletes\n4 clusters, silhouette={sil_male:.3f}', 
                  fontweight='bold', fontsize=12)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Female Analysis - PCA Plot
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                   data=data_female, palette='Reds', alpha=0.7, s=60, ax=ax3)
    ax3.set_title(f'Female Athletes\n3 clusters, silhouette={sil_female:.3f}', 
                  fontweight='bold', fontsize=12)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax3.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Cluster Size Comparison
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Get cluster sizes
    combined_sizes = [len(data_all[data_all['cluster'] == i]) for i in range(3)]
    male_sizes = [len(data_male[data_male['cluster'] == i]) for i in range(4)]
    female_sizes = [len(data_female[data_female['cluster'] == i]) for i in range(3)]
    
    # Create bar chart for cluster sizes
    categories = ['Combined', 'Male', 'Female']
    max_cluster_sizes = [max(combined_sizes), max(male_sizes), max(female_sizes)]
    
    bars = ax4.bar(categories, max_cluster_sizes, color=['purple', 'blue', 'red'], alpha=0.7)
    ax4.set_title('Max Cluster Size by Analysis', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Number of Athletes')
    
    # Add value labels on bars
    for bar, size in zip(bars, max_cluster_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{size}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Sport Distribution - Combined
    ax5 = fig.add_subplot(gs[1, 0])
    sport_cluster = pd.crosstab(data_all['sport'], data_all['cluster'], normalize='columns')
    sns.heatmap(sport_cluster, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax5)
    ax5.set_title('Sport Distribution - Combined', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Sport')
    
    # 6. Sport Distribution - Male
    ax6 = fig.add_subplot(gs[1, 1])
    sport_cluster_male = pd.crosstab(data_male['sport'], data_male['cluster'], normalize='columns')
    sns.heatmap(sport_cluster_male, cmap='Blues', annot=True, fmt='.2f', ax=ax6)
    ax6.set_title('Sport Distribution - Male', fontweight='bold', fontsize=11)
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Sport')
    
    # 7. Sport Distribution - Female
    ax7 = fig.add_subplot(gs[1, 2])
    sport_cluster_female = pd.crosstab(data_female['sport'], data_female['cluster'], normalize='columns')
    sns.heatmap(sport_cluster_female, cmap='Reds', annot=True, fmt='.2f', ax=ax7)
    ax7.set_title('Sport Distribution - Female', fontweight='bold', fontsize=11)
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Sport')
    
    # 8. Body Measurements by Cluster - Combined
    ax8 = fig.add_subplot(gs[1, 3])
    features = ['height_cm', 'weight_kg', 'bmi']
    cluster_means = data_all.groupby('cluster')[features].mean()
    
    x = np.arange(len(cluster_means))
    width = 0.25
    
    for i, feature in enumerate(features):
        ax8.bar(x + i*width, cluster_means[feature], width, 
                label=feature, alpha=0.8)
    
    ax8.set_title('Body Measurements by Cluster - Combined', fontweight='bold', fontsize=11)
    ax8.set_xlabel('Cluster')
    ax8.set_ylabel('Value')
    ax8.set_xticks(x + width)
    ax8.set_xticklabels(['0', '1', '2'])
    ax8.legend()
    
    # 9-12. Top Sports per Cluster for Combined Analysis
    for i in range(4):
        if i < 3:  # Only 3 clusters in combined analysis
            ax = fig.add_subplot(gs[2, i])
            cluster_data = data_all[data_all['cluster'] == i]
            sport_counts = cluster_data['sport'].value_counts().head(5)
            
            if len(sport_counts) > 0:
                sns.barplot(x=sport_counts.values, y=sport_counts.index, 
                           palette='viridis', ax=ax)
                ax.set_title(f'Top Sports - Combined Cluster {i}', fontweight='bold', fontsize=11)
                ax.set_xlabel('Number of Athletes')
                ax.set_ylabel('')
        else:
            # Summary statistics
            ax = fig.add_subplot(gs[2, i])
            ax.axis('off')
            
            # Add summary text
            summary_text = f"""
CLUSTERING SUMMARY

Combined Analysis:
• 3 clusters, silhouette={sil_all:.3f}
• Max cluster: {max_pct_all:.1f}%

Male Analysis:
• 4 clusters, silhouette={sil_male:.3f}
• Max cluster: {max_pct_male:.1f}%

Female Analysis:
• 3 clusters, silhouette={sil_female:.3f}
• Max cluster: {max_pct_female:.1f}%

Total Athletes: {len(df)}
Male: {len(df[df['Sex'] == 'M'])}
Female: {len(df[df['Sex'] == 'F'])}
            """
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 13-16. Body Type Archetype Descriptions
    for i in range(4):
        ax = fig.add_subplot(gs[3, i])
        ax.axis('off')
        
        if i < 3:  # Combined clusters
            cluster_data = data_all[data_all['cluster'] == i]
            cluster_means = cluster_data[['height_cm', 'weight_kg', 'bmi']].mean()
            
            # Get top sports
            top_sports = cluster_data['sport'].value_counts().head(3)
            
            # Body type description
            height = cluster_means['height_cm']
            weight = cluster_means['weight_kg']
            bmi = cluster_means['bmi']
            
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
            
            archetype_text = f"""
CLUSTER {i} ARCHETYPE
({len(cluster_data)} athletes)

Body Type:
{height_desc}, {build_desc}

Measurements:
• Height: {height:.1f} cm
• Weight: {weight:.1f} kg
• BMI: {bmi:.1f}

Top Sports:
{chr(10).join([f'• {sport}: {count} athletes' for sport, count in top_sports.items()])}
            """
            
            ax.text(0.05, 0.95, archetype_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            # Instructions
            instructions_text = """
HOW TO USE THESE RESULTS:

1. Identify your body type by comparing
   your measurements to cluster averages

2. Look at the dominant sports for your
   cluster to find suitable activities

3. Use gender-specific analysis for
   more accurate recommendations

4. Consider multiple clusters if you
   fall between body types
            """
            ax.text(0.05, 0.95, instructions_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_clustering_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive dashboard saved as 'comprehensive_clustering_dashboard.png'")
    
    # Show the plot
    plt.show()
    
    return fig

def main():
    """Run the comprehensive visualization."""
    print("🚀 CREATING COMPREHENSIVE CLUSTERING DASHBOARD")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create comprehensive visualization
    fig = create_comprehensive_visualization(df)
    
    print("\n✅ Dashboard created and displayed!")
    print("📊 The visualization shows:")
    print("   • PCA plots for all three analyses")
    print("   • Sport distributions by cluster")
    print("   • Body measurements comparisons")
    print("   • Top sports per cluster")
    print("   • Body type archetype descriptions")
    print("   • Summary statistics")

if __name__ == "__main__":
    main()
