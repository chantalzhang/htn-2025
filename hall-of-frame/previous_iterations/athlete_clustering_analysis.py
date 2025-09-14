"""
Athlete Body Type Clustering Analysis
=====================================

Complete workflow for clustering athletes by body type using physical measurements.
Designed for ~300 athletes across 10 sports with missing value handling.

Author: AI Assistant
Date: 2025-09-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AthleteClusteringAnalysis:
    """
    Complete clustering analysis pipeline for athlete body types.
    """
    
    def __init__(self, data_path):
        """Initialize with dataset path."""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.clusters = {}
        
    def load_data(self):
        """
        Step 1: Load the athlete dataset from CSV.
        """
        print("📊 Loading athlete dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} athletes across {self.df['sport'].nunique()} sports")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        # Display basic info
        print(f"\n🔍 Dataset Overview:")
        print(f"   • Sports: {sorted(self.df['sport'].unique())}")
        print(f"   • Gender distribution: {dict(self.df['Sex'].value_counts())}")
        print(f"   • Missing values per column:")
        missing = self.df.isnull().sum()
        for col, count in missing[missing > 0].items():
            print(f"     - {col}: {count} ({count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def handle_missing_values(self):
        """
        Step 2: Handle missing values with intelligent imputation.
        Impute with mean per sex and sport where possible.
        """
        print("\n🔧 Handling missing values...")
        self.df_processed = self.df.copy()
        
        # Clean Sex column - handle missing values
        if self.df_processed['Sex'].isnull().any():
            print("   Warning: Some athletes missing gender info, filling with 'M' as default")
            self.df_processed['Sex'].fillna('M', inplace=True)
        
        # Define numeric columns for imputation
        numeric_cols = ['height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 'Torso Length']
        
        # Filter to existing columns and check for numeric data
        numeric_cols = [col for col in numeric_cols if col in self.df_processed.columns]
        
        for col in numeric_cols:
            # Convert to numeric first, handling any string values
            self.df_processed[col] = pd.to_numeric(self.df_processed[col], errors='coerce')
            
            if self.df_processed[col].isnull().any():
                print(f"   Imputing {col}...")
                
                # Try imputation by sex and sport first
                for sex in ['M', 'F']:
                    for sport in self.df_processed['sport'].unique():
                        mask = (self.df_processed['Sex'] == sex) & (self.df_processed['sport'] == sport)
                        subset = self.df_processed.loc[mask, col]
                        
                        if subset.isnull().any() and subset.notna().any():
                            mean_val = subset.mean()
                            self.df_processed.loc[mask & self.df_processed[col].isnull(), col] = mean_val
                
                # Fallback: impute by sex only
                for sex in ['M', 'F']:
                    mask = self.df_processed['Sex'] == sex
                    subset = self.df_processed.loc[mask, col]
                    
                    if subset.isnull().any() and subset.notna().any():
                        mean_val = subset.mean()
                        self.df_processed.loc[mask & self.df_processed[col].isnull(), col] = mean_val
                
                # Final fallback: overall mean
                if self.df_processed[col].isnull().any():
                    overall_mean = self.df_processed[col].mean()
                    if pd.notna(overall_mean):
                        self.df_processed[col].fillna(overall_mean, inplace=True)
                    else:
                        # If still no valid data, use reasonable defaults
                        defaults = {'height_cm': 180, 'weight_kg': 75, 'Arm Span': 180, 'Leg Length': 90, 'Torso Length': 90}
                        self.df_processed[col].fillna(defaults.get(col, 0), inplace=True)
        
        print("✅ Missing value imputation complete")
        return self.df_processed
    
    def create_body_ratios(self):
        """
        Step 3: Create body ratio features to normalize proportions.
        """
        print("\n📐 Creating body ratio features...")
        
        # BMI = weight / height²
        self.df_processed['BMI'] = self.df_processed['weight_kg'] / (self.df_processed['height_cm'] / 100) ** 2
        
        # Arm span to height ratio
        if 'Arm Span' in self.df_processed.columns:
            self.df_processed['arm_span_height_ratio'] = self.df_processed['Arm Span'] / self.df_processed['height_cm']
        
        # Leg length to height ratio
        if 'Leg Length' in self.df_processed.columns:
            self.df_processed['leg_height_ratio'] = self.df_processed['Leg Length'] / self.df_processed['height_cm']
        
        # Torso length to height ratio
        if 'Torso Length' in self.df_processed.columns:
            self.df_processed['torso_height_ratio'] = self.df_processed['Torso Length'] / self.df_processed['height_cm']
        
        print("✅ Body ratios created:")
        ratio_cols = ['BMI', 'arm_span_height_ratio', 'leg_height_ratio', 'torso_height_ratio']
        for col in ratio_cols:
            if col in self.df_processed.columns:
                print(f"   • {col}: mean={self.df_processed[col].mean():.3f}, std={self.df_processed[col].std():.3f}")
        
        return self.df_processed
    
    def prepare_features(self):
        """
        Step 4: Prepare and standardize features for clustering.
        """
        print("\n⚙️ Preparing features for clustering...")
        
        # Define feature columns
        feature_cols = ['height_cm', 'weight_kg', 'BMI']
        
        # Add ratio features if they exist
        ratio_cols = ['arm_span_height_ratio', 'leg_height_ratio', 'torso_height_ratio']
        for col in ratio_cols:
            if col in self.df_processed.columns:
                feature_cols.append(col)
        
        # Extract features
        self.features = self.df_processed[feature_cols].copy()
        
        # Standardize features (z-score normalization)
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=feature_cols,
            index=self.features.index
        )
        
        print(f"✅ Prepared {len(feature_cols)} features: {feature_cols}")
        print(f"   Features shape: {self.features_scaled.shape}")
        
        return self.features_scaled
    
    def find_optimal_clusters(self, data, max_k=10):
        """
        Step 5: Find optimal number of clusters using elbow method and silhouette score.
        """
        print(f"\n🔍 Finding optimal number of clusters (max_k={max_k})...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        
        # Find elbow point (simplified)
        elbow_k = k_range[np.argmax(np.diff(np.diff(inertias))) + 2] if len(inertias) > 2 else 3
        
        # Find best silhouette score
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"   Elbow method suggests: {elbow_k} clusters")
        print(f"   Silhouette method suggests: {silhouette_k} clusters")
        
        return elbow_k, silhouette_k, inertias, silhouette_scores, k_range
    
    def perform_clustering(self):
        """
        Step 6: Perform K-means clustering - separate for males/females and combined.
        """
        print("\n🎯 Performing K-means clustering...")
        
        # Combined clustering (all athletes)
        print("   📊 Combined clustering (all athletes)...")
        elbow_k, sil_k, _, _, _ = self.find_optimal_clusters(self.features_scaled)
        optimal_k = sil_k  # Use silhouette score for final choice
        
        kmeans_combined = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.df_processed['cluster_combined'] = kmeans_combined.fit_predict(self.features_scaled)
        
        self.clusters['combined'] = {
            'model': kmeans_combined,
            'n_clusters': optimal_k,
            'silhouette': silhouette_score(self.features_scaled, kmeans_combined.labels_)
        }
        
        print(f"   ✅ Combined: {optimal_k} clusters, silhouette={self.clusters['combined']['silhouette']:.3f}")
        
        # Gender-specific clustering
        for gender in ['M', 'F']:
            print(f"   👤 {gender} clustering...")
            gender_mask = self.df_processed['Sex'] == gender
            gender_data = self.features_scaled[gender_mask]
            
            if len(gender_data) > 10:  # Minimum data for clustering
                elbow_k, sil_k, _, _, _ = self.find_optimal_clusters(gender_data, max_k=8)
                optimal_k = min(sil_k, len(gender_data) // 3)  # Reasonable cluster size
                
                kmeans_gender = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                gender_clusters = kmeans_gender.fit_predict(gender_data)
                
                self.df_processed.loc[gender_mask, f'cluster_{gender}'] = gender_clusters
                
                self.clusters[gender] = {
                    'model': kmeans_gender,
                    'n_clusters': optimal_k,
                    'silhouette': silhouette_score(gender_data, gender_clusters)
                }
                
                print(f"   ✅ {gender}: {optimal_k} clusters, silhouette={self.clusters[gender]['silhouette']:.3f}")
        
        return self.clusters
    
    def visualize_clusters(self):
        """
        Step 7: Create comprehensive visualizations showing all three clustering approaches.
        """
        print("\n📈 Creating comprehensive cluster visualizations...")
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(self.features_scaled)
        
        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'PC1': features_pca[:, 0],
            'PC2': features_pca[:, 1],
            'sport': self.df_processed['sport'],
            'sex': self.df_processed['Sex'],
            'cluster_combined': self.df_processed['cluster_combined']
        })
        
        # Add gender-specific clusters if they exist
        if 'cluster_M' in self.df_processed.columns:
            viz_df['cluster_M'] = self.df_processed['cluster_M']
        if 'cluster_F' in self.df_processed.columns:
            viz_df['cluster_F'] = self.df_processed['cluster_F']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Athlete Body Type Clustering: Male vs Female vs Combined Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Male clustering
        if 'cluster_M' in viz_df.columns:
            male_data = viz_df[viz_df['sex'] == 'M']
            scatter1 = axes[0, 0].scatter(male_data['PC1'], male_data['PC2'], 
                                         c=male_data['cluster_M'], 
                                         cmap='Blues', alpha=0.8, s=60, edgecolors='navy', linewidth=0.5)
            axes[0, 0].set_title(f'Male Athletes Only\n({len(male_data)} athletes, {self.clusters["M"]["n_clusters"]} clusters)')
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter1, ax=axes[0, 0], label='Male Cluster')
        
        # Plot 2: Female clustering
        if 'cluster_F' in viz_df.columns:
            female_data = viz_df[viz_df['sex'] == 'F']
            scatter2 = axes[0, 1].scatter(female_data['PC1'], female_data['PC2'], 
                                         c=female_data['cluster_F'], 
                                         cmap='Reds', alpha=0.8, s=60, edgecolors='darkred', linewidth=0.5)
            axes[0, 1].set_title(f'Female Athletes Only\n({len(female_data)} athletes, {self.clusters["F"]["n_clusters"]} clusters)')
            axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter2, ax=axes[0, 1], label='Female Cluster')
        
        # Plot 3: Combined clustering
        scatter3 = axes[0, 2].scatter(viz_df['PC1'], viz_df['PC2'], 
                                     c=viz_df['cluster_combined'], 
                                     cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        axes[0, 2].set_title(f'Combined (All Athletes)\n({len(viz_df)} athletes, {self.clusters["combined"]["n_clusters"]} clusters)')
        axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter3, ax=axes[0, 2], label='Combined Cluster')
        
        # Plot 4: Sports distribution
        sports = viz_df['sport'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(sports)))
        for i, sport in enumerate(sports):
            sport_data = viz_df[viz_df['sport'] == sport]
            axes[1, 0].scatter(sport_data['PC1'], sport_data['PC2'], 
                              c=[colors[i]], label=sport, alpha=0.7, s=50)
        axes[1, 0].set_title('Athletes by Sport')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 5: Gender overlay
        for gender, color, marker in zip(['M', 'F'], ['blue', 'red'], ['o', '^']):
            gender_data = viz_df[viz_df['sex'] == gender]
            axes[1, 1].scatter(gender_data['PC1'], gender_data['PC2'], 
                              c=color, label=f'{gender}ale ({len(gender_data)})', 
                              alpha=0.7, s=50, marker=marker)
        axes[1, 1].set_title('Athletes by Gender')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1, 1].legend()
        
        # Plot 6: Cluster quality metrics
        cluster_info = []
        if 'M' in self.clusters:
            cluster_info.append(f"Male: {self.clusters['M']['silhouette']:.3f}")
        if 'F' in self.clusters:
            cluster_info.append(f"Female: {self.clusters['F']['silhouette']:.3f}")
        cluster_info.append(f"Combined: {self.clusters['combined']['silhouette']:.3f}")
        
        axes[1, 2].text(0.1, 0.8, "Silhouette Scores:", fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
        for i, info in enumerate(cluster_info):
            axes[1, 2].text(0.1, 0.7 - i*0.1, info, fontsize=12, transform=axes[1, 2].transAxes)
        
        axes[1, 2].text(0.1, 0.4, "Higher scores = better clustering", fontsize=10, 
                       style='italic', transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        axes[1, 2].set_title('Clustering Quality')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_clusters(self):
        """
        Step 8: Analyze cluster characteristics and show dominant sports per cluster.
        """
        print("\n📊 Analyzing cluster characteristics...")
        
        # Combined clustering analysis
        print("\n🎯 COMBINED CLUSTERING ANALYSIS:")
        cluster_centers = pd.DataFrame(
            self.clusters['combined']['model'].cluster_centers_,
            columns=self.features_scaled.columns
        )
        
        # Inverse transform to original scale
        cluster_centers_original = pd.DataFrame(
            self.scaler.inverse_transform(cluster_centers),
            columns=self.features_scaled.columns
        )
        
        print("\nCluster Centers (Original Scale):")
        print(cluster_centers_original.round(2))
        
        # Sport distribution per cluster with dominant sports
        print("\n🏃 DOMINANT SPORTS PER CLUSTER:")
        sport_cluster = pd.crosstab(self.df_processed['sport'], 
                                   self.df_processed['cluster_combined'])
        
        sport_cluster_pct = pd.crosstab(self.df_processed['sport'], 
                                       self.df_processed['cluster_combined'], 
                                       normalize='columns') * 100
        
        for cluster_id in range(self.clusters['combined']['n_clusters']):
            print(f"\n📍 CLUSTER {cluster_id}:")
            cluster_sports = sport_cluster_pct[cluster_id].sort_values(ascending=False)
            top_3_sports = cluster_sports.head(3)
            
            print(f"   Top 3 Sports:")
            for sport, pct in top_3_sports.items():
                count = sport_cluster.loc[sport, cluster_id]
                print(f"   • {sport}: {pct:.1f}% ({count} athletes)")
            
            # Body type description
            center = cluster_centers_original.iloc[cluster_id]
            print(f"   Body Type: {center['height_cm']:.0f}cm, {center['weight_kg']:.0f}kg, BMI={center['BMI']:.1f}")
        
        # Gender-specific analysis if available
        if 'M' in self.clusters:
            print(f"\n👨 MALE CLUSTERING ANALYSIS ({self.clusters['M']['n_clusters']} clusters):")
            male_data = self.df_processed[self.df_processed['Sex'] == 'M']
            male_sport_cluster = pd.crosstab(male_data['sport'], male_data['cluster_M'])
            male_sport_pct = pd.crosstab(male_data['sport'], male_data['cluster_M'], normalize='columns') * 100
            
            for cluster_id in range(self.clusters['M']['n_clusters']):
                print(f"\n📍 MALE CLUSTER {cluster_id}:")
                cluster_sports = male_sport_pct[cluster_id].sort_values(ascending=False)
                top_3_sports = cluster_sports.head(3)
                
                for sport, pct in top_3_sports.items():
                    count = male_sport_cluster.loc[sport, cluster_id] if sport in male_sport_cluster.index else 0
                    if count > 0:
                        print(f"   • {sport}: {pct:.1f}% ({count} athletes)")
        
        if 'F' in self.clusters:
            print(f"\n👩 FEMALE CLUSTERING ANALYSIS ({self.clusters['F']['n_clusters']} clusters):")
            female_data = self.df_processed[self.df_processed['Sex'] == 'F']
            female_sport_cluster = pd.crosstab(female_data['sport'], female_data['cluster_F'])
            female_sport_pct = pd.crosstab(female_data['sport'], female_data['cluster_F'], normalize='columns') * 100
            
            for cluster_id in range(self.clusters['F']['n_clusters']):
                print(f"\n📍 FEMALE CLUSTER {cluster_id}:")
                cluster_sports = female_sport_pct[cluster_id].sort_values(ascending=False)
                top_3_sports = cluster_sports.head(3)
                
                for sport, pct in top_3_sports.items():
                    count = female_sport_cluster.loc[sport, cluster_id] if sport in female_sport_cluster.index else 0
                    if count > 0:
                        print(f"   • {sport}: {pct:.1f}% ({count} athletes)")
        
        return cluster_centers_original, sport_cluster_pct
    
    def generate_athlete_table(self):
        """
        Step 9: Generate table showing cluster membership per athlete.
        """
        print("\n📋 Generating athlete cluster membership table...")
        
        output_cols = ['Player', 'sport', 'Sex', 'height_cm', 'weight_kg', 'BMI', 'cluster_combined']
        
        # Add gender-specific clusters if they exist
        if 'cluster_M' in self.df_processed.columns:
            output_cols.append('cluster_M')
        if 'cluster_F' in self.df_processed.columns:
            output_cols.append('cluster_F')
        
        athlete_table = self.df_processed[output_cols].copy()
        athlete_table = athlete_table.round(2)
        
        print(f"✅ Generated table with {len(athlete_table)} athletes")
        print("\nSample of athlete cluster assignments:")
        print(athlete_table.head(10))
        
        return athlete_table
    
    def run_complete_analysis(self):
        """
        Run the complete clustering analysis pipeline.
        """
        print("🚀 Starting Complete Athlete Body Type Clustering Analysis")
        print("=" * 60)
        
        # Execute all steps
        self.load_data()
        self.handle_missing_values()
        self.create_body_ratios()
        self.prepare_features()
        self.perform_clustering()
        self.visualize_clusters()
        cluster_centers, sport_dist, gender_dist = self.analyze_clusters()
        athlete_table = self.generate_athlete_table()
        
        print("\n" + "=" * 60)
        print("✅ Analysis Complete! Key Results:")
        print(f"   • Total athletes analyzed: {len(self.df_processed)}")
        print(f"   • Optimal clusters (combined): {self.clusters['combined']['n_clusters']}")
        print(f"   • Combined clustering silhouette score: {self.clusters['combined']['silhouette']:.3f}")
        
        if 'M' in self.clusters:
            print(f"   • Male clusters: {self.clusters['M']['n_clusters']} (silhouette: {self.clusters['M']['silhouette']:.3f})")
        if 'F' in self.clusters:
            print(f"   • Female clusters: {self.clusters['F']['n_clusters']} (silhouette: {self.clusters['F']['silhouette']:.3f})")
        
        return {
            'processed_data': self.df_processed,
            'features': self.features_scaled,
            'clusters': self.clusters,
            'cluster_centers': cluster_centers,
            'sport_distribution': sport_dist,
            'gender_distribution': gender_dist,
            'athlete_table': athlete_table
        }


# Example usage for Jupyter Notebook
def main():
    """
    Main function to run the analysis - perfect for Jupyter Notebook cells.
    """
    # Initialize analysis
    data_path = "athlete_dataset_pipeline/athlete_dataset_merged.csv"
    analysis = AthleteClusteringAnalysis(data_path)
    
    # Run complete analysis
    results = analysis.run_complete_analysis()
    
    # Save results
    results['athlete_table'].to_csv('athlete_clusters.csv', index=False)
    print("\n💾 Results saved to 'athlete_clusters.csv'")
    
    return analysis, results


if __name__ == "__main__":
    analysis, results = main()
