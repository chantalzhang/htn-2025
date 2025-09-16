"""
Athlete Body Type Clustering Analysis

This script performs advanced clustering of athletes based on body measurements
to identify distinct body type archetypes and their associated sports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class AthleteBodyTypeClusterer:
    """
    Advanced clustering of athletes based on body measurements.
    Handles missing data with normal distribution sampling and creates
    ratio-based features for better body type differentiation.
    """
    
    def __init__(self, data_path, n_clusters=5, random_state=42):
        """Initialize the clusterer with data path and clustering parameters."""
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.df = None
        self.X = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        
    def load_data(self):
        """Load and clean the athlete dataset."""
        print("📊 Loading athlete dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean data
        invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
        self.df = self.df[~self.df['sport'].isin(invalid_sports)]
        
        # Standardize gender column
        self.df['Sex'] = self.df['Sex'].str.upper().map({'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'})
        
        print(f"✅ Loaded {len(self.df)} athletes")
        print(f"   Sports: {sorted(self.df['sport'].unique())}")
        print(f"   Gender distribution: {dict(self.df['Sex'].value_counts())}")
        
        return self.df
    
    def impute_missing_values(self, df, group_cols=None):
        """
        Impute missing values by sampling from a normal distribution
        centered at the group mean with group standard deviation.
        """
        df_imputed = df.copy()
        
        # If no grouping specified, use all data
        if group_cols is None:
            group_cols = []
        
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if group_cols:
                    # Calculate group statistics
                    group_means = df.groupby(group_cols)[col].mean()
                    group_stds = df.groupby(group_cols)[col].std().fillna(df[col].std())
                    
                    # Sample from normal distribution for each group
                    for group, group_data in df.groupby(group_cols):
                        mask = (df[col].isnull()) & (df[group_cols] == group).all(axis=1)
                        if mask.any():
                            group_key = group if isinstance(group, tuple) else (group,)
                            mu = group_means.get(group_key, df[col].mean())
                            sigma = group_stds.get(group_key, df[col].std())
                            
                            # Ensure sigma is not too small
                            sigma = max(sigma, df[col].std() * 0.1)
                            
                            # Sample from normal distribution
                            samples = np.random.normal(mu, sigma, size=mask.sum())
                            df_imputed.loc[mask, col] = np.clip(samples, 
                                                              mu - 2*sigma, 
                                                              mu + 2*sigma)
                else:
                    # Global imputation if no groups
                    mu = df[col].mean()
                    sigma = df[col].std()
                    mask = df[col].isnull()
                    samples = np.random.normal(mu, sigma, size=mask.sum())
                    df_imputed.loc[mask, col] = np.clip(samples, 
                                                      mu - 2*sigma, 
                                                      mu + 2*sigma)
        
        return df_imputed
    
    def create_body_ratio_features(self, df):
        """Create ratio-based features for better body type differentiation."""
        df_features = df.copy()
        
        # Basic ratios
        if all(col in df.columns for col in ['height_cm', 'weight_kg']):
            df_features['height_m'] = df['height_cm'] / 100
            df_features['bmi'] = df['weight_kg'] / (df_features['height_m'] ** 2)
            df_features['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
        
        # Limb ratios (if available)
        if 'Arm Span' in df.columns and 'height_cm' in df.columns:
            df_features['arm_span_ratio'] = df['Arm Span'] / df['height_cm']
        
        if 'Leg Length' in df.columns and 'height_cm' in df.columns:
            df_features['leg_length_ratio'] = df['Leg Length'] / df['height_cm']
        
        if 'Torso Length' in df.columns and 'height_cm' in df.columns:
            df_features['torso_length_ratio'] = df['Torso Length'] / df['height_cm']
        
        # Upper/lower body ratios
        if all(col in df.columns for col in ['Arm Span', 'Leg Length']):
            df_features['upper_lower_ratio'] = df['Arm Span'] / df['Leg Length']
        
        # Hand size ratios (if available)
        if 'Hand Length' in df.columns and 'height_cm' in df.columns:
            df_features['hand_length_ratio'] = df['Hand Length'] / df['height_cm']
        
        if all(col in df.columns for col in ['Hand Length', 'Hand Width']):
            df_features['hand_shape_ratio'] = df['Hand Length'] / df['Hand Width']
        
        # Remove temporary columns
        df_features = df_features.drop(columns=['height_m'], errors='ignore')
        
        return df_features
    
    def preprocess_data(self, gender=None):
        """Preprocess data for clustering, optionally filtered by gender."""
        print("\n🔄 Preprocessing data...")
        
        # Filter by gender if specified
        if gender is not None:
            df = self.df[self.df['Sex'] == gender].copy()
            print(f"   Filtering to {gender} athletes only")
        else:
            df = self.df.copy()
        
        # Select relevant columns
        measurement_cols = [
            'height_cm', 'weight_kg', 'bmi', 'Arm Span', 'Leg Length', 
            'Torso Length', 'Hand Length', 'Hand Width', 'Spike Reach', 'Block Reach'
        ]
        
        # Only keep columns that exist in the dataframe
        available_cols = [col for col in measurement_cols if col in df.columns]
        
        # Impute missing values by sport and position
        print("   Imputing missing values...")
        df_imputed = self.impute_missing_values(
            df[['sport', 'position'] + available_cols],
            group_cols=['sport', 'position']
        )
        
        # Create ratio-based features
        print("   Creating body ratio features...")
        df_features = self.create_body_ratio_features(df_imputed)
        
        # Select final features for clustering
        feature_cols = [
            'height_cm', 'weight_kg', 'bmi', 'weight_height_ratio',
            'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio',
            'upper_lower_ratio', 'hand_length_ratio', 'hand_shape_ratio'
        ]
        
        # Only keep features that exist
        feature_cols = [col for col in feature_cols if col in df_features.columns]
        
        # Store the feature matrix
        self.X = df_features[feature_cols]
        self.feature_names = feature_cols
        self.df_processed = df_features
        
        print(f"✅ Preprocessing complete. Using {len(feature_cols)} features:")
        print(f"   {', '.join(feature_cols)}\n")
        
        return self.X
    
    def find_optimal_clusters(self, max_clusters=8):
        """Find the optimal number of clusters using the elbow method and silhouette score."""
        print("🔍 Finding optimal number of clusters...")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Calculate metrics for different numbers of clusters
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:  # Silhouette requires >1 cluster
                silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            else:
                silhouette_scores.append(0)
                
            calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
        
        # Find the elbow point (using the kneed package if available, otherwise estimate)
        try:
            from kneed import KneeLocator
            kn = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
            optimal_k = kn.elbow
        except ImportError:
            # Simple heuristic if kneed is not available
            diffs = np.diff(inertias)
            optimal_k = np.argmin(diffs / diffs[0] > 0.1) + 2  # First point where decrease is <10% of initial
        
        # Also consider silhouette score
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2  # +2 because we started from k=2
        
        # Take the minimum of the two to avoid over-segmentation
        self.n_clusters = min(optimal_k, optimal_k_silhouette)
        
        # Plot the elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Elbow plot
        ax1.plot(range(2, max_clusters + 1), inertias, 'bo-')
        ax1.axvline(x=self.n_clusters, color='r', linestyle='--', label=f'Optimal k={self.n_clusters}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score plot
        ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'go-')
        ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--', 
                   label=f'Best silhouette k={optimal_k_silhouette}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs. Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        print(f"💡 Optimal number of clusters: {self.n_clusters}")
        
        return self.n_clusters
    
    def perform_clustering(self):
        """Perform K-means clustering with the optimal number of clusters."""
        print(f"\n🔍 Performing K-means clustering with k={self.n_clusters}...")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                            random_state=self.random_state,
                            n_init=10)
        self.labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the processed dataframe
        self.df_processed['cluster'] = self.labels
        
        # Calculate metrics
        self.silhouette = silhouette_score(X_scaled, self.labels)
        self.calinski = calinski_harabasz_score(X_scaled, self.labels)
        self.davies = davies_bouldin_score(X_scaled, self.labels)
        
        print(f"✅ Clustering complete")
        print(f"   Silhouette score: {self.silhouette:.3f}")
        print(f"   Calinski-Harabasz score: {self.calinski:.1f}")
        print(f"   Davies-Bouldin score: {self.davies:.3f}")
        
        return self.labels
    
    def analyze_clusters(self):
        """Analyze and describe the clusters."""
        print("\n📊 Analyzing clusters...")
        
        # Calculate cluster sizes
        cluster_sizes = self.df_processed['cluster'].value_counts().sort_index()
        
        # Calculate mean values for each feature by cluster
        cluster_means = self.df_processed.groupby('cluster')[self.feature_names].mean()
        
        # Calculate sport distribution by cluster
        sport_dist = pd.crosstab(self.df_processed['sport'], self.df_processed['cluster'], 
                                normalize='index')
        
        # Get top sports per cluster
        top_sports = {}
        for cluster in range(self.n_clusters):
            sports = sport_dist[cluster].sort_values(ascending=False).head(3)
            top_sports[cluster] = sports
        
        # Print cluster summaries
        print("\n🏆 CLUSTER SUMMARIES")
        print("=" * 80)
        
        for cluster in range(self.n_clusters):
            size = cluster_sizes[cluster]
            pct = (size / len(self.df_processed)) * 100
            
            print(f"\n📊 CLUSTER {cluster} - {size} athletes ({pct:.1f}%)")
            print("-" * 40)
            
            # Body type characteristics
            print("📏 BODY TYPE:")
            cluster_data = cluster_means.loc[cluster]
            
            if 'height_cm' in cluster_data:
                print(f"   • Height: {cluster_data['height_cm']:.1f} cm")
            if 'weight_kg' in cluster_data:
                print(f"   • Weight: {cluster_data['weight_kg']:.1f} kg")
            if 'bmi' in cluster_data:
                print(f"   • BMI: {cluster_data['bmi']:.1f}")
            if 'arm_span_ratio' in cluster_data:
                print(f"   • Arm span/height: {cluster_data['arm_span_ratio']:.3f}")
            if 'leg_length_ratio' in cluster_data:
                print(f"   • Leg length/height: {cluster_data['leg_length_ratio']:.3f}")
            if 'torso_length_ratio' in cluster_data:
                print(f"   • Torso length/height: {cluster_data['torso_length_ratio']:.3f}")
            
            # Top sports
            print("\n🏅 TOP SPORTS (by representation in cluster):")
            for sport, pct in top_sports[cluster].items():
                print(f"   • {sport}: {pct*100:.1f}%")
            
            # Sample athletes (if available)
            if 'Player' in self.df_processed.columns:
                sample_athletes = self.df_processed[self.df_processed['cluster'] == cluster]['Player'].dropna().sample(min(3, size), random_state=self.random_state)
                if len(sample_athletes) > 0:
                    print("\n👥 SAMPLE ATHLETES:")
                    for athlete in sample_athletes:
                        print(f"   • {athlete}")
            
            print("\n" + "-" * 40)
        
        return cluster_means, top_sports
    
    def visualize_clusters(self):
        """Create visualizations of the clustering results."""
        print("\n🎨 Creating visualizations...")
        
        # Reduce dimensions for visualization
        X_scaled = self.scaler.fit_transform(self.X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Add PCA coordinates to dataframe
        self.df_processed['pca_1'] = X_pca[:, 0]
        self.df_processed['pca_2'] = X_pca[:, 1]
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
        
        # 1. PCA scatter plot (colored by cluster)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                       data=self.df_processed, palette='viridis', 
                       alpha=0.7, ax=ax1)
        ax1.set_title('Athlete Clusters (PCA)', fontweight='bold')
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(title='Cluster')
        
        # 2. Sport distribution by cluster (heatmap)
        ax2 = fig.add_subplot(gs[0, 1])
        sport_cluster = pd.crosstab(self.df_processed['sport'], 
                                  self.df_processed['cluster'], 
                                  normalize='index')
        sns.heatmap(sport_cluster, cmap='YlOrRd', annot=True, 
                   fmt='.2f', ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Sport Distribution by Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Sport')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Cluster sizes
        ax3 = fig.add_subplot(gs[0, 2])
        cluster_sizes = self.df_processed['cluster'].value_counts().sort_index()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, 
                   palette='viridis', ax=ax3)
        ax3.set_title('Athletes per Cluster', fontweight='bold')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Athletes')
        
        # 4-6. Feature distributions by cluster (top 3 features)
        # Get feature importances from PCA
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.pca.components_[0][:len(self.feature_names)])
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance['feature'].head(3).tolist()
        
        for i, feature in enumerate(top_features, 1):
            ax = fig.add_subplot(gs[1, i-1])
            sns.boxplot(x='cluster', y=feature, data=self.df_processed, 
                       palette='viridis', ax=ax)
            ax.set_title(f'{feature} by Cluster', fontweight='bold')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(feature)
        
        # 7-9. Top sports per cluster
        for i in range(3):
            if i < self.n_clusters:
                ax = fig.add_subplot(gs[2, i])
                cluster_data = self.df_processed[self.df_processed['cluster'] == i]
                sport_counts = cluster_data['sport'].value_counts().head(5)
                
                if len(sport_counts) > 0:
                    sns.barplot(x=sport_counts.values, y=sport_counts.index, 
                               palette='viridis', ax=ax)
                    ax.set_title(f'Top Sports in Cluster {i}', fontweight='bold')
                    ax.set_xlabel('Number of Athletes')
                    ax.set_ylabel('')
            else:
                # If fewer than 3 clusters, leave the last subplot(s) empty
                ax = fig.add_subplot(gs[2, i])
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('athlete_clustering_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved as 'athlete_clustering_analysis.png'")
        
        # Also create a separate PCA plot with sport coloring
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='pca_1', y='pca_2', hue='sport', 
                       data=self.df_processed, alpha=0.7, 
                       palette='tab20')
        plt.title('Athletes Colored by Sport (PCA)', fontweight='bold')
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('athlete_sports_pca.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def run_analysis(self, gender=None):
        """Run the complete clustering analysis."""
        print(f"\n{'='*80}")
        print(f"🏃 ATHLETE BODY TYPE CLUSTERING ANALYSIS")
        if gender:
            print(f"   Gender: {gender}")
        print(f"{'='*80}\n")
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data(gender=gender)
        
        # Find optimal number of clusters
        self.find_optimal_clusters(max_clusters=6)
        
        # Perform clustering
        self.perform_clustering()
        
        # Analyze and visualize results
        self.analyze_clusters()
        self.visualize_clusters()
        
        # Save results
        output_file = f'athlete_clusters_{gender}.csv' if gender else 'athlete_clusters_all.csv'
        self.df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to '{output_file}'")
        
        return self.df_processed


def main():
    """Run the analysis for different gender groups."""
    # Initialize with the path to your dataset
    clusterer = AthleteBodyTypeClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    
    # Run for all athletes
    print("\n" + "="*60)
    print("🏆 ANALYZING ALL ATHLETES")
    print("="*60)
    clusterer.run_analysis()
    
    # Run for male athletes
    print("\n" + "="*60)
    print("🏃 ANALYZING MALE ATHLETES")
    print("="*60)
    clusterer_male = AthleteBodyTypeClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    clusterer_male.run_analysis(gender='M')
    
    # Run for female athletes
    print("\n" + "="*60)
    print("🏃 ANALYZING FEMALE ATHLETES")
    print("="*60)
    clusterer_female = AthleteBodyTypeClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    clusterer_female.run_analysis(gender='F')
    
    print("\n✅ Analysis complete! Check the generated visualizations and CSV files.")


if __name__ == "__main__":
    main()
