"""
Improved Athlete Body Type Clustering Analysis
==============================================

This script addresses the issue of having ~99% of athletes in one "average" cluster
by implementing:

1. Better body ratio features for differentiation
2. Proper missing data handling with normal distribution sampling
3. Separate clustering for male/female athletes
4. Optimized cluster count to avoid single dominant cluster
5. Focus on creating distinct body-type archetypes with 2-3 dominant sports per cluster

Author: AI Assistant
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

class ImprovedAthleteClusterer:
    """
    Improved clustering of athletes based on body measurements.
    Focuses on creating meaningful body-type archetypes with distinct sports preferences.
    """
    
    def __init__(self, data_path, random_state=42):
        """Initialize the clusterer with data path and parameters."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.df_processed = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = PCA(n_components=2, random_state=random_state)
        self.clusters = {}
        
    def load_and_clean_data(self):
        """Load and clean the athlete dataset."""
        print("📊 Loading and cleaning athlete dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean data - remove invalid sports
        invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
        self.df = self.df[~self.df['sport'].isin(invalid_sports)]
        
        # Standardize gender column
        self.df['Sex'] = self.df['Sex'].str.upper().map({'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'})
        
        # Remove rows with missing gender (we need this for separate clustering)
        self.df = self.df.dropna(subset=['Sex'])
        
        print(f"✅ Loaded {len(self.df)} athletes")
        print(f"   Sports: {sorted(self.df['sport'].unique())}")
        print(f"   Gender distribution: {dict(self.df['Sex'].value_counts())}")
        
        return self.df
    
    def impute_missing_values_advanced(self, df, group_cols=None):
        """
        Advanced missing value imputation using normal distribution sampling
        centered at group means with group standard deviations.
        """
        df_imputed = df.copy()
        
        # If no grouping specified, use all data
        if group_cols is None:
            group_cols = []
        
        # Convert numeric columns to proper numeric types
        measurement_cols = ['height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 
                           'Torso Length', 'Hand Length', 'Hand Width', 'Spike Reach', 'Block Reach']
        
        for col in measurement_cols:
            if col in df_imputed.columns:
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')
        
        # Only process numeric columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                print(f"   Imputing {col}...")
                
                if group_cols:
                    # Calculate group statistics
                    group_means = df_imputed.groupby(group_cols)[col].mean()
                    group_stds = df_imputed.groupby(group_cols)[col].std().fillna(df_imputed[col].std())
                    
                    # Sample from normal distribution for each group
                    for group, group_data in df_imputed.groupby(group_cols):
                        mask = (df_imputed[col].isnull()) & (df_imputed[group_cols] == group).all(axis=1)
                        if mask.any():
                            group_key = group if isinstance(group, tuple) else (group,)
                            mu = group_means.get(group_key, df_imputed[col].mean())
                            sigma = group_stds.get(group_key, df_imputed[col].std())
                            
                            # Ensure sigma is not too small (at least 10% of mean)
                            sigma = max(sigma, abs(mu) * 0.1, df_imputed[col].std() * 0.1)
                            
                            # Sample from normal distribution with bounds
                            samples = np.random.normal(mu, sigma, size=mask.sum())
                            # Clip to reasonable bounds (within 3 standard deviations)
                            samples = np.clip(samples, mu - 3*sigma, mu + 3*sigma)
                            df_imputed.loc[mask, col] = samples
                else:
                    # Global imputation if no groups
                    mu = df_imputed[col].mean()
                    sigma = df_imputed[col].std()
                    mask = df_imputed[col].isnull()
                    samples = np.random.normal(mu, sigma, size=mask.sum())
                    samples = np.clip(samples, mu - 3*sigma, mu + 3*sigma)
                    df_imputed.loc[mask, col] = samples
        
        return df_imputed
    
    def create_advanced_body_features(self, df):
        """Create advanced body ratio and normalized features for better differentiation."""
        df_features = df.copy()
        
        # Basic measurements
        if all(col in df.columns for col in ['height_cm', 'weight_kg']):
            df_features['height_m'] = df['height_cm'] / 100
            df_features['bmi'] = df['weight_kg'] / (df_features['height_m'] ** 2)
            df_features['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
            df_features['height_weight_ratio'] = df['height_cm'] / df['weight_kg']
        
        # Limb ratios (if available)
        if 'Arm Span' in df.columns and 'height_cm' in df.columns:
            df_features['arm_span_ratio'] = df['Arm Span'] / df['height_cm']
            df_features['arm_span_excess'] = df['Arm Span'] - df['height_cm']  # Absolute difference
        
        if 'Leg Length' in df.columns and 'height_cm' in df.columns:
            df_features['leg_length_ratio'] = df['Leg Length'] / df['height_cm']
            df_features['leg_length_excess'] = df['Leg Length'] - (df['height_cm'] * 0.5)  # Expected leg length
        
        if 'Torso Length' in df.columns and 'height_cm' in df.columns:
            df_features['torso_length_ratio'] = df['Torso Length'] / df['height_cm']
            df_features['torso_length_excess'] = df['Torso Length'] - (df['height_cm'] * 0.3)  # Expected torso length
        
        # Upper/lower body ratios
        if all(col in df.columns for col in ['Arm Span', 'Leg Length']):
            df_features['upper_lower_ratio'] = df['Arm Span'] / df['Leg Length']
        
        # Hand size ratios (if available)
        if 'Hand Length' in df.columns and 'height_cm' in df.columns:
            df_features['hand_length_ratio'] = df['Hand Length'] / df['height_cm']
        
        if all(col in df.columns for col in ['Hand Length', 'Hand Width']):
            df_features['hand_shape_ratio'] = df['Hand Length'] / df['Hand Width']
            df_features['hand_area'] = df['Hand Length'] * df['Hand Width']
        
        # Reach measurements (if available)
        if 'Spike Reach' in df.columns and 'height_cm' in df.columns:
            df_features['spike_reach_ratio'] = df['Spike Reach'] / df['height_cm']
            df_features['spike_reach_excess'] = df['Spike Reach'] - df['height_cm']
        
        if 'Block Reach' in df.columns and 'height_cm' in df.columns:
            df_features['block_reach_ratio'] = df['Block Reach'] / df['height_cm']
            df_features['block_reach_excess'] = df['Block Reach'] - df['height_cm']
        
        # Body type indicators
        if 'bmi' in df_features.columns:
            df_features['bmi_category'] = pd.cut(df_features['bmi'], 
                                               bins=[0, 18.5, 25, 30, 100], 
                                               labels=['underweight', 'normal', 'overweight', 'obese'])
            # Convert to numeric for clustering
            df_features['bmi_category_numeric'] = df_features['bmi_category'].cat.codes
        
        # Remove temporary columns
        df_features = df_features.drop(columns=['height_m'], errors='ignore')
        
        return df_features
    
    def preprocess_data(self, gender=None):
        """Preprocess data for clustering, optionally filtered by gender."""
        print(f"\n🔄 Preprocessing data{' for ' + gender if gender else ''}...")
        
        # Filter by gender if specified
        if gender is not None:
            df = self.df[self.df['Sex'] == gender].copy()
            print(f"   Filtering to {gender} athletes only: {len(df)} athletes")
        else:
            df = self.df.copy()
        
        # Select relevant columns for imputation
        measurement_cols = [
            'height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 
            'Torso Length', 'Hand Length', 'Hand Width', 'Spike Reach', 'Block Reach'
        ]
        
        # Only keep columns that exist in the dataframe
        available_cols = [col for col in measurement_cols if col in df.columns]
        
        # Impute missing values by sport and position (if available)
        print("   Imputing missing values with normal distribution sampling...")
        impute_cols = ['sport'] + available_cols
        if 'position' in df.columns:
            impute_cols.append('position')
        
        df_imputed = self.impute_missing_values_advanced(
            df[impute_cols],
            group_cols=['sport'] + (['position'] if 'position' in df.columns else [])
        )
        
        # Create advanced body features
        print("   Creating advanced body ratio features...")
        df_features = self.create_advanced_body_features(df_imputed)
        
        # Select final features for clustering
        feature_cols = [
            'height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 'height_weight_ratio',
            'arm_span_ratio', 'arm_span_excess', 'leg_length_ratio', 'leg_length_excess',
            'torso_length_ratio', 'torso_length_excess', 'upper_lower_ratio',
            'hand_length_ratio', 'hand_shape_ratio', 'hand_area',
            'spike_reach_ratio', 'spike_reach_excess', 'block_reach_ratio', 'block_reach_excess',
            'bmi_category_numeric'
        ]
        
        # Only keep features that exist and have sufficient data
        feature_cols = [col for col in feature_cols if col in df_features.columns]
        
        # Remove features with too many missing values (>50%)
        feature_cols = [col for col in feature_cols if df_features[col].isnull().sum() / len(df_features) < 0.5]
        
        # Store the feature matrix
        self.X = df_features[feature_cols].fillna(df_features[feature_cols].mean())
        self.feature_names = feature_cols
        
        # Ensure we keep the original columns for analysis
        self.df_processed = df_features.copy()
        if 'Sex' not in self.df_processed.columns and 'Sex' in df.columns:
            self.df_processed['Sex'] = df['Sex']
        
        print(f"✅ Preprocessing complete. Using {len(feature_cols)} features:")
        print(f"   {', '.join(feature_cols)}\n")
        
        return self.X
    
    def find_optimal_clusters_advanced(self, max_clusters=10):
        """Find optimal number of clusters with focus on avoiding single dominant cluster."""
        print("🔍 Finding optimal number of clusters...")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Calculate metrics for different numbers of clusters
        results = []
        
        for k in range(2, min(max_clusters + 1, len(self.X) // 3)):  # Ensure reasonable cluster sizes
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            calinski = calinski_harabasz_score(X_scaled, cluster_labels)
            davies = davies_bouldin_score(X_scaled, cluster_labels)
            
            # Calculate cluster balance (penalize single dominant cluster)
            cluster_sizes = np.bincount(cluster_labels)
            max_cluster_pct = np.max(cluster_sizes) / len(cluster_labels)
            balance_score = 1 - max_cluster_pct  # Higher is better (more balanced)
            
            # Combined score (weighted)
            combined_score = (silhouette * 0.4 + balance_score * 0.4 + (1/davies) * 0.2)
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies': davies,
                'balance_score': balance_score,
                'max_cluster_pct': max_cluster_pct,
                'combined_score': combined_score
            })
        
        # Find optimal k based on combined score
        optimal_result = max(results, key=lambda x: x['combined_score'])
        self.n_clusters = optimal_result['k']
        
        # Plot the results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        k_values = [r['k'] for r in results]
        silhouettes = [r['silhouette'] for r in results]
        balance_scores = [r['balance_score'] for r in results]
        max_cluster_pcts = [r['max_cluster_pct'] for r in results]
        combined_scores = [r['combined_score'] for r in results]
        
        # Silhouette scores
        axes[0, 0].plot(k_values, silhouettes, 'bo-')
        axes[0, 0].axvline(x=self.n_clusters, color='r', linestyle='--', label=f'Optimal k={self.n_clusters}')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score vs. Number of Clusters')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Balance scores
        axes[0, 1].plot(k_values, balance_scores, 'go-')
        axes[0, 1].axvline(x=self.n_clusters, color='r', linestyle='--', label=f'Optimal k={self.n_clusters}')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Balance Score (1 - max_cluster_pct)')
        axes[0, 1].set_title('Cluster Balance vs. Number of Clusters')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max cluster percentage
        axes[1, 0].plot(k_values, max_cluster_pcts, 'ro-')
        axes[1, 0].axvline(x=self.n_clusters, color='r', linestyle='--', label=f'Optimal k={self.n_clusters}')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Max Cluster Percentage')
        axes[1, 0].set_title('Largest Cluster Size vs. Number of Clusters')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined scores
        axes[1, 1].plot(k_values, combined_scores, 'mo-')
        axes[1, 1].axvline(x=self.n_clusters, color='r', linestyle='--', label=f'Optimal k={self.n_clusters}')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].set_title('Combined Score vs. Number of Clusters')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters_advanced.png', dpi=300, bbox_inches='tight')
        
        print(f"💡 Optimal number of clusters: {self.n_clusters}")
        print(f"   Silhouette score: {optimal_result['silhouette']:.3f}")
        print(f"   Balance score: {optimal_result['balance_score']:.3f}")
        print(f"   Max cluster percentage: {optimal_result['max_cluster_pct']:.1%}")
        
        return self.n_clusters, results
    
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
        
        # Calculate cluster balance
        cluster_sizes = np.bincount(self.labels)
        self.max_cluster_pct = np.max(cluster_sizes) / len(self.labels)
        
        print(f"✅ Clustering complete")
        print(f"   Silhouette score: {self.silhouette:.3f}")
        print(f"   Calinski-Harabasz score: {self.calinski:.1f}")
        print(f"   Davies-Bouldin score: {self.davies:.3f}")
        print(f"   Max cluster percentage: {self.max_cluster_pct:.1%}")
        
        return self.labels
    
    def analyze_clusters_detailed(self):
        """Analyze and describe the clusters in detail."""
        print("\n📊 Analyzing clusters in detail...")
        
        # Calculate cluster sizes
        cluster_sizes = self.df_processed['cluster'].value_counts().sort_index()
        
        # Calculate mean values for each feature by cluster
        cluster_means = self.df_processed.groupby('cluster')[self.feature_names].mean()
        
        # Calculate sport distribution by cluster
        sport_dist = pd.crosstab(self.df_processed['sport'], self.df_processed['cluster'], 
                                normalize='columns')
        
        # Get top sports per cluster
        top_sports = {}
        for cluster in range(self.n_clusters):
            sports = sport_dist[cluster].sort_values(ascending=False).head(3)
            top_sports[cluster] = sports
        
        # Print detailed cluster summaries
        print("\n🏆 DETAILED CLUSTER SUMMARIES")
        print("=" * 80)
        
        for cluster in range(self.n_clusters):
            size = cluster_sizes[cluster]
            pct = (size / len(self.df_processed)) * 100
            
            print(f"\n📊 CLUSTER {cluster} - {size} athletes ({pct:.1f}%)")
            print("-" * 50)
            
            # Body type characteristics
            print("📏 BODY TYPE CHARACTERISTICS:")
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
            if 'upper_lower_ratio' in cluster_data:
                print(f"   • Upper/lower body ratio: {cluster_data['upper_lower_ratio']:.3f}")
            
            # Top sports
            print("\n🏅 TOP SPORTS (by representation in cluster):")
            for sport, pct in top_sports[cluster].items():
                print(f"   • {sport}: {pct*100:.1f}%")
            
            # Body type archetype description
            print("\n🎯 BODY TYPE ARCHETYPE:")
            self._describe_body_archetype(cluster, cluster_data)
            
            print("\n" + "-" * 50)
        
        return cluster_means, top_sports
    
    def _describe_body_archetype(self, cluster, cluster_data):
        """Describe the body type archetype for a cluster."""
        height = cluster_data.get('height_cm', 0)
        weight = cluster_data.get('weight_kg', 0)
        bmi = cluster_data.get('bmi', 0)
        arm_span_ratio = cluster_data.get('arm_span_ratio', 1.0)
        leg_ratio = cluster_data.get('leg_length_ratio', 0.5)
        
        # Height category
        if height > 200:
            height_desc = "Very tall"
        elif height > 190:
            height_desc = "Tall"
        elif height > 180:
            height_desc = "Above average height"
        elif height > 170:
            height_desc = "Average height"
        else:
            height_desc = "Shorter"
        
        # BMI category
        if bmi < 18.5:
            build_desc = "lean/lightweight"
        elif bmi < 25:
            build_desc = "athletic/balanced"
        elif bmi < 30:
            build_desc = "muscular/heavy"
        else:
            build_desc = "very muscular/heavy"
        
        # Limb proportions
        if arm_span_ratio > 1.05:
            limb_desc = "long-armed"
        elif arm_span_ratio < 0.95:
            limb_desc = "shorter-armed"
        else:
            limb_desc = "proportionally-armed"
        
        if leg_ratio > 0.52:
            leg_desc = "long-legged"
        elif leg_ratio < 0.48:
            leg_desc = "shorter-legged"
        else:
            leg_desc = "proportionally-legged"
        
        print(f"   {height_desc}, {build_desc}, {limb_desc}, {leg_desc}")
    
    def visualize_clusters_advanced(self):
        """Create advanced visualizations of the clustering results."""
        print("\n🎨 Creating advanced visualizations...")
        
        # Reduce dimensions for visualization
        X_scaled = self.scaler.fit_transform(self.X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Add PCA coordinates to dataframe
        self.df_processed['pca_1'] = X_pca[:, 0]
        self.df_processed['pca_2'] = X_pca[:, 1]
        
        # Set up the figure
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
        
        # 1. PCA scatter plot (colored by cluster)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                       data=self.df_processed, palette='viridis', 
                       alpha=0.7, s=60, ax=ax1)
        ax1.set_title('Athlete Clusters (PCA)', fontweight='bold', fontsize=12)
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Sport distribution by cluster (heatmap)
        ax2 = fig.add_subplot(gs[0, 1])
        sport_cluster = pd.crosstab(self.df_processed['sport'], 
                                  self.df_processed['cluster'], 
                                  normalize='columns')
        sns.heatmap(sport_cluster, cmap='YlOrRd', annot=True, 
                   fmt='.2f', ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Sport Distribution by Cluster', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Sport')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Cluster sizes
        ax3 = fig.add_subplot(gs[0, 2])
        cluster_sizes = self.df_processed['cluster'].value_counts().sort_index()
        bars = sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, 
                          palette='viridis', ax=ax3)
        ax3.set_title('Athletes per Cluster', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Athletes')
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            pct = (height / len(self.df_processed)) * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Gender distribution by cluster
        ax4 = fig.add_subplot(gs[0, 3])
        gender_cluster = pd.crosstab(self.df_processed['Sex'], 
                                   self.df_processed['cluster'], 
                                   normalize='columns')
        gender_cluster.plot(kind='bar', stacked=True, ax=ax4, 
                           color=['lightblue', 'lightcoral'])
        ax4.set_title('Gender Distribution by Cluster', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Proportion')
        ax4.legend(['Male', 'Female'])
        ax4.tick_params(axis='x', rotation=0)
        
        # 5-7. Feature distributions by cluster (top 3 features)
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
            ax.set_title(f'{feature} by Cluster', fontweight='bold', fontsize=11)
            ax.set_xlabel('Cluster')
            ax.set_ylabel(feature)
        
        # 8. Sports colored by cluster
        ax8 = fig.add_subplot(gs[1, 3])
        sports = self.df_processed['sport'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(sports)))
        for i, sport in enumerate(sports):
            sport_data = self.df_processed[self.df_processed['sport'] == sport]
            ax8.scatter(sport_data['pca_1'], sport_data['pca_2'], 
                       c=[colors[i]], label=sport, alpha=0.7, s=50)
        ax8.set_title('Athletes by Sport (PCA)', fontweight='bold', fontsize=12)
        ax8.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        ax8.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 9-12. Top sports per cluster (bar charts)
        for i in range(4):
            if i < self.n_clusters:
                ax = fig.add_subplot(gs[2, i])
                cluster_data = self.df_processed[self.df_processed['cluster'] == i]
                sport_counts = cluster_data['sport'].value_counts().head(5)
                
                if len(sport_counts) > 0:
                    sns.barplot(x=sport_counts.values, y=sport_counts.index, 
                               palette='viridis', ax=ax)
                    ax.set_title(f'Top Sports in Cluster {i}', fontweight='bold', fontsize=11)
                    ax.set_xlabel('Number of Athletes')
                    ax.set_ylabel('')
            else:
                # If fewer than 4 clusters, leave the subplot empty
                ax = fig.add_subplot(gs[2, i])
                ax.axis('off')
        
        # 13-16. Body measurement comparisons
        key_measurements = ['height_cm', 'weight_kg', 'bmi', 'arm_span_ratio']
        for i, measurement in enumerate(key_measurements):
            if measurement in self.df_processed.columns and i < 4:
                ax = fig.add_subplot(gs[3, i])
                sns.violinplot(x='cluster', y=measurement, data=self.df_processed, 
                              palette='viridis', ax=ax)
                ax.set_title(f'{measurement} Distribution', fontweight='bold', fontsize=11)
                ax.set_xlabel('Cluster')
                ax.set_ylabel(measurement)
        
        plt.tight_layout()
        plt.savefig('improved_athlete_clustering_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Advanced visualizations saved as 'improved_athlete_clustering_analysis.png'")
        
        return fig
    
    def run_analysis(self, gender=None):
        """Run the complete improved clustering analysis."""
        print(f"\n{'='*80}")
        print(f"🏃 IMPROVED ATHLETE BODY TYPE CLUSTERING ANALYSIS")
        if gender:
            print(f"   Gender: {gender}")
        print(f"{'='*80}\n")
        
        # Load and preprocess data
        self.load_and_clean_data()
        self.preprocess_data(gender=gender)
        
        # Find optimal number of clusters
        self.find_optimal_clusters_advanced(max_clusters=8)
        
        # Perform clustering
        self.perform_clustering()
        
        # Analyze and visualize results
        self.analyze_clusters_detailed()
        self.visualize_clusters_advanced()
        
        # Save results
        output_file = f'improved_athlete_clusters_{gender}.csv' if gender else 'improved_athlete_clusters_all.csv'
        self.df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to '{output_file}'")
        
        return self.df_processed


def main():
    """Run the improved analysis for different gender groups."""
    # Initialize with the path to your dataset
    clusterer = ImprovedAthleteClusterer(
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
    clusterer_male = ImprovedAthleteClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    clusterer_male.run_analysis(gender='M')
    
    # Run for female athletes
    print("\n" + "="*60)
    print("🏃 ANALYZING FEMALE ATHLETES")
    print("="*60)
    clusterer_female = ImprovedAthleteClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    clusterer_female.run_analysis(gender='F')
    
    print("\n✅ Improved analysis complete! Check the generated visualizations and CSV files.")


if __name__ == "__main__":
    main()
