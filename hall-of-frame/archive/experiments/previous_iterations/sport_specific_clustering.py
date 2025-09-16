"""
Sport-Specific Athlete Body Type Clustering
==========================================

A comprehensive clustering pipeline that emphasizes sport-specific body traits
and produces meaningful body type archetypes for female, male, and combined athletes.

Features:
- Sport-specific feature weighting
- No data imputation (preserves original measurements)
- Separate clustering for Female, Male, and Combined athletes
- PCA/t-SNE visualization
- Detailed cluster analysis with dominant sports

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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

class SportSpecificClusterer:
    """
    Advanced clustering system that emphasizes sport-specific body traits
    and creates meaningful body type archetypes.
    """
    
    def __init__(self, data_path, random_state=42):
        """Initialize the clusterer with data path and parameters."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.df_processed = None
        self.scaler = RobustScaler()
        self.results = {}
        
        # Sport-specific feature weights based on ideal body types
        self.sport_weights = {
            'basketball': {
                'height_cm': 2.0, 'weight_kg': 1.2, 'bmi': 0.8,
                'arm_span_ratio': 2.5, 'leg_length_ratio': 1.8, 'torso_length_ratio': 1.0,
                'hand_length_ratio': 1.5, 'upper_lower_ratio': 1.8
            },
            'gymnastics': {
                'height_cm': 0.5, 'weight_kg': 1.5, 'bmi': 1.8,
                'arm_span_ratio': 1.2, 'leg_length_ratio': 1.0, 'torso_length_ratio': 1.5,
                'hand_length_ratio': 1.0, 'upper_lower_ratio': 1.2
            },
            'swimming': {
                'height_cm': 1.5, 'weight_kg': 0.8, 'bmi': 0.6,
                'arm_span_ratio': 2.0, 'leg_length_ratio': 1.2, 'torso_length_ratio': 2.0,
                'hand_length_ratio': 1.8, 'upper_lower_ratio': 1.5
            },
            'weightlifting': {
                'height_cm': 1.0, 'weight_kg': 2.5, 'bmi': 2.8,
                'arm_span_ratio': 1.0, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.5,
                'hand_length_ratio': 1.2, 'upper_lower_ratio': 1.0
            },
            'distance_running': {
                'height_cm': 0.8, 'weight_kg': 0.5, 'bmi': 0.4,
                'arm_span_ratio': 0.8, 'leg_length_ratio': 1.5, 'torso_length_ratio': 0.8,
                'hand_length_ratio': 0.8, 'upper_lower_ratio': 0.8
            },
            'sprint_running': {
                'height_cm': 1.2, 'weight_kg': 1.8, 'bmi': 1.5,
                'arm_span_ratio': 1.0, 'leg_length_ratio': 1.8, 'torso_length_ratio': 1.0,
                'hand_length_ratio': 1.0, 'upper_lower_ratio': 1.0
            },
            'soccer': {
                'height_cm': 1.0, 'weight_kg': 1.0, 'bmi': 1.0,
                'arm_span_ratio': 1.0, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.0,
                'hand_length_ratio': 1.0, 'upper_lower_ratio': 1.0
            },
            'tennis': {
                'height_cm': 1.2, 'weight_kg': 1.0, 'bmi': 0.9,
                'arm_span_ratio': 1.5, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.0,
                'hand_length_ratio': 1.5, 'upper_lower_ratio': 1.2
            },
            'rowing': {
                'height_cm': 1.8, 'weight_kg': 1.5, 'bmi': 1.2,
                'arm_span_ratio': 1.8, 'leg_length_ratio': 1.5, 'torso_length_ratio': 1.8,
                'hand_length_ratio': 1.2, 'upper_lower_ratio': 1.5
            },
            'volleyball': {
                'height_cm': 2.2, 'weight_kg': 1.3, 'bmi': 0.9,
                'arm_span_ratio': 2.0, 'leg_length_ratio': 1.8, 'torso_length_ratio': 1.2,
                'hand_length_ratio': 1.5, 'upper_lower_ratio': 1.8
            },
            'wrestling': {
                'height_cm': 0.8, 'weight_kg': 2.0, 'bmi': 2.2,
                'arm_span_ratio': 1.2, 'leg_length_ratio': 1.0, 'torso_length_ratio': 1.5,
                'hand_length_ratio': 1.2, 'upper_lower_ratio': 1.0
            }
        }
        
    def load_and_clean_data(self):
        """Load and clean the athlete dataset."""
        print("📊 Loading athlete dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean data - remove invalid sports
        invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
        self.df = self.df[~self.df['sport'].isin(invalid_sports)]
        
        # Standardize gender column
        self.df['Sex'] = self.df['Sex'].str.upper().map({'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'})
        
        # Remove rows with missing gender
        self.df = self.df.dropna(subset=['Sex'])
        
        print(f"✅ Loaded {len(self.df)} athletes")
        print(f"   Sports: {sorted(self.df['sport'].unique())}")
        print(f"   Gender distribution: {dict(self.df['Sex'].value_counts())}")
        
        return self.df
    
    def create_sport_weighted_features(self, df):
        """Create sport-weighted features that emphasize sport-specific body traits."""
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
            df_features['arm_span_excess'] = df['Arm Span'] - df['height_cm']
        
        if 'Leg Length' in df.columns and 'height_cm' in df.columns:
            df_features['leg_length_ratio'] = df['Leg Length'] / df['height_cm']
            df_features['leg_length_excess'] = df['Leg Length'] - (df['height_cm'] * 0.5)
        
        if 'Torso Length' in df.columns and 'height_cm' in df.columns:
            df_features['torso_length_ratio'] = df['Torso Length'] / df['height_cm']
            df_features['torso_length_excess'] = df['Torso Length'] - (df['height_cm'] * 0.3)
        
        # Upper/lower body ratios
        if all(col in df.columns for col in ['Arm Span', 'Leg Length']):
            df_features['upper_lower_ratio'] = df['Arm Span'] / df['Leg Length']
        
        # Hand size ratios (if available)
        if 'Hand Length' in df.columns and 'height_cm' in df.columns:
            df_features['hand_length_ratio'] = df['Hand Length'] / df['height_cm']
        
        if all(col in df.columns for col in ['Hand Length', 'Hand Width']):
            df_features['hand_shape_ratio'] = df['Hand Length'] / df['Hand Width']
            df_features['hand_area'] = df['Hand Length'] * df['Hand Width']
        
        # Apply sport-specific weighting
        print("   Applying sport-specific feature weighting...")
        weighted_features = []
        
        for idx, row in df_features.iterrows():
            sport = row['sport']
            if sport in self.sport_weights:
                weights = self.sport_weights[sport]
                weighted_row = {}
                
                # Apply weights to each feature
                for feature, weight in weights.items():
                    if feature in df_features.columns:
                        weighted_row[f'weighted_{feature}'] = row[feature] * weight
                    else:
                        # Use default value if feature missing
                        weighted_row[f'weighted_{feature}'] = 0
                
                weighted_features.append(weighted_row)
            else:
                # Default weights for unknown sports
                weighted_row = {}
                for feature in ['height_cm', 'weight_kg', 'bmi', 'arm_span_ratio', 
                               'leg_length_ratio', 'torso_length_ratio', 'hand_length_ratio', 'upper_lower_ratio']:
                    if feature in df_features.columns:
                        weighted_row[f'weighted_{feature}'] = row[feature]
                    else:
                        weighted_row[f'weighted_{feature}'] = 0
                weighted_features.append(weighted_row)
        
        # Add weighted features to dataframe
        weighted_df = pd.DataFrame(weighted_features, index=df_features.index)
        df_features = pd.concat([df_features, weighted_df], axis=1)
        
        # Remove temporary columns
        df_features = df_features.drop(columns=['height_m'], errors='ignore')
        
        return df_features
    
    def handle_missing_values_conservative(self, df):
        """Handle missing values conservatively - only sample if absolutely necessary."""
        df_processed = df.copy()
        
        # Convert numeric columns to proper numeric types
        measurement_cols = ['height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 
                           'Torso Length', 'Hand Length', 'Hand Width', 'Spike Reach', 'Block Reach']
        
        for col in measurement_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Only impute if more than 50% of data is missing for a feature
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_pct = df_processed[col].isnull().sum() / len(df_processed)
            if missing_pct > 0.5:
                print(f"   Warning: {col} has {missing_pct:.1%} missing values - using mean imputation")
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif missing_pct > 0:
                print(f"   Note: {col} has {missing_pct:.1%} missing values - preserving original data")
        
        return df_processed
    
    def preprocess_data(self, gender=None):
        """Preprocess data for clustering, optionally filtered by gender."""
        print(f"\n🔄 Preprocessing data{' for ' + gender if gender else ''}...")
        
        # Filter by gender if specified
        if gender is not None:
            df = self.df[self.df['Sex'] == gender].copy()
            print(f"   Filtering to {gender} athletes only: {len(df)} athletes")
        else:
            df = self.df.copy()
        
        # Handle missing values conservatively
        print("   Handling missing values conservatively...")
        df_processed = self.handle_missing_values_conservative(df)
        
        # Create sport-weighted features
        df_features = self.create_sport_weighted_features(df_processed)
        
        # Select final features for clustering
        feature_cols = [
            'height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 'height_weight_ratio',
            'arm_span_ratio', 'arm_span_excess', 'leg_length_ratio', 'leg_length_excess',
            'torso_length_ratio', 'torso_length_excess', 'upper_lower_ratio',
            'hand_length_ratio', 'hand_shape_ratio', 'hand_area'
        ]
        
        # Add weighted features
        weighted_feature_cols = [f'weighted_{col}' for col in [
            'height_cm', 'weight_kg', 'bmi', 'arm_span_ratio', 
            'leg_length_ratio', 'torso_length_ratio', 'hand_length_ratio', 'upper_lower_ratio'
        ]]
        
        # Only keep features that exist and have sufficient data
        all_feature_cols = feature_cols + weighted_feature_cols
        available_features = [col for col in all_feature_cols if col in df_features.columns]
        available_features = [col for col in available_features if df_features[col].isnull().sum() / len(df_features) < 0.8]
        
        # Store the feature matrix
        self.X = df_features[available_features].fillna(0)  # Fill remaining NaN with 0
        self.feature_names = available_features
        
        # Ensure we keep the original columns for analysis
        self.df_processed = df_features.copy()
        if 'Sex' not in self.df_processed.columns and 'Sex' in df.columns:
            self.df_processed['Sex'] = df['Sex']
        
        print(f"✅ Preprocessing complete. Using {len(available_features)} features:")
        print(f"   {', '.join(available_features[:5])}... (+{len(available_features)-5} more)")
        
        return self.X
    
    def find_optimal_clusters_sport_specific(self, gender=None):
        """Find optimal number of clusters with sport-specific considerations."""
        print(f"🔍 Finding optimal number of clusters for {'all athletes' if gender is None else gender + ' athletes'}...")
        
        # Define sensible ranges based on gender and sport diversity
        if gender == 'M':
            k_range = range(4, 8)  # 4-7 clusters for men
        elif gender == 'F':
            k_range = range(3, 6)  # 3-5 clusters for women
        else:
            k_range = range(5, 9)  # 5-8 clusters for combined
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Calculate metrics for different numbers of clusters
        results = []
        
        for k in k_range:
            if k >= len(self.X) // 3:  # Ensure reasonable cluster sizes
                break
                
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            calinski = calinski_harabasz_score(X_scaled, cluster_labels)
            
            # Calculate cluster balance
            cluster_sizes = np.bincount(cluster_labels)
            max_cluster_pct = np.max(cluster_sizes) / len(cluster_labels)
            balance_score = 1 - max_cluster_pct
            
            # Sport diversity score (penalize clusters dominated by single sport)
            sport_diversity = 0
            for cluster_id in range(k):
                cluster_data = self.df_processed.iloc[cluster_labels == cluster_id]
                if len(cluster_data) > 0:
                    sport_counts = cluster_data['sport'].value_counts()
                    if len(sport_counts) > 0:
                        max_sport_pct = sport_counts.iloc[0] / len(cluster_data)
                        sport_diversity += 1 - max_sport_pct
            sport_diversity /= k
            
            # Combined score (weighted)
            combined_score = (silhouette * 0.3 + balance_score * 0.3 + sport_diversity * 0.4)
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'calinski': calinski,
                'balance_score': balance_score,
                'sport_diversity': sport_diversity,
                'max_cluster_pct': max_cluster_pct,
                'combined_score': combined_score
            })
        
        # Find optimal k based on combined score
        if results:
            optimal_result = max(results, key=lambda x: x['combined_score'])
            self.n_clusters = optimal_result['k']
        else:
            self.n_clusters = min(k_range)
            optimal_result = {'silhouette': 0, 'balance_score': 0, 'sport_diversity': 0, 'max_cluster_pct': 1.0}
        
        print(f"💡 Optimal number of clusters: {self.n_clusters}")
        print(f"   Silhouette score: {optimal_result['silhouette']:.3f}")
        print(f"   Balance score: {optimal_result['balance_score']:.3f}")
        print(f"   Sport diversity: {optimal_result['sport_diversity']:.3f}")
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
        
        # Calculate cluster balance
        cluster_sizes = np.bincount(self.labels)
        self.max_cluster_pct = np.max(cluster_sizes) / len(self.labels)
        
        print(f"✅ Clustering complete")
        print(f"   Silhouette score: {self.silhouette:.3f}")
        print(f"   Calinski-Harabasz score: {self.calinski:.1f}")
        print(f"   Max cluster percentage: {self.max_cluster_pct:.1%}")
        
        return self.labels
    
    def analyze_clusters_detailed(self):
        """Analyze and describe the clusters in detail."""
        print("\n📊 Analyzing clusters...")
        
        # Calculate cluster sizes
        cluster_sizes = self.df_processed['cluster'].value_counts().sort_index()
        
        # Calculate mean values for each feature by cluster
        basic_features = ['height_cm', 'weight_kg', 'bmi']
        available_basic = [f for f in basic_features if f in self.df_processed.columns]
        cluster_means = self.df_processed.groupby('cluster')[available_basic].mean()
        
        # Calculate sport distribution by cluster
        sport_dist = pd.crosstab(self.df_processed['sport'], self.df_processed['cluster'], 
                                normalize='columns')
        
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
            print("-" * 50)
            
            # Body type characteristics
            print("📏 BODY TYPE CHARACTERISTICS:")
            cluster_data = cluster_means.loc[cluster]
            
            for feature in available_basic:
                print(f"   • {feature}: {cluster_data[feature]:.1f}")
            
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
        
        print(f"   {height_desc}, {build_desc}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the clustering results."""
        print("\n🎨 Creating visualizations...")
        
        # Reduce dimensions for visualization
        X_scaled = self.scaler.fit_transform(self.X)
        
        # PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(X_scaled)//4))
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Add coordinates to dataframe
        self.df_processed['pca_1'] = X_pca[:, 0]
        self.df_processed['pca_2'] = X_pca[:, 1]
        self.df_processed['tsne_1'] = X_tsne[:, 0]
        self.df_processed['tsne_2'] = X_tsne[:, 1]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🏃 SPORT-SPECIFIC ATHLETE BODY TYPE CLUSTERING', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # PCA scatter plot (colored by cluster)
        sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', 
                       data=self.df_processed, palette='viridis', 
                       alpha=0.7, s=60, ax=axes[0,0])
        axes[0,0].set_title('Clusters (PCA)', fontweight='bold')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0,0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # t-SNE scatter plot (colored by cluster)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='cluster', 
                       data=self.df_processed, palette='viridis', 
                       alpha=0.7, s=60, ax=axes[0,1])
        axes[0,1].set_title('Clusters (t-SNE)', fontweight='bold')
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
        axes[0,1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Sport distribution heatmap
        sport_cluster = pd.crosstab(self.df_processed['sport'], 
                                  self.df_processed['cluster'], 
                                  normalize='columns')
        sns.heatmap(sport_cluster, cmap='YlOrRd', annot=True, 
                   fmt='.2f', ax=axes[0,2], cbar_kws={'label': 'Proportion'})
        axes[0,2].set_title('Sport Distribution by Cluster', fontweight='bold')
        axes[0,2].set_xlabel('Cluster')
        axes[0,2].set_ylabel('Sport')
        
        # Cluster sizes
        cluster_sizes = self.df_processed['cluster'].value_counts().sort_index()
        bars = sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, 
                          palette='viridis', ax=axes[1,0])
        axes[1,0].set_title('Athletes per Cluster', fontweight='bold')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Number of Athletes')
        
        # Add percentage labels
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            pct = (height / len(self.df_processed)) * 100
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                          f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Sports colored by cluster
        sports = self.df_processed['sport'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(sports)))
        for i, sport in enumerate(sports):
            sport_data = self.df_processed[self.df_processed['sport'] == sport]
            axes[1,1].scatter(sport_data['pca_1'], sport_data['pca_2'], 
                             c=[colors[i]], label=sport, alpha=0.7, s=50)
        axes[1,1].set_title('Athletes by Sport (PCA)', fontweight='bold')
        axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Body measurements by cluster
        basic_features = ['height_cm', 'weight_kg', 'bmi']
        available_basic = [f for f in basic_features if f in self.df_processed.columns]
        cluster_means = self.df_processed.groupby('cluster')[available_basic].mean()
        
        x = np.arange(len(cluster_means))
        width = 0.25
        
        for i, feature in enumerate(available_basic):
            axes[1,2].bar(x + i*width, cluster_means[feature], width, 
                         label=feature, alpha=0.8)
        
        axes[1,2].set_title('Body Measurements by Cluster', fontweight='bold')
        axes[1,2].set_xlabel('Cluster')
        axes[1,2].set_ylabel('Value')
        axes[1,2].set_xticks(x + width)
        axes[1,2].set_xticklabels([f'{i}' for i in range(len(cluster_means))])
        axes[1,2].legend()
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self, gender=None):
        """Run the complete sport-specific clustering analysis."""
        print(f"\n{'='*80}")
        print(f"🏃 SPORT-SPECIFIC ATHLETE BODY TYPE CLUSTERING")
        if gender:
            print(f"   Gender: {gender}")
        print(f"{'='*80}\n")
        
        # Load and preprocess data
        self.load_and_clean_data()
        self.preprocess_data(gender=gender)
        
        # Find optimal number of clusters
        self.find_optimal_clusters_sport_specific(gender=gender)
        
        # Perform clustering
        self.perform_clustering()
        
        # Analyze and visualize results
        self.analyze_clusters_detailed()
        fig = self.create_visualizations()
        
        # Save results
        output_file = f'sport_specific_clusters_{gender}.csv' if gender else 'sport_specific_clusters_all.csv'
        self.df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to '{output_file}'")
        
        # Store results
        self.results = {
            'gender': gender,
            'n_clusters': self.n_clusters,
            'silhouette': self.silhouette,
            'max_cluster_pct': self.max_cluster_pct,
            'data': self.df_processed
        }
        
        return self.df_processed, fig


def main():
    """Run the complete sport-specific clustering analysis."""
    print("🚀 SPORT-SPECIFIC ATHLETE BODY TYPE CLUSTERING PIPELINE")
    print("=" * 70)
    
    # Initialize clusterer
    clusterer = SportSpecificClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    
    results = {}
    
    # Run for all athletes
    print("\n" + "="*60)
    print("🏆 ANALYZING ALL ATHLETES")
    print("="*60)
    data_all, fig_all = clusterer.run_analysis()
    results['all'] = clusterer.results
    
    # Save visualization
    plt.figure(fig_all.number)
    plt.savefig('sport_specific_clustering_all.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'sport_specific_clustering_all.png'")
    plt.show()
    
    # Run for male athletes
    print("\n" + "="*60)
    print("🏃 ANALYZING MALE ATHLETES")
    print("="*60)
    clusterer_male = SportSpecificClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    data_male, fig_male = clusterer_male.run_analysis(gender='M')
    results['male'] = clusterer_male.results
    
    # Save visualization
    plt.figure(fig_male.number)
    plt.savefig('sport_specific_clustering_male.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'sport_specific_clustering_male.png'")
    plt.show()
    
    # Run for female athletes
    print("\n" + "="*60)
    print("🏃 ANALYZING FEMALE ATHLETES")
    print("="*60)
    clusterer_female = SportSpecificClusterer(
        'athlete_dataset_pipeline/athlete_dataset_merged.csv',
        random_state=42
    )
    data_female, fig_female = clusterer_female.run_analysis(gender='F')
    results['female'] = clusterer_female.results
    
    # Save visualization
    plt.figure(fig_female.number)
    plt.savefig('sport_specific_clustering_female.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'sport_specific_clustering_female.png'")
    plt.show()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"All athletes: {results['all']['n_clusters']} clusters, silhouette={results['all']['silhouette']:.3f}, max_cluster={results['all']['max_cluster_pct']:.1%}")
    print(f"Male athletes: {results['male']['n_clusters']} clusters, silhouette={results['male']['silhouette']:.3f}, max_cluster={results['male']['max_cluster_pct']:.1%}")
    print(f"Female athletes: {results['female']['n_clusters']} clusters, silhouette={results['female']['silhouette']:.3f}, max_cluster={results['female']['max_cluster_pct']:.1%}")
    
    print("\n📁 Files generated:")
    print("   • sport_specific_clusters_all.csv")
    print("   • sport_specific_clusters_male.csv")
    print("   • sport_specific_clusters_female.csv")
    print("   • sport_specific_clustering_all.png")
    print("   • sport_specific_clustering_male.png")
    print("   • sport_specific_clustering_female.png")
    
    print("\n✅ Sport-specific clustering analysis complete!")
    
    return results


if __name__ == "__main__":
    results = main()
