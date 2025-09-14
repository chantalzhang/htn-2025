"""
Hyperparameter Tuning for Athlete Body Type Clustering

This script systematically tests different clustering algorithms, parameters,
and preprocessing techniques to optimize clustering quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ClusteringHyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for athlete clustering.
    """
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize the tuner.
        
        Args:
            data_path: Path to athlete dataset CSV
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Load and clean data
        self.df = None
        self.X = None
        self.feature_names = None
        self.results = []
        
    def load_and_prepare_data(self):
        """Load and prepare data for clustering experiments."""
        print("📊 Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Clean invalid sports
        invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
        self.df = self.df[~self.df['sport'].isin(invalid_sports)]
        
        print(f"✅ Loaded {len(self.df)} athletes across {self.df['sport'].nunique()} sports")
        
        # Define all possible features for clustering
        potential_features = [
            'height_cm', 'weight_kg', 'bmi',
            'Arm Span', 'Leg Length', 'Torso Length',
            'Spike Reach', 'Block Reach'
        ]
        
        # Select available features
        available_features = [f for f in potential_features if f in self.df.columns]
        print(f"🎯 Available features: {available_features}")
        
        # Create feature matrix - only select numeric columns
        numeric_features = []
        for col in available_features:
            if col in self.df.columns:
                # Try to convert to numeric
                numeric_col = pd.to_numeric(self.df[col], errors='coerce')
                if not numeric_col.isna().all():  # If at least some values are numeric
                    numeric_features.append(col)
        
        print(f"🔢 Numeric features: {numeric_features}")
        
        if not numeric_features:
            raise ValueError("No numeric features found for clustering")
        
        self.X = self.df[numeric_features].copy()
        self.feature_names = numeric_features
        
        # Convert all columns to numeric and handle missing values
        for col in self.X.columns:
            self.X[col] = pd.to_numeric(self.X[col], errors='coerce')
            if self.X[col].isnull().any():
                mean_val = self.X[col].mean()
                self.X[col].fillna(mean_val, inplace=True)
        
        print(f"📐 Feature matrix shape: {self.X.shape}")
        
        return self.X
    
    def create_engineered_features(self, X):
        """Create additional engineered features."""
        X_eng = X.copy()
        
        # BMI if not already present
        if 'height_cm' in X.columns and 'weight_kg' in X.columns and 'bmi' not in X.columns:
            X_eng['BMI'] = X['weight_kg'] / (X['height_cm'] / 100) ** 2
        
        # Body ratios if limb measurements available
        if 'Arm Span' in X.columns and 'height_cm' in X.columns:
            X_eng['arm_span_height_ratio'] = X['Arm Span'] / X['height_cm']
        
        if 'Leg Length' in X.columns and 'height_cm' in X.columns:
            X_eng['leg_height_ratio'] = X['Leg Length'] / X['height_cm']
        
        if 'Torso Length' in X.columns and 'height_cm' in X.columns:
            X_eng['torso_height_ratio'] = X['Torso Length'] / X['height_cm']
        
        if 'weight_kg' in X.columns and 'height_cm' in X.columns:
            X_eng['weight_height_ratio'] = X['weight_kg'] / X['height_cm']
        
        return X_eng
    
    def test_preprocessing_methods(self, X):
        """Test different preprocessing methods."""
        preprocessing_methods = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'No_Scaling': None
        }
        
        preprocessed_data = {}
        
        for name, scaler in preprocessing_methods.items():
            if scaler is None:
                preprocessed_data[name] = X.values
            else:
                preprocessed_data[name] = scaler.fit_transform(X)
        
        return preprocessed_data
    
    def test_clustering_algorithms(self, X_scaled, scaler_name):
        """Test different clustering algorithms with various parameters."""
        
        algorithms = {
            'KMeans': {
                'class': KMeans,
                'params': {
                    'n_clusters': [3, 4, 5, 6, 7, 8],
                    'init': ['k-means++', 'random'],
                    'n_init': [10, 20],
                    'max_iter': [300, 500]
                }
            },
            'AgglomerativeClustering': {
                'class': AgglomerativeClustering,
                'params': {
                    'n_clusters': [3, 4, 5, 6, 7, 8],
                    'linkage': ['ward', 'complete', 'average'],
                    'metric': ['euclidean']
                }
            },
            'GaussianMixture': {
                'class': GaussianMixture,
                'params': {
                    'n_components': [3, 4, 5, 6, 7, 8],
                    'covariance_type': ['full', 'tied', 'diag'],
                    'init_params': ['kmeans', 'random']
                }
            },
            'SpectralClustering': {
                'class': SpectralClustering,
                'params': {
                    'n_clusters': [3, 4, 5, 6, 7, 8],
                    'affinity': ['rbf', 'nearest_neighbors'],
                    'gamma': [0.1, 1.0, 10.0]
                }
            }
        }
        
        for algo_name, algo_info in algorithms.items():
            print(f"\n🔬 Testing {algo_name} with {scaler_name} scaling...")
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(algo_info['params'])
            
            for i, params in enumerate(param_combinations[:20]):  # Limit to 20 combinations per algorithm
                try:
                    # Create and fit model
                    if algo_name == 'GaussianMixture':
                        model = algo_info['class'](random_state=self.random_state, **params)
                        labels = model.fit_predict(X_scaled)
                    else:
                        if 'random_state' in algo_info['class']().get_params():
                            params['random_state'] = self.random_state
                        model = algo_info['class'](**params)
                        labels = model.fit_predict(X_scaled)
                    
                    # Calculate metrics
                    n_clusters = len(np.unique(labels))
                    if n_clusters > 1 and n_clusters < len(X_scaled):
                        silhouette = silhouette_score(X_scaled, labels)
                        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                        davies_bouldin = davies_bouldin_score(X_scaled, labels)
                        
                        # Calculate sport homogeneity (custom metric)
                        sport_homogeneity = self._calculate_sport_homogeneity(labels)
                        
                        # Store results
                        result = {
                            'algorithm': algo_name,
                            'scaler': scaler_name,
                            'params': params,
                            'n_clusters': n_clusters,
                            'silhouette_score': silhouette,
                            'calinski_harabasz_score': calinski_harabasz,
                            'davies_bouldin_score': davies_bouldin,
                            'sport_homogeneity': sport_homogeneity,
                            'labels': labels
                        }
                        
                        self.results.append(result)
                        
                        if i % 5 == 0:
                            print(f"   Tested {i+1} combinations...")
                
                except Exception as e:
                    continue
    
    def _generate_param_combinations(self, param_dict):
        """Generate all combinations of parameters."""
        import itertools
        
        keys = param_dict.keys()
        values = param_dict.values()
        combinations = []
        
        for combination in itertools.product(*values):
            param_combo = dict(zip(keys, combination))
            combinations.append(param_combo)
        
        return combinations
    
    def _calculate_sport_homogeneity(self, labels):
        """Calculate how well clusters group similar sports together."""
        sport_homogeneity_scores = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_sports = self.df.loc[cluster_mask, 'sport']
            
            if len(cluster_sports) > 0:
                # Calculate the proportion of the most common sport in this cluster
                most_common_sport_count = cluster_sports.value_counts().iloc[0]
                homogeneity = most_common_sport_count / len(cluster_sports)
                sport_homogeneity_scores.append(homogeneity)
        
        return np.mean(sport_homogeneity_scores) if sport_homogeneity_scores else 0
    
    def run_comprehensive_tuning(self):
        """Run comprehensive hyperparameter tuning."""
        print("🚀 Starting Comprehensive Clustering Hyperparameter Tuning")
        print("=" * 70)
        
        # Load data
        X = self.load_and_prepare_data()
        
        # Create engineered features
        X_engineered = self.create_engineered_features(X)
        print(f"🔧 Created engineered features: {X_engineered.shape[1]} total features")
        
        # Test different feature sets
        feature_sets = {
            'basic': ['height_cm', 'weight_kg'],
            'basic_plus_bmi': ['height_cm', 'weight_kg', 'BMI'],
            'all_available': list(X_engineered.columns)
        }
        
        for feature_set_name, features in feature_sets.items():
            available_features = [f for f in features if f in X_engineered.columns]
            if len(available_features) < 2:
                continue
                
            print(f"\n📊 Testing feature set: {feature_set_name} ({len(available_features)} features)")
            X_subset = X_engineered[available_features]
            
            # Test preprocessing methods
            preprocessed_data = self.test_preprocessing_methods(X_subset)
            
            # Test clustering algorithms for each preprocessing method
            for scaler_name, X_scaled in preprocessed_data.items():
                self.test_clustering_algorithms(X_scaled, f"{feature_set_name}_{scaler_name}")
        
        print(f"\n✅ Completed tuning! Tested {len(self.results)} configurations")
        
        return self.results
    
    def analyze_results(self):
        """Analyze and rank the results."""
        if not self.results:
            print("❌ No results to analyze. Run tuning first.")
            return
        
        print("\n📊 Analyzing Results...")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Create composite score (higher is better)
        # Normalize metrics to 0-1 scale
        results_df['silhouette_norm'] = (results_df['silhouette_score'] - results_df['silhouette_score'].min()) / (results_df['silhouette_score'].max() - results_df['silhouette_score'].min())
        results_df['calinski_norm'] = (results_df['calinski_harabasz_score'] - results_df['calinski_harabasz_score'].min()) / (results_df['calinski_harabasz_score'].max() - results_df['calinski_harabasz_score'].min())
        results_df['davies_norm'] = 1 - ((results_df['davies_bouldin_score'] - results_df['davies_bouldin_score'].min()) / (results_df['davies_bouldin_score'].max() - results_df['davies_bouldin_score'].min()))  # Invert because lower is better
        results_df['sport_homogeneity_norm'] = results_df['sport_homogeneity']
        
        # Composite score with weights
        results_df['composite_score'] = (
            0.3 * results_df['silhouette_norm'] +
            0.2 * results_df['calinski_norm'] +
            0.2 * results_df['davies_norm'] +
            0.3 * results_df['sport_homogeneity_norm']
        )
        
        # Sort by composite score
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        # Display top 10 results
        print("\n🏆 TOP 10 CLUSTERING CONFIGURATIONS:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(results_df.head(10).iterrows()):
            print(f"\n#{i+1} - Composite Score: {row['composite_score']:.3f}")
            print(f"   Algorithm: {row['algorithm']}")
            print(f"   Scaler: {row['scaler']}")
            print(f"   Clusters: {row['n_clusters']}")
            print(f"   Silhouette: {row['silhouette_score']:.3f}")
            print(f"   Sport Homogeneity: {row['sport_homogeneity']:.3f}")
            print(f"   Parameters: {row['params']}")
        
        # Create visualization (skip for now to avoid GUI issues)
        # self._create_results_visualization(results_df)
        
        return results_df
    
    def _create_results_visualization(self, results_df):
        """Create visualizations of the tuning results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clustering Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        # 1. Composite score by algorithm
        algo_scores = results_df.groupby('algorithm')['composite_score'].agg(['mean', 'max', 'std'])
        algo_scores.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Composite Score by Algorithm')
        axes[0, 0].set_ylabel('Composite Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Silhouette score distribution
        axes[0, 1].hist(results_df['silhouette_score'], bins=20, alpha=0.7)
        axes[0, 1].set_title('Silhouette Score Distribution')
        axes[0, 1].set_xlabel('Silhouette Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Sport homogeneity vs silhouette score
        scatter = axes[0, 2].scatter(results_df['silhouette_score'], results_df['sport_homogeneity'], 
                                   c=results_df['composite_score'], cmap='viridis', alpha=0.6)
        axes[0, 2].set_title('Sport Homogeneity vs Silhouette Score')
        axes[0, 2].set_xlabel('Silhouette Score')
        axes[0, 2].set_ylabel('Sport Homogeneity')
        plt.colorbar(scatter, ax=axes[0, 2], label='Composite Score')
        
        # 4. Number of clusters vs performance
        cluster_perf = results_df.groupby('n_clusters')['composite_score'].mean()
        cluster_perf.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Performance by Number of Clusters')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Average Composite Score')
        
        # 5. Scaler comparison
        scaler_perf = results_df.groupby('scaler')['composite_score'].mean().sort_values(ascending=False)
        scaler_perf.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Performance by Scaler')
        axes[1, 1].set_ylabel('Average Composite Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Top configurations
        top_10 = results_df.head(10)
        axes[1, 2].barh(range(len(top_10)), top_10['composite_score'])
        axes[1, 2].set_yticks(range(len(top_10)))
        axes[1, 2].set_yticklabels([f"{row['algorithm'][:6]}_{row['n_clusters']}c" for _, row in top_10.iterrows()])
        axes[1, 2].set_title('Top 10 Configurations')
        axes[1, 2].set_xlabel('Composite Score')
        
        plt.tight_layout()
        plt.show()
    
    def apply_best_configuration(self):
        """Apply the best configuration found during tuning."""
        if not self.results:
            print("❌ No results available. Run tuning first.")
            return None
        
        # Get best result
        results_df = pd.DataFrame(self.results)
        
        # Calculate composite scores
        results_df['silhouette_norm'] = (results_df['silhouette_score'] - results_df['silhouette_score'].min()) / (results_df['silhouette_score'].max() - results_df['silhouette_score'].min())
        results_df['sport_homogeneity_norm'] = results_df['sport_homogeneity']
        results_df['composite_score'] = 0.5 * results_df['silhouette_norm'] + 0.5 * results_df['sport_homogeneity_norm']
        
        best_result = results_df.loc[results_df['composite_score'].idxmax()]
        
        print(f"\n🎯 Applying Best Configuration:")
        print(f"   Algorithm: {best_result['algorithm']}")
        print(f"   Parameters: {best_result['params']}")
        print(f"   Clusters: {best_result['n_clusters']}")
        print(f"   Silhouette Score: {best_result['silhouette_score']:.3f}")
        print(f"   Sport Homogeneity: {best_result['sport_homogeneity']:.3f}")
        
        # Analyze cluster composition
        labels = best_result['labels']
        
        print(f"\n📊 Cluster Analysis:")
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_sports = self.df.loc[cluster_mask, 'sport'].value_counts()
            cluster_size = cluster_mask.sum()
            
            print(f"\n   Cluster {cluster_id} ({cluster_size} athletes):")
            for sport, count in cluster_sports.head(3).items():
                pct = count / cluster_size * 100
                print(f"     • {sport}: {count} athletes ({pct:.1f}%)")
        
        return best_result


def main():
    """Main execution function."""
    # Initialize tuner
    tuner = ClusteringHyperparameterTuner("athlete_dataset_pipeline/athlete_dataset_merged.csv")
    
    # Run comprehensive tuning
    results = tuner.run_comprehensive_tuning()
    
    # Analyze results
    results_df = tuner.analyze_results()
    
    # Apply best configuration
    best_config = tuner.apply_best_configuration()
    
    return tuner, results_df, best_config


if __name__ == "__main__":
    tuner, results_df, best_config = main()
