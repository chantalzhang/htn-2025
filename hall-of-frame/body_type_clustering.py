"""
Comprehensive Body Type Clustering Pipeline for Athletes

This module performs K-means clustering on athlete body measurements to identify
similar body types across different sports and genders.
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
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BodyTypeClusterer:
    """
    Complete pipeline for athlete body type clustering analysis.
    """
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize the clustering pipeline.
        
        Args:
            data_path: Path to athlete dataset CSV
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Clustering features as specified
        self.clustering_features = [
            'height_cm', 'weight_kg', 'BMI', 
            'arm_span_height_ratio', 'leg_height_ratio', 
            'torso_height_ratio', 'weight_height_ratio'
        ]
        
        # Data storage
        self.df = None
        self.df_processed = None
        self.features_scaled = {}  # Will store scaled features for each group
        self.scalers = {}  # Will store scalers for each group
        self.clusters = {}  # Will store clustering results
        self.cluster_results = {}  # Will store detailed results
        
    def load_and_preprocess_data(self):
        """
        Load data and perform preprocessing with normal distribution sampling.
        """
        print("📊 Loading and preprocessing athlete dataset...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} athletes")
        
        # Clean up invalid/test sports
        print("\n🧹 Cleaning dataset...")
        original_count = len(self.df)
        
        # Remove test sports and invalid entries
        invalid_sports = ['basketball_test_missing', 'test', 'missing', 'unknown']
        self.df = self.df[~self.df['sport'].isin(invalid_sports)]
        
        cleaned_count = len(self.df)
        if cleaned_count != original_count:
            print(f"   Removed {original_count - cleaned_count} athletes with invalid sports")
        
        print(f"✅ Clean dataset: {len(self.df)} athletes")
        print(f"📋 Available columns: {list(self.df.columns)}")
        
        # Check for required columns
        missing_features = [f for f in self.clustering_features if f not in self.df.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            # Remove missing features from clustering
            self.clustering_features = [f for f in self.clustering_features if f in self.df.columns]
            print(f"🎯 Using available features: {self.clustering_features}")
        
        # Ensure sex column exists (try different variations)
        sex_columns = ['sex', 'Sex', 'gender', 'Gender']
        sex_col = None
        for col in sex_columns:
            if col in self.df.columns:
                sex_col = col
                break
        
        if sex_col is None:
            raise ValueError("No sex/gender column found in dataset")
        
        # Standardize sex column
        if sex_col != 'sex':
            self.df['sex'] = self.df[sex_col]
        
        # Clean sex values
        self.df['sex'] = self.df['sex'].str.upper()
        self.df['sex'] = self.df['sex'].map({'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'})
        
        print(f"👥 Gender distribution: {dict(self.df['sex'].value_counts())}")
        print(f"🏃 Sports: {list(self.df['sport'].unique())}")
        
        # Handle missing values using normal distribution sampling
        self._handle_missing_values()
        
        return self.df
    
    def _handle_missing_values(self):
        """
        Handle missing values by sampling from normal distributions.
        """
        print("\n🔧 Handling missing values with normal distribution sampling...")
        
        self.df_processed = self.df.copy()
        
        for feature in self.clustering_features:
            if feature in self.df_processed.columns:
                missing_count = self.df_processed[feature].isnull().sum()
                
                if missing_count > 0:
                    print(f"   📐 {feature}: {missing_count} missing values")
                    
                    # Calculate mean and std from observed values
                    observed_values = self.df_processed[feature].dropna()
                    if len(observed_values) > 0:
                        mean_val = observed_values.mean()
                        std_val = observed_values.std()
                        
                        # Sample from normal distribution
                        sampled_values = np.random.normal(mean_val, std_val, missing_count)
                        
                        # Fill missing values
                        missing_mask = self.df_processed[feature].isnull()
                        self.df_processed.loc[missing_mask, feature] = sampled_values
                        
                        print(f"      Filled with N({mean_val:.2f}, {std_val:.2f})")
        
        print("✅ Missing value handling complete")
    
    def find_optimal_k(self, X, max_k=15, group_name=""):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X: Feature matrix
            max_k: Maximum number of clusters to test
            group_name: Name of the group being clustered
            
        Returns:
            Tuple of (optimal_k_elbow, optimal_k_silhouette, metrics)
        """
        print(f"🔍 Finding optimal k for {group_name} (n={len(X)})...")
        
        if len(X) < 4:
            print(f"   ⚠️  Too few samples for clustering ({len(X)})")
            return 2, 2, {}
        
        k_range = range(2, min(max_k + 1, len(X)))
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
        
        # Find elbow point
        if len(inertias) >= 3:
            # Calculate second derivative to find elbow
            second_deriv = np.diff(np.diff(inertias))
            elbow_idx = np.argmax(second_deriv) + 2
            optimal_k_elbow = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[0]
        else:
            optimal_k_elbow = k_range[0]
        
        # Find best silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        optimal_k_silhouette = k_range[best_silhouette_idx]
        
        print(f"   📈 Elbow method suggests: k={optimal_k_elbow}")
        print(f"   📊 Silhouette method suggests: k={optimal_k_silhouette} (score: {max(silhouette_scores):.3f})")
        
        # Print all k values and their silhouette scores for analysis
        print(f"   📋 All silhouette scores:")
        for k, score in zip(k_range, silhouette_scores):
            print(f"      k={k}: {score:.3f}")
        
        metrics = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
        
        return optimal_k_elbow, optimal_k_silhouette, metrics
    
    def perform_clustering(self):
        """
        Perform K-means clustering for women, men, and combined datasets.
        """
        print("\n🎯 Performing K-means clustering...")
        
        # Prepare data groups
        groups = {
            'women': self.df_processed[self.df_processed['sex'] == 'F'],
            'men': self.df_processed[self.df_processed['sex'] == 'M'],
            'combined': self.df_processed
        }
        
        for group_name, group_data in groups.items():
            if len(group_data) < 4:
                print(f"⚠️  Skipping {group_name}: too few samples ({len(group_data)})")
                continue
                
            print(f"\n👥 Processing {group_name} group ({len(group_data)} athletes)...")
            
            # Extract and scale features
            X = group_data[self.clustering_features].copy()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store scaled features and scaler
            self.features_scaled[group_name] = pd.DataFrame(
                X_scaled, 
                columns=self.clustering_features,
                index=group_data.index
            )
            self.scalers[group_name] = scaler
            
            # Find optimal k
            k_elbow, k_silhouette, metrics = self.find_optimal_k(
                X_scaled, max_k=8, group_name=group_name
            )
            
            # For better sport separation, let's try higher k values
            # Use a compromise between silhouette score and practical interpretability
            if group_name == 'combined' and len(group_data) > 100:
                # For combined dataset, try k=5-6 for better sport separation
                optimal_k = min(6, max(k_silhouette, 5))
            elif len(group_data) > 50:
                # For gender-specific, try k=4-5
                optimal_k = min(5, max(k_silhouette, 4))
            else:
                optimal_k = k_silhouette
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate final silhouette score
            final_silhouette = silhouette_score(X_scaled, cluster_labels)
            
            # Store results
            self.clusters[group_name] = {
                'model': kmeans,
                'labels': cluster_labels,
                'n_clusters': optimal_k,
                'silhouette_score': final_silhouette,
                'optimization_metrics': metrics,
                'data_indices': group_data.index
            }
            
            print(f"   ✅ Final clustering: k={optimal_k}, silhouette={final_silhouette:.3f}")
    
    def analyze_clusters(self):
        """
        Perform post-clustering analysis including sport distribution and cluster stats.
        """
        print("\n📊 Analyzing cluster characteristics...")
        
        for group_name, cluster_info in self.clusters.items():
            print(f"\n🎯 {group_name.upper()} CLUSTERING ANALYSIS:")
            
            # Get data for this group
            data_indices = cluster_info['data_indices']
            group_data = self.df_processed.loc[data_indices].copy()
            group_data['cluster'] = cluster_info['labels']
            
            # Calculate cluster statistics
            cluster_stats = []
            sport_distributions = {}
            
            for cluster_id in range(cluster_info['n_clusters']):
                cluster_mask = group_data['cluster'] == cluster_id
                cluster_data = group_data[cluster_mask]
                
                # Basic stats
                stats = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(group_data) * 100
                }
                
                # Physical characteristics
                for feature in self.clustering_features:
                    if feature in cluster_data.columns:
                        stats[f'{feature}_mean'] = cluster_data[feature].mean()
                        stats[f'{feature}_std'] = cluster_data[feature].std()
                
                cluster_stats.append(stats)
                
                # Sport distribution
                sport_dist = cluster_data['sport'].value_counts()
                sport_dist_pct = (sport_dist / len(cluster_data) * 100).round(1)
                sport_distributions[cluster_id] = {
                    'counts': dict(sport_dist),
                    'percentages': dict(sport_dist_pct),
                    'dominant_sport': sport_dist.index[0] if len(sport_dist) > 0 else 'None'
                }
                
                # Print cluster summary
                print(f"\n📍 CLUSTER {cluster_id} ({len(cluster_data)} athletes, {stats['percentage']:.1f}%):")
                print(f"   🏃 Dominant sports:")
                for sport, count in sport_dist.head(3).items():
                    pct = sport_dist_pct[sport]
                    print(f"      • {sport}: {count} athletes ({pct:.1f}%)")
                
                if 'height_cm' in cluster_data.columns and 'weight_kg' in cluster_data.columns:
                    height_avg = cluster_data['height_cm'].mean()
                    weight_avg = cluster_data['weight_kg'].mean()
                    bmi_avg = cluster_data['BMI'].mean() if 'BMI' in cluster_data.columns else 0
                    print(f"   📏 Body type: {height_avg:.0f}cm, {weight_avg:.0f}kg, BMI={bmi_avg:.1f}")
            
            # Store analysis results
            self.cluster_results[group_name] = {
                'cluster_stats': pd.DataFrame(cluster_stats),
                'sport_distributions': sport_distributions,
                'group_data': group_data
            }
    
    def create_visualizations(self):
        """
        Generate PCA/t-SNE visualizations and sport distribution charts.
        """
        print("\n📈 Creating visualizations...")
        
        # Create main figure
        fig = plt.figure(figsize=(20, 15))
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # PCA visualizations
        for i, (group_name, cluster_info) in enumerate(self.clusters.items()):
            # PCA plot
            ax_pca = fig.add_subplot(gs[0, i])
            self._create_pca_plot(group_name, ax_pca)
            
            # Sport distribution bar chart
            ax_bar = fig.add_subplot(gs[1, i])
            self._create_sport_distribution_chart(group_name, ax_bar)
        
        # Combined comparison plots
        if len(self.clusters) > 1:
            # Silhouette score comparison
            ax_silhouette = fig.add_subplot(gs[2, 0])
            self._create_silhouette_comparison(ax_silhouette)
            
            # Cluster size comparison
            ax_sizes = fig.add_subplot(gs[2, 1])
            self._create_cluster_size_comparison(ax_sizes)
        
        plt.suptitle('Athlete Body Type Clustering Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Create detailed t-SNE visualization
        self._create_tsne_visualization()
    
    def _create_pca_plot(self, group_name, ax):
        """Create PCA plot for a specific group."""
        if group_name not in self.clusters:
            return
            
        # Get data
        X_scaled = self.features_scaled[group_name]
        cluster_info = self.clusters[group_name]
        
        # Perform PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                           c=cluster_info['labels'], 
                           cmap='tab10', alpha=0.7, s=50)
        
        # Add cluster centers
        if hasattr(cluster_info['model'], 'cluster_centers_'):
            centers_pca = pca.transform(cluster_info['model'].cluster_centers_)
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centers')
        
        ax.set_title(f'{group_name.title()} Clustering\n'
                    f'k={cluster_info["n_clusters"]}, '
                    f'silhouette={cluster_info["silhouette_score"]:.3f}')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    def _create_sport_distribution_chart(self, group_name, ax):
        """Create sport distribution chart for a specific group."""
        if group_name not in self.cluster_results:
            return
            
        sport_dist = self.cluster_results[group_name]['sport_distributions']
        
        # Prepare data for stacked bar chart
        sports = set()
        for cluster_data in sport_dist.values():
            sports.update(cluster_data['counts'].keys())
        sports = sorted(list(sports))
        
        cluster_ids = sorted(sport_dist.keys())
        
        # Create stacked bar chart
        bottom = np.zeros(len(cluster_ids))
        colors = plt.cm.Set3(np.linspace(0, 1, len(sports)))
        
        for sport, color in zip(sports, colors):
            values = []
            for cluster_id in cluster_ids:
                values.append(sport_dist[cluster_id]['counts'].get(sport, 0))
            
            ax.bar(cluster_ids, values, bottom=bottom, label=sport, color=color)
            bottom += values
        
        ax.set_title(f'{group_name.title()} - Sport Distribution by Cluster')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Athletes')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _create_silhouette_comparison(self, ax):
        """Create silhouette score comparison chart."""
        groups = list(self.clusters.keys())
        scores = [self.clusters[group]['silhouette_score'] for group in groups]
        
        bars = ax.bar(groups, scores, color=['lightblue', 'lightcoral', 'lightgreen'][:len(groups)])
        ax.set_title('Clustering Quality Comparison')
        ax.set_ylabel('Silhouette Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
    
    def _create_cluster_size_comparison(self, ax):
        """Create cluster size comparison chart."""
        data = []
        for group_name, results in self.cluster_results.items():
            for _, row in results['cluster_stats'].iterrows():
                data.append({
                    'group': group_name,
                    'cluster': f"C{row['cluster_id']}",
                    'size': row['size']
                })
        
        if data:
            df_sizes = pd.DataFrame(data)
            
            # Create grouped bar chart
            groups = df_sizes['group'].unique()
            x = np.arange(len(groups))
            width = 0.15
            
            max_clusters = df_sizes.groupby('group')['cluster'].nunique().max()
            
            for i in range(max_clusters):
                cluster_name = f"C{i}"
                sizes = []
                for group in groups:
                    group_data = df_sizes[(df_sizes['group'] == group) & (df_sizes['cluster'] == cluster_name)]
                    sizes.append(group_data['size'].iloc[0] if len(group_data) > 0 else 0)
                
                ax.bar(x + i * width, sizes, width, label=cluster_name)
            
            ax.set_title('Cluster Sizes by Group')
            ax.set_xlabel('Group')
            ax.set_ylabel('Number of Athletes')
            ax.set_xticks(x + width * (max_clusters - 1) / 2)
            ax.set_xticklabels(groups)
            ax.legend()
    
    def _create_tsne_visualization(self):
        """Create detailed t-SNE visualization."""
        print("🔮 Creating t-SNE visualization...")
        
        if 'combined' not in self.clusters:
            return
        
        # Get combined data
        X_scaled = self.features_scaled['combined']
        cluster_labels = self.clusters['combined']['labels']
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by cluster
        scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                             c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
        ax1.set_title('t-SNE: Colored by Cluster')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Plot 2: Shaped by sex, colored by cluster
        combined_data = self.cluster_results['combined']['group_data']
        
        for sex, marker in [('M', 'o'), ('F', '^')]:
            mask = combined_data['sex'] == sex
            if mask.sum() > 0:
                indices = combined_data[mask].index
                # Find positions in X_tsne that correspond to these indices
                tsne_indices = [i for i, idx in enumerate(X_scaled.index) if idx in indices]
                if tsne_indices:
                    ax2.scatter(X_tsne[tsne_indices, 0], X_tsne[tsne_indices, 1],
                              c=cluster_labels[tsne_indices], cmap='tab10', 
                              marker=marker, alpha=0.7, s=50, label=f'{sex}')
        
        ax2.set_title('t-SNE: Shaped by Sex, Colored by Cluster')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_cluster_assignment_table(self):
        """
        Generate comprehensive cluster assignment table.
        
        Returns:
            DataFrame with cluster assignments for all athletes
        """
        print("\n📋 Generating cluster assignment table...")
        
        # Start with original data
        result_df = self.df_processed.copy()
        
        # Add cluster assignments
        for group_name, cluster_info in self.clusters.items():
            cluster_col = f'cluster_{group_name}'
            result_df[cluster_col] = np.nan
            
            # Fill cluster assignments for this group
            data_indices = cluster_info['data_indices']
            result_df.loc[data_indices, cluster_col] = cluster_info['labels']
        
        # Add dominant sport per cluster
        for group_name, results in self.cluster_results.items():
            dominant_col = f'dominant_sport_{group_name}'
            result_df[dominant_col] = np.nan
            
            sport_dist = results['sport_distributions']
            cluster_col = f'cluster_{group_name}'
            
            for cluster_id, dist_info in sport_dist.items():
                mask = result_df[cluster_col] == cluster_id
                result_df.loc[mask, dominant_col] = dist_info['dominant_sport']
        
        # Select relevant columns for output
        output_cols = ['sport', 'sex'] + self.clustering_features
        
        # Add cluster columns
        cluster_cols = [col for col in result_df.columns if col.startswith('cluster_')]
        dominant_cols = [col for col in result_df.columns if col.startswith('dominant_sport_')]
        
        output_cols.extend(cluster_cols + dominant_cols)
        
        # Filter to existing columns
        available_cols = [col for col in output_cols if col in result_df.columns]
        
        final_table = result_df[available_cols].copy()
        
        print(f"✅ Generated table with {len(final_table)} athletes and {len(available_cols)} columns")
        
        return final_table
    
    def run_complete_analysis(self):
        """
        Run the complete body type clustering analysis pipeline.
        
        Returns:
            Dictionary with all results
        """
        print("🚀 Starting Complete Body Type Clustering Analysis")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Perform clustering
        self.perform_clustering()
        
        # Step 3: Analyze clusters
        self.analyze_clusters()
        
        # Step 4: Create visualizations
        self.create_visualizations()
        
        # Step 5: Generate assignment table
        assignment_table = self.generate_cluster_assignment_table()
        
        print("\n🎉 Analysis complete!")
        
        return {
            'clusters': self.clusters,
            'cluster_results': self.cluster_results,
            'assignment_table': assignment_table,
            'features_used': self.clustering_features
        }


def main():
    """Main execution function."""
    # Initialize and run analysis
    clusterer = BodyTypeClusterer("athlete_dataset_pipeline/athlete_dataset_merged.csv")
    results = clusterer.run_complete_analysis()
    
    # Save results
    results['assignment_table'].to_csv('athlete_body_type_clusters.csv', index=False)
    print("💾 Results saved to 'athlete_body_type_clusters.csv'")
    
    return results


if __name__ == "__main__":
    results = main()
