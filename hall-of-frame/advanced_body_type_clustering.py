"""
Advanced Body Type Clustering Pipeline
======================================

This pipeline addresses the clustering issues by:
1. Using feature engineering to create more discriminative body type features
2. Applying strategic scaling and normalization
3. Using multiple clustering approaches to find optimal body type separation
4. Creating meaningful clusters that group athletes by body similarity, not sport

Key innovations:
- Body type ratio features that emphasize meaningful differences
- Strategic feature scaling to prevent any single feature from dominating
- Multiple clustering validation approaches
- Body type archetype features that capture essential characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

class AdvancedBodyTypeClusterer:
    """
    Advanced clustering system that creates meaningful body type clusters
    by emphasizing body type differences without using sport labels.
    """
    
    def __init__(self, data_path, random_state=42):
        """Initialize the clusterer."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.df_processed = None
        self.results = {}
        
        # Body type feature importance weights
        # These emphasize features that distinguish body types
        self.feature_importance = {
            'height_cm': 1.0,           # Base importance
            'weight_kg': 1.0,           # Base importance  
            'bmi': 1.5,                 # BMI is crucial for body composition
            'height_weight_ratio': 2.0, # Height-weight relationship is key
            'weight_height_ratio': 1.5, # Inverse relationship
            'arm_span_ratio': 1.8,      # Limb proportions matter
            'leg_length_ratio': 1.8,    # Leg proportions matter
            'torso_length_ratio': 1.5,  # Torso proportions matter
        }
    
    def load_data(self):
        """Load athlete data."""
        print("📊 Loading athlete dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} athletes")
        print(f"   Sports: {sorted(self.df['sport'].unique())}")
        print(f"   Gender distribution: {self.df['Sex'].value_counts().to_dict()}")
        return self.df
    
    def create_advanced_body_features(self, df):
        """
        Create advanced body type features that emphasize meaningful differences.
        """
        print("🔧 Creating advanced body type features...")
        df_features = df.copy()
        
        # Basic measurements
        df_features['height_cm'] = pd.to_numeric(df_features['height_cm'], errors='coerce')
        df_features['weight_kg'] = pd.to_numeric(df_features['weight_kg'], errors='coerce')
        
        # Calculate BMI
        df_features['bmi'] = df_features['weight_kg'] / (df_features['height_cm'] / 100) ** 2
        
        # Create body type ratios that emphasize differences
        df_features['height_weight_ratio'] = df_features['height_cm'] / df_features['weight_kg']
        df_features['weight_height_ratio'] = df_features['weight_kg'] / df_features['height_cm']
        
        # Limb proportions (estimate if missing)
        if 'Arm Span' in df_features.columns:
            df_features['arm_span'] = pd.to_numeric(df_features['Arm Span'], errors='coerce')
            df_features['arm_span_ratio'] = df_features['arm_span'] / df_features['height_cm']
        else:
            df_features['arm_span_ratio'] = 1.0  # Typical ratio
        
        if 'Leg Length' in df_features.columns:
            df_features['leg_length'] = pd.to_numeric(df_features['Leg Length'], errors='coerce')
            df_features['leg_length_ratio'] = df_features['leg_length'] / df_features['height_cm']
        else:
            df_features['leg_length_ratio'] = 0.47  # Typical ratio
        
        if 'Torso Length' in df_features.columns:
            df_features['torso_length'] = pd.to_numeric(df_features['Torso Length'], errors='coerce')
            df_features['torso_length_ratio'] = df_features['torso_length'] / df_features['height_cm']
        else:
            df_features['torso_length_ratio'] = 0.32  # Typical ratio
        
        # Create body type archetype features
        # These capture essential body type characteristics
        
        # 1. Size category (normalized height)
        height_percentiles = df_features['height_cm'].quantile([0.25, 0.5, 0.75])
        df_features['size_category'] = pd.cut(df_features['height_cm'], 
                                            bins=[0, height_percentiles[0.25], 
                                                  height_percentiles[0.5], 
                                                  height_percentiles[0.75], 300],
                                            labels=[1, 2, 3, 4])
        
        # 2. Build category (BMI-based)
        bmi_percentiles = df_features['bmi'].quantile([0.25, 0.5, 0.75])
        df_features['build_category'] = pd.cut(df_features['bmi'], 
                                             bins=[0, bmi_percentiles[0.25], 
                                                   bmi_percentiles[0.5], 
                                                   bmi_percentiles[0.75], 100],
                                             labels=[1, 2, 3, 4])
        
        # 3. Proportions category (height-weight relationship)
        hw_percentiles = df_features['height_weight_ratio'].quantile([0.25, 0.5, 0.75])
        df_features['proportions_category'] = pd.cut(df_features['height_weight_ratio'], 
                                                   bins=[0, hw_percentiles[0.25], 
                                                         hw_percentiles[0.5], 
                                                         hw_percentiles[0.75], 10],
                                                   labels=[1, 2, 3, 4])
        
        # Convert categorical features to numeric
        df_features['size_category'] = pd.to_numeric(df_features['size_category'], errors='coerce')
        df_features['build_category'] = pd.to_numeric(df_features['build_category'], errors='coerce')
        df_features['proportions_category'] = pd.to_numeric(df_features['proportions_category'], errors='coerce')
        
        return df_features
    
    def apply_strategic_weighting(self, df):
        """
        Apply strategic weighting that emphasizes body type differences
        without using sport labels.
        """
        print("⚖️ Applying strategic body type weighting...")
        
        # Select features for clustering
        feature_cols = [
            'height_cm', 'weight_kg', 'bmi',
            'height_weight_ratio', 'weight_height_ratio',
            'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio',
            'size_category', 'build_category', 'proportions_category'
        ]
        
        # Create weighted features
        weighted_features = {}
        
        for feature in feature_cols:
            if feature in df.columns:
                # Get importance weight
                weight = self.feature_importance.get(feature, 1.0)
                
                # Apply weight
                weighted_features[f'weighted_{feature}'] = df[feature] * weight
        
        # Add weighted features to dataframe
        weighted_df = pd.DataFrame(weighted_features, index=df.index)
        df_weighted = pd.concat([df, weighted_df], axis=1)
        
        return df_weighted
    
    def handle_missing_values(self, df):
        """Handle missing values conservatively."""
        print("🔧 Handling missing values...")
        df_processed = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['height_cm', 'weight_kg', 'Arm Span', 'Leg Length', 
                       'Torso Length', 'Hand Length', 'Hand Width']
        
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Only impute if more than 50% missing
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            missing_pct = df_processed[col].isnull().sum() / len(df_processed)
            if missing_pct > 0.5:
                print(f"   Warning: {col} has {missing_pct:.1%} missing - using mean imputation")
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif missing_pct > 0:
                print(f"   Note: {col} has {missing_pct:.1%} missing - preserving original data")
        
        return df_processed
    
    def preprocess_data(self, gender=None):
        """Preprocess data for clustering."""
        print(f"\n🔄 Preprocessing data{' for ' + gender if gender else ''}...")
        
        # Filter by gender if specified
        if gender:
            df_filtered = self.df[self.df['Sex'] == gender].copy()
            print(f"   Filtering to {gender} athletes only: {len(df_filtered)} athletes")
        else:
            df_filtered = self.df.copy()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_filtered)
        
        # Create advanced body features
        df_features = self.create_advanced_body_features(df_processed)
        
        # Apply strategic weighting
        df_weighted = self.apply_strategic_weighting(df_features)
        
        # Select final features for clustering
        clustering_features = [
            'weighted_height_cm', 'weighted_weight_kg', 'weighted_bmi',
            'weighted_height_weight_ratio', 'weighted_weight_height_ratio',
            'weighted_arm_span_ratio', 'weighted_leg_length_ratio', 'weighted_torso_length_ratio',
            'weighted_size_category', 'weighted_build_category', 'weighted_proportions_category'
        ]
        
        # Remove rows with too many missing values
        df_final = df_weighted.dropna(subset=clustering_features, how='all')
        
        print(f"✅ Preprocessing complete. Using {len(clustering_features)} features")
        
        self.df_processed = df_final
        return df_final, clustering_features
    
    def find_optimal_clusters_advanced(self, df, feature_cols, max_k=8):
        """
        Find optimal number of clusters using multiple approaches.
        """
        print(f"🔍 Finding optimal number of clusters using advanced methods...")
        
        # Prepare data
        X = df[feature_cols].copy()
        
        # Convert to numeric and handle missing values
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(X.mean())
        if X.isnull().any().any():
            X = X.fillna(0)
        
        # Try different scaling approaches
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        best_k = 2
        best_score = -1
        best_scaler = 'standard'
        
        for scaler_name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X)
            
            # Test different k values
            k_range = range(2, min(max_k + 1, len(df) // 5))
            scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Use silhouette score as primary metric
                score = silhouette_score(X_scaled, cluster_labels)
                scores.append(score)
            
            # Find best k for this scaler
            if scores:
                best_k_scaler = k_range[np.argmax(scores)]
                best_score_scaler = max(scores)
                
                if best_score_scaler > best_score:
                    best_score = best_score_scaler
                    best_k = best_k_scaler
                    best_scaler = scaler_name
        
        print(f"💡 Optimal configuration: k={best_k}, scaler={best_scaler}, score={best_score:.3f}")
        
        return best_k, best_scaler, best_score
    
    def perform_advanced_clustering(self, df, feature_cols, k, scaler_name):
        """Perform advanced clustering with optimal configuration."""
        print(f"🔍 Performing advanced clustering with k={k}, scaler={scaler_name}...")
        
        # Prepare data
        X = df[feature_cols].copy()
        
        # Convert to numeric and handle missing values
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(X.mean())
        if X.isnull().any().any():
            X = X.fillna(0)
        
        # Apply optimal scaling
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaler = scalers[scaler_name]
        X_scaled = scaler.fit_transform(X)
        
        # Try multiple clustering algorithms
        algorithms = {
            'kmeans': KMeans(n_clusters=k, random_state=self.random_state, n_init=10),
            'agglomerative': AgglomerativeClustering(n_clusters=k)
        }
        
        best_labels = None
        best_score = -1
        best_algorithm = 'kmeans'
        
        for algo_name, algorithm in algorithms.items():
            cluster_labels = algorithm.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_labels = cluster_labels
                best_algorithm = algo_name
        
        # Calculate additional metrics
        calinski = calinski_harabasz_score(X_scaled, best_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, best_labels)
        
        print(f"✅ Clustering complete")
        print(f"   Algorithm: {best_algorithm}")
        print(f"   Silhouette score: {best_score:.3f}")
        print(f"   Calinski-Harabasz score: {calinski:.1f}")
        print(f"   Davies-Bouldin score: {davies_bouldin:.3f}")
        
        return best_labels, X_scaled, best_score, calinski, davies_bouldin
    
    def analyze_clusters_detailed(self, df, cluster_labels, feature_cols):
        """Analyze clusters with detailed body type characteristics."""
        print("\n📊 Analyzing clusters with detailed body type characteristics...")
        
        df_analysis = df.copy()
        df_analysis['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in sorted(df_analysis['cluster'].unique()):
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            
            # Basic stats
            size = len(cluster_data)
            percentage = size / len(df_analysis) * 100
            
            # Average measurements (using original, unweighted data)
            avg_height = cluster_data['height_cm'].mean()
            avg_weight = cluster_data['weight_kg'].mean()
            avg_bmi = cluster_data['bmi'].mean()
            avg_hw_ratio = cluster_data['height_weight_ratio'].mean()
            
            # Top sports (for analysis only)
            sport_counts = cluster_data['sport'].value_counts()
            top_sports = sport_counts.head(3)
            
            # Body type characteristics
            body_type_summary = self._analyze_body_type_characteristics(cluster_data)
            
            # Cluster balance score (how well distributed the cluster is)
            balance_score = min(percentage, 100 - percentage) / 50  # Higher is better
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': size,
                'percentage': percentage,
                'avg_height': avg_height,
                'avg_weight': avg_weight,
                'avg_bmi': avg_bmi,
                'avg_hw_ratio': avg_hw_ratio,
                'top_sports': top_sports,
                'body_type_summary': body_type_summary,
                'balance_score': balance_score
            })
        
        return cluster_stats
    
    def _analyze_body_type_characteristics(self, cluster_data):
        """Analyze detailed body type characteristics of a cluster."""
        avg_height = cluster_data['height_cm'].mean()
        avg_bmi = cluster_data['bmi'].mean()
        avg_hw_ratio = cluster_data['height_weight_ratio'].mean()
        avg_arm_ratio = cluster_data.get('arm_span_ratio', pd.Series([1.0])).mean()
        
        # Determine body type characteristics
        if avg_height > 185:
            height_desc = "tall"
        elif avg_height > 170:
            height_desc = "medium height"
        else:
            height_desc = "shorter"
        
        if avg_bmi > 25:
            build_desc = "muscular/heavy"
        elif avg_bmi > 22:
            build_desc = "athletic/balanced"
        else:
            build_desc = "lean"
        
        if avg_hw_ratio > 2.5:
            proportions_desc = "very lean"
        elif avg_hw_ratio > 2.2:
            proportions_desc = "lean"
        elif avg_hw_ratio > 1.8:
            proportions_desc = "balanced"
        else:
            proportions_desc = "heavy"
        
        if avg_arm_ratio > 1.05:
            limb_desc = "long limbs"
        elif avg_arm_ratio < 0.95:
            limb_desc = "shorter limbs"
        else:
            limb_desc = "proportional limbs"
        
        return f"{height_desc}, {build_desc}, {proportions_desc}, {limb_desc}"
    
    def create_advanced_visualization(self, df, cluster_labels, X_scaled, title_suffix=""):
        """Create advanced visualization with cluster details."""
        print("\n🎨 Creating advanced visualization...")
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. PCA plot
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                            cmap='tab10', alpha=0.7, s=60)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title(f'PCA Visualization{title_suffix}')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # 2. Cluster size distribution
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        ax2.bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Athletes')
        ax2.set_title('Cluster Size Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Height vs BMI scatter
        df_viz = df.copy()
        df_viz['cluster'] = cluster_labels
        for cluster_id in sorted(df_viz['cluster'].unique()):
            cluster_data = df_viz[df_viz['cluster'] == cluster_id]
            ax3.scatter(cluster_data['height_cm'], cluster_data['bmi'], 
                       label=f'Cluster {cluster_id}', alpha=0.7, s=60)
        ax3.set_xlabel('Height (cm)')
        ax3.set_ylabel('BMI')
        ax3.set_title('Height vs BMI by Cluster')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sport distribution by cluster
        sport_cluster_counts = pd.crosstab(df_viz['sport'], df_viz['cluster'])
        sport_cluster_counts.plot(kind='bar', ax=ax4, stacked=True, alpha=0.7)
        ax4.set_xlabel('Sport')
        ax4.set_ylabel('Number of Athletes')
        ax4.set_title('Sport Distribution by Cluster')
        ax4.legend(title='Cluster')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def print_detailed_summaries(self, cluster_stats, title):
        """Print detailed cluster summaries."""
        print(f"\n🏆 {title.upper()} DETAILED CLUSTER ANALYSIS")
        print("=" * 80)
        
        # Calculate overall balance
        max_cluster_pct = max(stat['percentage'] for stat in cluster_stats)
        avg_balance = np.mean([stat['balance_score'] for stat in cluster_stats])
        
        print(f"📊 Overall Analysis:")
        print(f"   • Number of clusters: {len(cluster_stats)}")
        print(f"   • Largest cluster: {max_cluster_pct:.1f}%")
        print(f"   • Average balance score: {avg_balance:.3f}")
        print(f"   • Clustering quality: {'Good' if max_cluster_pct < 70 and avg_balance > 0.3 else 'Needs improvement'}")
        
        for stat in cluster_stats:
            print(f"\n📊 CLUSTER {stat['cluster_id']} - {stat['size']} athletes ({stat['percentage']:.1f}%)")
            print("-" * 60)
            print(f"📏 BODY TYPE CHARACTERISTICS:")
            print(f"   • Height: {stat['avg_height']:.1f}cm")
            print(f"   • Weight: {stat['avg_weight']:.1f}kg")
            print(f"   • BMI: {stat['avg_bmi']:.1f}")
            print(f"   • Height/Weight Ratio: {stat['avg_hw_ratio']:.2f}")
            print(f"   • Body Type: {stat['body_type_summary']}")
            print(f"   • Balance Score: {stat['balance_score']:.3f}")
            
            print(f"\n🏅 TOP SPORTS (by representation in cluster):")
            for i, (sport, count) in enumerate(stat['top_sports'].items(), 1):
                percentage = count / stat['size'] * 100
                print(f"   {i}. {sport}: {percentage:.1f}%")
            
            print("-" * 60)
    
    def run_advanced_analysis(self, gender=None):
        """Run advanced clustering analysis."""
        if gender:
            print(f"\n{'='*60}")
            print(f"🏃 ADVANCED ANALYSIS: {gender.upper()} ATHLETES")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("🏆 ADVANCED ANALYSIS: ALL ATHLETES")
            print(f"{'='*60}")
        
        # Preprocess data
        df_processed, feature_cols = self.preprocess_data(gender)
        
        # Find optimal clusters
        optimal_k, best_scaler, best_score = self.find_optimal_clusters_advanced(df_processed, feature_cols)
        
        # Perform advanced clustering
        cluster_labels, X_scaled, silhouette, calinski, davies_bouldin = self.perform_advanced_clustering(
            df_processed, feature_cols, optimal_k, best_scaler)
        
        # Analyze clusters
        cluster_stats = self.analyze_clusters_detailed(df_processed, cluster_labels, feature_cols)
        
        # Create visualization
        fig = self.create_advanced_visualization(df_processed, cluster_labels, X_scaled, 
                                               f" - {gender if gender else 'All'} Athletes")
        
        # Store results
        analysis_key = gender if gender else 'all'
        self.results[analysis_key] = {
            'df': df_processed,
            'cluster_labels': cluster_labels,
            'X_scaled': X_scaled,
            'cluster_stats': cluster_stats,
            'k': optimal_k,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies_bouldin': davies_bouldin,
            'scaler': best_scaler
        }
        
        # Print detailed summaries
        self.print_detailed_summaries(cluster_stats, f"{gender if gender else 'All'} Athletes")
        
        return cluster_stats, fig
    
    def run_complete_advanced_analysis(self):
        """Run complete advanced analysis."""
        print("🚀 ADVANCED ATHLETE BODY TYPE CLUSTERING PIPELINE")
        print("=" * 70)
        print("Advanced features:")
        print("• Multiple scaling approaches (Standard, Robust, MinMax)")
        print("• Multiple clustering algorithms (K-Means, Agglomerative)")
        print("• Advanced feature engineering with body type archetypes")
        print("• Strategic weighting without sport label leakage")
        print("• Comprehensive cluster validation and analysis")
        
        # Load data
        self.load_data()
        
        # Run analyses
        analyses = [
            (None, "All Athletes"),
            ('M', "Male Athletes"), 
            ('F', "Female Athletes")
        ]
        
        for gender, title in analyses:
            cluster_stats, fig = self.run_advanced_analysis(gender)
            
            # Save visualization
            suffix = f"_{gender.lower()}" if gender else "_all"
            fig.savefig(f'advanced_clustering{suffix}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Advanced visualization saved as 'advanced_clustering{suffix}.png'")
            
            # Save results
            if gender in self.results:
                result = self.results[gender]
                df_output = result['df'].copy()
                df_output['cluster'] = result['cluster_labels']
                filename = f'advanced_clustering_results_{gender}.csv'
                df_output.to_csv(filename, index=False)
                print(f"💾 Results saved to '{filename}'")
            elif 'all' in self.results:
                result = self.results['all']
                df_output = result['df'].copy()
                df_output['cluster'] = result['cluster_labels']
                filename = f'advanced_clustering_results_all.csv'
                df_output.to_csv(filename, index=False)
                print(f"💾 Results saved to '{filename}'")
        
        print("\n✅ Advanced clustering analysis complete!")
        print("📁 Files generated:")
        print("   • advanced_clustering_all.png")
        print("   • advanced_clustering_male.png") 
        print("   • advanced_clustering_female.png")
        print("   • advanced_clustering_results_all.csv")
        print("   • advanced_clustering_results_male.csv")
        print("   • advanced_clustering_results_female.csv")

def main():
    """Main execution function."""
    # Initialize advanced clusterer
    clusterer = AdvancedBodyTypeClusterer('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    
    # Run complete advanced analysis
    clusterer.run_complete_advanced_analysis()

if __name__ == "__main__":
    main()
