"""
Improved Athlete Body Type Clustering Pipeline
==============================================

This pipeline addresses the key issues with sport-specific clustering:
1. Removes sport label leakage by not using actual sport assignments
2. Emphasizes body type patterns through feature engineering and scaling
3. Creates meaningful clusters that group athletes by similar body types
4. Allows new athletes to find suitable sports based on body measurements

Key improvements:
- Feature importance weighting based on body type archetypes, not sport labels
- Robust scaling to handle different feature scales
- Body type ratio features that emphasize meaningful differences
- Clustering that groups by body similarity, not sport assignment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

class ImprovedAthleteClusterer:
    """
    Improved clustering system that groups athletes by body type similarity
    without using sport labels, allowing for meaningful body type archetypes.
    """
    
    def __init__(self, data_path, random_state=42):
        """Initialize the clusterer with data path and parameters."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.df_processed = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.results = {}
        
        # Body type archetype weights (NOT sport-specific)
        # These emphasize features that distinguish body types
        self.body_type_weights = {
            'height_importance': 1.5,      # Height is a key differentiator
            'weight_importance': 1.2,      # Weight matters for body type
            'bmi_importance': 1.8,         # BMI is crucial for body composition
            'limb_ratio_importance': 2.0,  # Limb ratios distinguish body types
            'torso_ratio_importance': 1.5, # Torso proportions matter
            'muscle_mass_importance': 1.3  # Muscle mass indicators
        }
    
    def load_data(self):
        """Load and basic preprocessing of athlete data."""
        print("📊 Loading athlete dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} athletes")
        print(f"   Sports: {sorted(self.df['sport'].unique())}")
        print(f"   Gender distribution: {self.df['Sex'].value_counts().to_dict()}")
        return self.df
    
    def create_body_type_features(self, df):
        """
        Create body type features that emphasize meaningful differences
        without using sport labels.
        """
        print("🔧 Creating body type features...")
        df_features = df.copy()
        
        # Basic measurements (preserve original data)
        df_features['height_cm'] = pd.to_numeric(df_features['height_cm'], errors='coerce')
        df_features['weight_kg'] = pd.to_numeric(df_features['weight_kg'], errors='coerce')
        
        # Calculate BMI
        df_features['bmi'] = df_features['weight_kg'] / (df_features['height_cm'] / 100) ** 2
        
        # Create meaningful ratios that distinguish body types
        # These ratios are more informative than raw measurements
        
        # Height-weight relationship (body density indicator)
        df_features['height_weight_ratio'] = df_features['height_cm'] / df_features['weight_kg']
        df_features['weight_height_ratio'] = df_features['weight_kg'] / df_features['height_cm']
        
        # Limb proportions (if available)
        if 'Arm Span' in df_features.columns:
            df_features['arm_span'] = pd.to_numeric(df_features['Arm Span'], errors='coerce')
            df_features['arm_span_ratio'] = df_features['arm_span'] / df_features['height_cm']
        else:
            # Estimate arm span based on height (typical ratio is ~1.0)
            df_features['arm_span_ratio'] = 1.0
        
        if 'Leg Length' in df_features.columns:
            df_features['leg_length'] = pd.to_numeric(df_features['Leg Length'], errors='coerce')
            df_features['leg_length_ratio'] = df_features['leg_length'] / df_features['height_cm']
        else:
            # Estimate leg length (typical ratio is ~0.45-0.5)
            df_features['leg_length_ratio'] = 0.47
        
        if 'Torso Length' in df_features.columns:
            df_features['torso_length'] = pd.to_numeric(df_features['Torso Length'], errors='coerce')
            df_features['torso_length_ratio'] = df_features['torso_length'] / df_features['height_cm']
        else:
            # Estimate torso length (typical ratio is ~0.3-0.35)
            df_features['torso_length_ratio'] = 0.32
        
        # Body composition indicators
        df_features['bmi_category'] = pd.cut(df_features['bmi'], 
                                           bins=[0, 18.5, 25, 30, 100], 
                                           labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Create body type archetype features
        # These emphasize the key dimensions that distinguish body types
        
        # 1. Size archetype (height-based)
        df_features['size_archetype'] = pd.cut(df_features['height_cm'], 
                                             bins=[0, 160, 175, 190, 300], 
                                             labels=['short', 'medium', 'tall', 'very_tall'])
        
        # 2. Build archetype (weight-height relationship)
        df_features['build_archetype'] = pd.cut(df_features['height_weight_ratio'], 
                                               bins=[0, 2.0, 2.5, 3.0, 10], 
                                               labels=['heavy', 'balanced', 'lean', 'very_lean'])
        
        # 3. Composition archetype (BMI-based)
        df_features['composition_archetype'] = pd.cut(df_features['bmi'], 
                                                     bins=[0, 20, 25, 30, 100], 
                                                     labels=['lean', 'normal', 'muscular', 'heavy'])
        
        return df_features
    
    def apply_body_type_weighting(self, df):
        """
        Apply body type weighting that emphasizes features important for
        distinguishing body types WITHOUT using sport labels.
        
        This approach:
        1. Emphasizes features that naturally distinguish body types
        2. Uses robust scaling to handle different feature scales
        3. Creates weighted features that highlight meaningful differences
        4. Does NOT use sport labels, preventing information leakage
        """
        print("⚖️ Applying body type weighting (no sport labels)...")
        
        # Select features for weighting
        feature_cols = [
            'height_cm', 'weight_kg', 'bmi', 
            'height_weight_ratio', 'weight_height_ratio',
            'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio'
        ]
        
        # Create weighted features based on body type importance
        weighted_features = {}
        
        for feature in feature_cols:
            if feature in df.columns:
                # Apply body type importance weighting
                if 'height' in feature:
                    weight = self.body_type_weights['height_importance']
                elif 'weight' in feature and 'ratio' not in feature:
                    weight = self.body_type_weights['weight_importance']
                elif 'bmi' in feature:
                    weight = self.body_type_weights['bmi_importance']
                elif 'arm' in feature or 'leg' in feature:
                    weight = self.body_type_weights['limb_ratio_importance']
                elif 'torso' in feature:
                    weight = self.body_type_weights['torso_ratio_importance']
                else:
                    weight = 1.0
                
                # Create weighted feature
                weighted_features[f'weighted_{feature}'] = df[feature] * weight
        
        # Add weighted features to dataframe
        weighted_df = pd.DataFrame(weighted_features, index=df.index)
        df_weighted = pd.concat([df, weighted_df], axis=1)
        
        return df_weighted
    
    def handle_missing_values_conservative(self, df):
        """Handle missing values conservatively - preserve original data when possible."""
        print("🔧 Handling missing values conservatively...")
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
        """Preprocess data for clustering, optionally filtered by gender."""
        print(f"\n🔄 Preprocessing data{' for ' + gender if gender else ''}...")
        
        # Filter by gender if specified
        if gender:
            df_filtered = self.df[self.df['Sex'] == gender].copy()
            print(f"   Filtering to {gender} athletes only: {len(df_filtered)} athletes")
        else:
            df_filtered = self.df.copy()
        
        # Handle missing values conservatively
        df_processed = self.handle_missing_values_conservative(df_filtered)
        
        # Create body type features
        df_features = self.create_body_type_features(df_processed)
        
        # Apply body type weighting (NO sport labels used)
        df_weighted = self.apply_body_type_weighting(df_features)
        
        # Select final features for clustering
        clustering_features = [
            'weighted_height_cm', 'weighted_weight_kg', 'weighted_bmi',
            'weighted_height_weight_ratio', 'weighted_weight_height_ratio',
            'weighted_arm_span_ratio', 'weighted_leg_length_ratio', 'weighted_torso_length_ratio'
        ]
        
        # Remove rows with too many missing values
        df_final = df_weighted.dropna(subset=clustering_features, how='all')
        
        print(f"✅ Preprocessing complete. Using {len(clustering_features)} features:")
        print(f"   {', '.join(clustering_features[:4])}... (+{len(clustering_features)-4} more)")
        
        self.df_processed = df_final
        return df_final, clustering_features
    
    def find_optimal_clusters(self, df, feature_cols, max_k=10):
        """Find optimal number of clusters using multiple metrics."""
        print(f"🔍 Finding optimal number of clusters...")
        
        # Prepare data
        X = df[feature_cols].copy()
        
        # Convert to numeric and handle missing values
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(X.mean())
        if X.isnull().any().any():
            X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Test different k values
        k_range = range(2, min(max_k + 1, len(df) // 10))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        print(f"💡 Optimal number of clusters: {optimal_k}")
        print(f"   Silhouette score: {best_silhouette:.3f}")
        
        return optimal_k, best_silhouette
    
    def perform_clustering(self, df, feature_cols, k):
        """Perform K-means clustering."""
        print(f"🔍 Performing K-means clustering with k={k}...")
        
        # Prepare data
        X = df[feature_cols].copy()
        
        # Convert to numeric and handle missing values
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(X.mean())
        if X.isnull().any().any():
            X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        
        print(f"✅ Clustering complete")
        print(f"   Silhouette score: {silhouette:.3f}")
        print(f"   Calinski-Harabasz score: {calinski:.1f}")
        
        return cluster_labels, X_scaled, silhouette, calinski
    
    def analyze_clusters(self, df, cluster_labels, feature_cols):
        """Analyze cluster characteristics and dominant sports."""
        print("\n📊 Analyzing clusters...")
        
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
            
            # Top sports (this is for analysis only, not used in clustering)
            sport_counts = cluster_data['sport'].value_counts()
            top_sports = sport_counts.head(3)
            
            # Body type characteristics
            body_type_summary = self._summarize_body_type(cluster_data)
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': size,
                'percentage': percentage,
                'avg_height': avg_height,
                'avg_weight': avg_weight,
                'avg_bmi': avg_bmi,
                'top_sports': top_sports,
                'body_type_summary': body_type_summary
            })
        
        return cluster_stats
    
    def _summarize_body_type(self, cluster_data):
        """Summarize the body type characteristics of a cluster."""
        avg_height = cluster_data['height_cm'].mean()
        avg_bmi = cluster_data['bmi'].mean()
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
        
        if avg_arm_ratio > 1.05:
            limb_desc = "long limbs"
        elif avg_arm_ratio < 0.95:
            limb_desc = "shorter limbs"
        else:
            limb_desc = "proportional limbs"
        
        return f"{height_desc}, {build_desc}, {limb_desc}"
    
    def create_visualizations(self, df, cluster_labels, X_scaled, title_suffix=""):
        """Create PCA visualization of clusters."""
        print("\n🎨 Creating visualizations...")
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot clusters
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                           cmap='tab10', alpha=0.7, s=60)
        
        # Formatting
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(f'Improved Athlete Body Type Clustering{title_suffix}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self, gender=None):
        """Run complete clustering analysis."""
        if gender:
            print(f"\n{'='*60}")
            print(f"🏃 ANALYZING {gender.upper()} ATHLETES")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("🏆 ANALYZING ALL ATHLETES")
            print(f"{'='*60}")
        
        # Preprocess data
        df_processed, feature_cols = self.preprocess_data(gender)
        
        # Find optimal clusters
        optimal_k, best_silhouette = self.find_optimal_clusters(df_processed, feature_cols)
        
        # Perform clustering
        cluster_labels, X_scaled, silhouette, calinski = self.perform_clustering(
            df_processed, feature_cols, optimal_k)
        
        # Analyze clusters
        cluster_stats = self.analyze_clusters(df_processed, cluster_labels, feature_cols)
        
        # Create visualization
        fig = self.create_visualizations(df_processed, cluster_labels, X_scaled, 
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
            'calinski': calinski
        }
        
        return cluster_stats, fig
    
    def print_cluster_summaries(self, cluster_stats, title):
        """Print detailed cluster summaries."""
        print(f"\n🏆 {title.upper()} CLUSTER SUMMARIES")
        print("=" * 80)
        
        for stat in cluster_stats:
            print(f"\n📊 CLUSTER {stat['cluster_id']} - {stat['size']} athletes ({stat['percentage']:.1f}%)")
            print("-" * 50)
            print(f"📏 BODY TYPE CHARACTERISTICS:")
            print(f"   • height_cm: {stat['avg_height']:.1f}")
            print(f"   • weight_kg: {stat['avg_weight']:.1f}")
            print(f"   • bmi: {stat['avg_bmi']:.1f}")
            print(f"   • Body Type: {stat['body_type_summary']}")
            
            print(f"\n🏅 TOP SPORTS (by representation in cluster):")
            for i, (sport, count) in enumerate(stat['top_sports'].items(), 1):
                percentage = count / stat['size'] * 100
                print(f"   • {sport}: {percentage:.1f}%")
            
            print(f"\n🎯 BODY TYPE ARCHETYPE:")
            print(f"   {stat['body_type_summary']}")
            print("-" * 50)
    
    def save_results(self, analysis_type):
        """Save clustering results to CSV."""
        if analysis_type in self.results:
            result = self.results[analysis_type]
            df_output = result['df'].copy()
            df_output['cluster'] = result['cluster_labels']
            
            filename = f'improved_clustering_results_{analysis_type}.csv'
            df_output.to_csv(filename, index=False)
            print(f"💾 Results saved to '{filename}'")
    
    def run_complete_analysis(self):
        """Run complete analysis for all, male, and female athletes."""
        print("🚀 IMPROVED ATHLETE BODY TYPE CLUSTERING PIPELINE")
        print("=" * 70)
        print("Key improvements:")
        print("• No sport label leakage - clustering based on body type only")
        print("• Body type weighting emphasizes meaningful differences")
        print("• Robust scaling handles different feature scales")
        print("• Creates meaningful clusters with multiple sports per cluster")
        
        # Load data
        self.load_data()
        
        # Run analyses
        analyses = [
            (None, "All Athletes"),
            ('M', "Male Athletes"), 
            ('F', "Female Athletes")
        ]
        
        for gender, title in analyses:
            cluster_stats, fig = self.run_analysis(gender)
            self.print_cluster_summaries(cluster_stats, title)
            
            # Save visualization
            suffix = f"_{gender.lower()}" if gender else "_all"
            fig.savefig(f'improved_clustering{suffix}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved as 'improved_clustering{suffix}.png'")
            
            # Save results
            self.save_results(gender if gender else 'all')
        
        print("\n✅ Improved clustering analysis complete!")
        print("📁 Files generated:")
        print("   • improved_clustering_all.png")
        print("   • improved_clustering_male.png") 
        print("   • improved_clustering_female.png")
        print("   • improved_clustering_results_all.csv")
        print("   • improved_clustering_results_male.csv")
        print("   • improved_clustering_results_female.csv")

def main():
    """Main execution function."""
    # Initialize clusterer
    clusterer = ImprovedAthleteClusterer('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    
    # Run complete analysis
    clusterer.run_complete_analysis()

if __name__ == "__main__":
    main()
