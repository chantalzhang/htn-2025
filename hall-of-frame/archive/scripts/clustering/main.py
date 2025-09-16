"""
Main orchestrator for athlete body type clustering analysis.
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import AthleteDataLoader
from feature_engineering import FeatureEngineer
from kmeans_clustering import AthleteKMeansClusterer
from visualization import ClusteringVisualizer
from analysis import ClusterAnalyzer


class AthleteClusteringPipeline:
    """Main pipeline for athlete body type clustering analysis."""
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize the clustering pipeline.
        
        Args:
            data_path: Path to the merged athlete dataset CSV
            random_state: Random state for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        
        # Initialize components
        self.data_loader = AthleteDataLoader(data_path)
        self.feature_engineer = FeatureEngineer()
        self.clusterer = AthleteKMeansClusterer(random_state)
        self.visualizer = ClusteringVisualizer()
        self.analyzer = ClusterAnalyzer()
        
        # Data storage
        self.df = None
        self.df_processed = None
        self.features_scaled = None
        self.clusters = None
        self.cluster_centers = None
    
    def run_complete_analysis(self) -> dict:
        """
        Run the complete clustering analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        print("🚀 Starting Complete Athlete Body Type Clustering Analysis")
        print("=" * 60)
        
        # Step 1: Load data
        self.df = self.data_loader.load_data()
        
        # Step 2: Handle missing values
        self.df_processed = self.data_loader.handle_missing_values()
        
        # Step 3: Create body ratio features
        self.df_processed = self.feature_engineer.create_body_ratios(self.df_processed)
        
        # Step 4: Select and scale features
        features_df = self.feature_engineer.select_clustering_features(self.df_processed)
        self.features_scaled = self.feature_engineer.scale_features(features_df)
        
        # Update df_processed to match features (remove rows with NaN)
        self.df_processed = self.df_processed.loc[self.features_scaled.index]
        
        # Step 5: Perform clustering
        self.clusters = self.clusterer.perform_clustering(self.features_scaled, self.df_processed)
        
        # Step 6: Get cluster centers
        self.cluster_centers = self.clusterer.get_cluster_centers(
            self.features_scaled, self.feature_engineer.scaler
        )
        
        # Step 7: Create visualizations
        self.visualizer.create_pca_visualization(
            self.features_scaled, self.df_processed, self.clusters
        )
        
        # Step 8: Analyze clusters
        analysis_results = self.analyzer.analyze_clusters(
            self.df_processed, self.clusters, self.cluster_centers
        )
        
        # Step 9: Generate athlete table
        athlete_table = self.analyzer.generate_athlete_table(self.df_processed, self.clusters)
        
        print("\n🎉 Analysis complete!")
        
        return {
            'clusters': self.clusters,
            'cluster_centers': self.cluster_centers,
            'analysis_results': analysis_results,
            'athlete_table': athlete_table,
            'feature_importance': self.feature_engineer.get_feature_importance(),
            'data_summary': self.data_loader.get_data_summary()
        }
    
    def save_results(self, output_path: str = "athlete_clustering_results.csv"):
        """
        Save clustering results to CSV.
        
        Args:
            output_path: Path to save results CSV
        """
        if self.df_processed is not None:
            athlete_table = self.analyzer.generate_athlete_table(self.df_processed, self.clusters)
            athlete_table.to_csv(output_path, index=False)
            print(f"💾 Results saved to {output_path}")
        else:
            print("❌ No results to save. Run analysis first.")


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = AthleteClusteringPipeline(
        data_path="athlete_dataset_pipeline/athlete_dataset_merged.csv"
    )
    
    # Run analysis
    results = pipeline.run_complete_analysis()
    
    # Save results
    pipeline.save_results()
    
    return results


if __name__ == "__main__":
    results = main()
