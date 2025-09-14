"""
K-means clustering implementation for athlete body type analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, List


class AthleteKMeansClusterer:
    """Handles K-means clustering for athlete body type analysis."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize K-means clusterer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.clusters = {}
        
    def find_optimal_clusters(self, data: pd.DataFrame, max_k: int = 10) -> Tuple[int, int, List, List, range]:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            data: Scaled feature data
            max_k: Maximum number of clusters to test
            
        Returns:
            Tuple of (elbow_k, silhouette_k, inertias, silhouette_scores, k_range)
        """
        print(f"\n🔍 Finding optimal number of clusters (max_k={max_k})...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
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
    
    def perform_clustering(self, features_scaled: pd.DataFrame, df_processed: pd.DataFrame) -> Dict:
        """
        Perform K-means clustering - separate for males/females and combined.
        
        Args:
            features_scaled: Scaled feature data
            df_processed: Original processed data with athlete info
            
        Returns:
            Dictionary with clustering results
        """
        print("\n🎯 Performing K-means clustering...")
        
        # Combined clustering (all athletes)
        print("   📊 Combined clustering (all athletes)...")
        elbow_k, sil_k, _, _, _ = self.find_optimal_clusters(features_scaled)
        optimal_k = sil_k  # Use silhouette score for final choice
        
        kmeans_combined = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        df_processed['cluster_combined'] = kmeans_combined.fit_predict(features_scaled)
        
        self.clusters['combined'] = {
            'model': kmeans_combined,
            'n_clusters': optimal_k,
            'silhouette': silhouette_score(features_scaled, kmeans_combined.labels_)
        }
        
        print(f"   ✅ Combined: {optimal_k} clusters, silhouette={self.clusters['combined']['silhouette']:.3f}")
        
        # Gender-specific clustering
        for gender in ['M', 'F']:
            gender_mask = df_processed['Sex'] == gender
            if gender_mask.sum() > 10:  # Only cluster if enough data points
                print(f"   👤 {gender} clustering...")
                
                gender_features = features_scaled[gender_mask]
                elbow_k, sil_k, _, _, _ = self.find_optimal_clusters(gender_features)
                optimal_k_gender = sil_k
                
                kmeans_gender = KMeans(n_clusters=optimal_k_gender, random_state=self.random_state, n_init=10)
                gender_labels = kmeans_gender.fit_predict(gender_features)
                
                # Add gender-specific cluster labels to main dataframe
                df_processed.loc[gender_mask, f'cluster_{gender}'] = gender_labels
                
                self.clusters[gender] = {
                    'model': kmeans_gender,
                    'n_clusters': optimal_k_gender,
                    'silhouette': silhouette_score(gender_features, gender_labels)
                }
                
                print(f"   ✅ {gender}: {optimal_k_gender} clusters, silhouette={self.clusters[gender]['silhouette']:.3f}")
        
        return self.clusters
    
    def get_cluster_centers(self, features_scaled: pd.DataFrame, scaler) -> Dict[str, pd.DataFrame]:
        """
        Get cluster centers in original scale.
        
        Args:
            features_scaled: Scaled feature data
            scaler: The scaler used for feature scaling
            
        Returns:
            Dictionary with cluster centers for each clustering approach
        """
        cluster_centers = {}
        
        for cluster_type, cluster_info in self.clusters.items():
            if 'model' in cluster_info:
                # Get centers in scaled space
                centers_scaled = cluster_info['model'].cluster_centers_
                
                # Transform back to original scale
                centers_original = pd.DataFrame(
                    scaler.inverse_transform(centers_scaled),
                    columns=features_scaled.columns
                )
                
                cluster_centers[cluster_type] = centers_original
        
        return cluster_centers
    
    def get_cluster_quality_metrics(self) -> pd.DataFrame:
        """
        Get quality metrics for all clustering approaches.
        
        Returns:
            DataFrame with quality metrics
        """
        metrics = []
        
        for cluster_type, cluster_info in self.clusters.items():
            if 'silhouette' in cluster_info:
                metrics.append({
                    'clustering_type': cluster_type,
                    'n_clusters': cluster_info['n_clusters'],
                    'silhouette_score': cluster_info['silhouette']
                })
        
        return pd.DataFrame(metrics)
