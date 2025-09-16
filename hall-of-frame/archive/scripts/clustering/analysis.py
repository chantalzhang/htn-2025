"""
Analysis tools for athlete clustering results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class ClusterAnalyzer:
    """Handles analysis of clustering results."""
    
    def __init__(self):
        """Initialize cluster analyzer."""
        pass
    
    def analyze_clusters(self, df_processed: pd.DataFrame, clusters: Dict, 
                        cluster_centers: Dict) -> Dict:
        """
        Analyze cluster characteristics and show dominant sports per cluster.
        
        Args:
            df_processed: DataFrame with cluster assignments
            clusters: Dictionary with clustering results
            cluster_centers: Dictionary with cluster centers
            
        Returns:
            Dictionary with analysis results
        """
        print("\n📊 Analyzing cluster characteristics...")
        
        analysis_results = {}
        
        # Combined clustering analysis
        if 'combined' in clusters:
            print("\n🎯 COMBINED CLUSTERING ANALYSIS:")
            
            if 'combined' in cluster_centers:
                print("\nCluster Centers (Original Scale):")
                print(cluster_centers['combined'].round(2))
            
            # Sport distribution per cluster with dominant sports
            print("\n🏃 DOMINANT SPORTS PER CLUSTER:")
            sport_cluster = pd.crosstab(df_processed['sport'], 
                                       df_processed['cluster_combined'])
            
            sport_cluster_pct = pd.crosstab(df_processed['sport'], 
                                           df_processed['cluster_combined'], 
                                           normalize='columns') * 100
            
            combined_analysis = {}
            for cluster_id in range(clusters['combined']['n_clusters']):
                print(f"\n📍 CLUSTER {cluster_id}:")
                cluster_sports = sport_cluster_pct[cluster_id].sort_values(ascending=False)
                top_3_sports = cluster_sports.head(3)
                
                cluster_info = {
                    'top_sports': {},
                    'body_type': {}
                }
                
                print(f"   Top 3 Sports:")
                for sport, pct in top_3_sports.items():
                    count = sport_cluster.loc[sport, cluster_id]
                    print(f"   • {sport}: {pct:.1f}% ({count} athletes)")
                    cluster_info['top_sports'][sport] = {'percentage': pct, 'count': count}
                
                # Body type description
                if 'combined' in cluster_centers:
                    center = cluster_centers['combined'].iloc[cluster_id]
                    body_desc = f"{center['height_cm']:.0f}cm, {center['weight_kg']:.0f}kg, BMI={center['BMI']:.1f}"
                    print(f"   Body Type: {body_desc}")
                    cluster_info['body_type'] = {
                        'height_cm': center['height_cm'],
                        'weight_kg': center['weight_kg'],
                        'BMI': center['BMI']
                    }
                
                combined_analysis[cluster_id] = cluster_info
            
            analysis_results['combined'] = combined_analysis
        
        # Gender-specific analysis
        for gender in ['M', 'F']:
            if gender in clusters:
                print(f"\n👨👩 {gender.upper()} CLUSTERING ANALYSIS ({clusters[gender]['n_clusters']} clusters):")
                gender_data = df_processed[df_processed['Sex'] == gender]
                gender_sport_cluster = pd.crosstab(gender_data['sport'], gender_data[f'cluster_{gender}'])
                gender_sport_pct = pd.crosstab(gender_data['sport'], gender_data[f'cluster_{gender}'], 
                                             normalize='columns') * 100
                
                gender_analysis = {}
                for cluster_id in range(clusters[gender]['n_clusters']):
                    print(f"\n📍 {gender.upper()} CLUSTER {cluster_id}:")
                    cluster_sports = gender_sport_pct[cluster_id].sort_values(ascending=False)
                    top_3_sports = cluster_sports.head(3)
                    
                    cluster_info = {'top_sports': {}}
                    for sport, pct in top_3_sports.items():
                        count = gender_sport_cluster.loc[sport, cluster_id] if sport in gender_sport_cluster.index else 0
                        if count > 0:
                            print(f"   • {sport}: {pct:.1f}% ({count} athletes)")
                            cluster_info['top_sports'][sport] = {'percentage': pct, 'count': count}
                    
                    gender_analysis[cluster_id] = cluster_info
                
                analysis_results[gender] = gender_analysis
        
        return analysis_results
    
    def generate_athlete_table(self, df_processed: pd.DataFrame, clusters: Dict) -> pd.DataFrame:
        """
        Generate table showing cluster membership per athlete.
        
        Args:
            df_processed: DataFrame with cluster assignments
            clusters: Dictionary with clustering results
            
        Returns:
            DataFrame with athlete cluster membership
        """
        print("\n📋 Generating athlete cluster membership table...")
        
        output_cols = ['Player', 'sport', 'Sex', 'height_cm', 'weight_kg', 'BMI', 'cluster_combined']
        
        # Add gender-specific clusters if they exist
        if 'M' in clusters:
            output_cols.append('cluster_M')
        if 'F' in clusters:
            output_cols.append('cluster_F')
        
        # Filter to existing columns
        available_cols = [col for col in output_cols if col in df_processed.columns]
        
        athlete_table = df_processed[available_cols].copy()
        
        print(f"✅ Generated table with {len(athlete_table)} athletes and {len(available_cols)} columns")
        return athlete_table
    
    def get_cluster_statistics(self, df_processed: pd.DataFrame, clusters: Dict) -> Dict:
        """
        Get detailed statistics for each cluster.
        
        Args:
            df_processed: DataFrame with cluster assignments
            clusters: Dictionary with clustering results
            
        Returns:
            Dictionary with cluster statistics
        """
        stats = {}
        
        for cluster_type in ['combined', 'M', 'F']:
            if cluster_type in clusters:
                cluster_col = f'cluster_{cluster_type}' if cluster_type != 'combined' else 'cluster_combined'
                
                if cluster_col in df_processed.columns:
                    cluster_stats = {}
                    
                    for cluster_id in range(clusters[cluster_type]['n_clusters']):
                        cluster_mask = df_processed[cluster_col] == cluster_id
                        cluster_data = df_processed[cluster_mask]
                        
                        if len(cluster_data) > 0:
                            numeric_cols = ['height_cm', 'weight_kg', 'BMI']
                            available_numeric = [col for col in numeric_cols if col in cluster_data.columns]
                            
                            cluster_stats[cluster_id] = {
                                'size': len(cluster_data),
                                'sports': dict(cluster_data['sport'].value_counts()),
                                'gender_dist': dict(cluster_data['Sex'].value_counts()) if 'Sex' in cluster_data.columns else {},
                                'numeric_stats': cluster_data[available_numeric].describe().to_dict() if available_numeric else {}
                            }
                    
                    stats[cluster_type] = cluster_stats
        
        return stats
