"""
Visualization tools for athlete clustering analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, Tuple


class ClusteringVisualizer:
    """Handles visualization of clustering results."""
    
    def __init__(self, style: str = 'whitegrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Seaborn style for plots
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (15, 10)
        
    def create_pca_visualization(self, features_scaled: pd.DataFrame, df_processed: pd.DataFrame, 
                               clusters: Dict) -> plt.Figure:
        """
        Create PCA visualization of clusters.
        
        Args:
            features_scaled: Scaled feature data
            df_processed: Original data with cluster assignments
            clusters: Dictionary with clustering results
            
        Returns:
            Matplotlib figure
        """
        print("\n📈 Creating PCA visualizations...")
        
        # Perform PCA
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(features_scaled)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Athlete Body Type Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Combined clustering visualization
        if 'combined' in clusters:
            scatter = axes[0, 0].scatter(pca_data[:, 0], pca_data[:, 1], 
                                       c=df_processed['cluster_combined'], 
                                       cmap='viridis', alpha=0.7, s=50)
            axes[0, 0].set_title(f'Combined Clustering ({clusters["combined"]["n_clusters"]} clusters)')
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter, ax=axes[0, 0])
        
        # Gender-specific clustering
        gender_colors = {'M': 'blue', 'F': 'red'}
        
        if 'M' in clusters:
            male_mask = df_processed['Sex'] == 'M'
            male_pca = pca_data[male_mask]
            scatter_m = axes[0, 1].scatter(male_pca[:, 0], male_pca[:, 1], 
                                         c=df_processed.loc[male_mask, 'cluster_M'], 
                                         cmap='Blues', alpha=0.7, s=50)
            axes[0, 1].set_title(f'Male Clustering ({clusters["M"]["n_clusters"]} clusters)')
            axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter_m, ax=axes[0, 1])
        
        if 'F' in clusters:
            female_mask = df_processed['Sex'] == 'F'
            female_pca = pca_data[female_mask]
            scatter_f = axes[0, 2].scatter(female_pca[:, 0], female_pca[:, 1], 
                                         c=df_processed.loc[female_mask, 'cluster_F'], 
                                         cmap='Reds', alpha=0.7, s=50)
            axes[0, 2].set_title(f'Female Clustering ({clusters["F"]["n_clusters"]} clusters)')
            axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter_f, ax=axes[0, 2])
        
        # Sport distribution
        sport_counts = df_processed['sport'].value_counts()
        axes[1, 0].bar(range(len(sport_counts)), sport_counts.values)
        axes[1, 0].set_xticks(range(len(sport_counts)))
        axes[1, 0].set_xticklabels(sport_counts.index, rotation=45, ha='right')
        axes[1, 0].set_title('Athletes by Sport')
        axes[1, 0].set_ylabel('Number of Athletes')
        
        # Gender distribution by sport
        gender_sport = pd.crosstab(df_processed['sport'], df_processed['Sex'])
        gender_sport.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Gender Distribution by Sport')
        axes[1, 1].set_ylabel('Number of Athletes')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(['Male', 'Female'])
        
        # Clustering quality comparison
        quality_data = []
        for cluster_type, cluster_info in clusters.items():
            if 'silhouette' in cluster_info:
                quality_data.append({
                    'Type': cluster_type.title(),
                    'Silhouette Score': cluster_info['silhouette'],
                    'N Clusters': cluster_info['n_clusters']
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            bars = axes[1, 2].bar(quality_df['Type'], quality_df['Silhouette Score'])
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].set_ylabel('Silhouette Score')
            
            # Add cluster count labels on bars
            for i, (bar, n_clusters) in enumerate(zip(bars, quality_df['N Clusters'])):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{n_clusters} clusters', ha='center', va='bottom', fontsize=9)
        
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        axes[1, 2].set_title('Clustering Quality')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_sport_cluster_heatmap(self, df_processed: pd.DataFrame) -> plt.Figure:
        """
        Create heatmap showing sport distribution across clusters.
        
        Args:
            df_processed: DataFrame with cluster assignments
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create crosstab of sports vs clusters
        sport_cluster = pd.crosstab(df_processed['sport'], df_processed['cluster_combined'], normalize='columns') * 100
        
        # Create heatmap
        sns.heatmap(sport_cluster, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        ax.set_title('Sport Distribution Across Clusters (%)')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Sport')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame) -> plt.Figure:
        """
        Plot feature importance based on variance.
        
        Args:
            feature_importance: DataFrame with feature importance scores
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(feature_importance['feature'], feature_importance['variance'])
        ax.set_title('Feature Importance (Variance)')
        ax.set_ylabel('Variance')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig
