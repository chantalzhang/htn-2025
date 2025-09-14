"""
Optimized Athlete Body Type Clustering Analysis
==============================================

This script implements the user's recommended approach for cluster optimization:
- Men only: 3-4 clusters
- Women only: 2-3 clusters  
- Combined: 3-5 clusters

Focus on body type similarity rather than sport, with proper silhouette score validation.

Author: AI Assistant
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

class OptimizedAthleteClusterer:
    """
    Optimized clustering of athletes based on body measurements.
    Uses sensible cluster counts and focuses on body type archetypes.
    """
    
    def __init__(self, data_path, random_state=42):
        """Initialize the clusterer with data path and parameters."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.df_processed = None
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.results = {}
        
    def load_and_clean_data(self):
        """Load and clean the athlete dataset."""
        print("📊 Loading and cleaning athlete dataset...")
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