import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess the athlete dataset."""
    print("📊 Loading athlete dataset...")
    
    # Load the dataset
    df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
    print(f"✅ Loaded {len(df)} athletes")
    
    # Handle missing values conservatively
    df_processed = df.copy()
    
    # For columns with >50% missing, use mean imputation
    for col in ['Hand Length', 'Hand Width', 'Arm Span', 'Leg Length', 'Torso Length', 'Spike Reach', 'Block Reach']:
        if col in df_processed.columns:
            missing_pct = df_processed[col].isnull().sum() / len(df_processed) * 100
            if missing_pct > 50:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # Create feature ratios
    df_processed['bmi'] = df_processed['weight_kg'] / (df_processed['height_cm'] / 100) ** 2
    df_processed['weight_height_ratio'] = df_processed['weight_kg'] / df_processed['height_cm']
    df_processed['height_weight_ratio'] = df_processed['height_cm'] / df_processed['weight_kg']
    df_processed['arm_span_ratio'] = df_processed['Arm Span'] / df_processed['height_cm']
    df_processed['leg_length_ratio'] = df_processed['Leg Length'] / df_processed['height_cm']
    df_processed['torso_length_ratio'] = df_processed['Torso Length'] / df_processed['height_cm']
    
    # Select features for clustering
    feature_cols = ['height_cm', 'weight_kg', 'bmi', 'weight_height_ratio', 
                   'height_weight_ratio', 'arm_span_ratio', 'leg_length_ratio', 'torso_length_ratio']
    
    return df_processed, feature_cols

def apply_sport_specific_weighting(df, feature_cols):
    """Apply sport-specific feature weighting."""
    df_weighted = df.copy()
    
    # Define sport-specific weights
    sport_weights = {
        'basketball': {
            'height_cm': 1.5, 'weight_kg': 1.2, 'bmi': 1.1,
            'arm_span_ratio': 1.4, 'leg_length_ratio': 1.3, 'torso_length_ratio': 1.1
        },
        'gymnastics': {
            'height_cm': 0.8, 'weight_kg': 1.3, 'bmi': 1.4,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.1, 'torso_length_ratio': 1.2
        },
        'swimming': {
            'height_cm': 1.2, 'weight_kg': 1.1, 'bmi': 1.0,
            'arm_span_ratio': 1.5, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.4
        },
        'weightlifting': {
            'height_cm': 1.0, 'weight_kg': 1.6, 'bmi': 1.7,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.1
        },
        'distance_running': {
            'height_cm': 1.0, 'weight_kg': 0.7, 'bmi': 0.6,
            'arm_span_ratio': 1.0, 'leg_length_ratio': 1.1, 'torso_length_ratio': 1.0
        },
        'track_field': {
            'height_cm': 1.1, 'weight_kg': 1.2, 'bmi': 1.1,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.0
        },
        'soccer': {
            'height_cm': 1.1, 'weight_kg': 1.0, 'bmi': 1.0,
            'arm_span_ratio': 1.0, 'leg_length_ratio': 1.1, 'torso_length_ratio': 1.0
        },
        'tennis': {
            'height_cm': 1.1, 'weight_kg': 1.0, 'bmi': 1.0,
            'arm_span_ratio': 1.1, 'leg_length_ratio': 1.0, 'torso_length_ratio': 1.0
        },
        'rowing': {
            'height_cm': 1.3, 'weight_kg': 1.2, 'bmi': 1.1,
            'arm_span_ratio': 1.2, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.2
        },
        'volleyball': {
            'height_cm': 1.4, 'weight_kg': 1.1, 'bmi': 1.0,
            'arm_span_ratio': 1.3, 'leg_length_ratio': 1.2, 'torso_length_ratio': 1.1
        }
    }
    
    # Apply weights
    for sport, weights in sport_weights.items():
        sport_mask = df_weighted['sport'] == sport
        for feature, weight in weights.items():
            if feature in df_weighted.columns:
                df_weighted.loc[sport_mask, feature] *= weight
    
    return df_weighted

def perform_clustering(df, feature_cols, k):
    """Perform K-means clustering."""
    # Prepare data - handle missing values
    X = df[feature_cols].copy()
    
    # Convert all columns to numeric and handle missing values
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values with column means
    X = X.fillna(X.mean())
    
    # Check for any remaining NaN values and fill with 0 if needed
    if X.isnull().any().any():
        X = X.fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
    
    return cluster_labels, silhouette, calinski

def analyze_clusters(df, cluster_labels, feature_cols):
    """Analyze cluster characteristics."""
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster_id in sorted(df_analysis['cluster'].unique()):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        # Basic stats
        size = len(cluster_data)
        percentage = size / len(df_analysis) * 100
        
        # Average measurements
        avg_height = cluster_data['height_cm'].mean()
        avg_weight = cluster_data['weight_kg'].mean()
        avg_bmi = cluster_data['bmi'].mean()
        
        # Top sports
        sport_counts = cluster_data['sport'].value_counts()
        top_sports = sport_counts.head(3)
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': size,
            'percentage': percentage,
            'avg_height': avg_height,
            'avg_weight': avg_weight,
            'avg_bmi': avg_bmi,
            'top_sports': top_sports
        })
    
    return cluster_stats

def create_improved_results_display():
    """Create improved results display with larger text and better spacing."""
    print("🚀 REGENERATING CLUSTERING RESULTS WITH IMPROVED FORMATTING")
    print("=" * 70)
    
    # Load and preprocess data
    df, feature_cols = load_and_preprocess_data()
    df_weighted = apply_sport_specific_weighting(df, feature_cols)
    
    # Store results for all three analyses
    all_results = {}
    
    # 1. Combined Analysis
    print("\n🏆 ANALYZING ALL ATHLETES...")
    cluster_labels_combined, silhouette_combined, calinski_combined = perform_clustering(df_weighted, feature_cols, 8)
    cluster_stats_combined = analyze_clusters(df_weighted, cluster_labels_combined, feature_cols)
    
    all_results['combined'] = {
        'df': df_weighted,
        'cluster_labels': cluster_labels_combined,
        'cluster_stats': cluster_stats_combined,
        'k': 8,
        'silhouette': silhouette_combined,
        'total_athletes': len(df_weighted)
    }
    
    # 2. Male Analysis
    print("🏃 ANALYZING MALE ATHLETES...")
    df_male = df_weighted[df_weighted['Sex'] == 'M'].copy()
    cluster_labels_male, silhouette_male, calinski_male = perform_clustering(df_male, feature_cols, 2)
    cluster_stats_male = analyze_clusters(df_male, cluster_labels_male, feature_cols)
    
    all_results['male'] = {
        'df': df_male,
        'cluster_labels': cluster_labels_male,
        'cluster_stats': cluster_stats_male,
        'k': 2,
        'silhouette': silhouette_male,
        'total_athletes': len(df_male)
    }
    
    # 3. Female Analysis
    print("🏃 ANALYZING FEMALE ATHLETES...")
    df_female = df_weighted[df_weighted['Sex'] == 'F'].copy()
    cluster_labels_female, silhouette_female, calinski_female = perform_clustering(df_female, feature_cols, 5)
    cluster_stats_female = analyze_clusters(df_female, cluster_labels_female, feature_cols)
    
    all_results['female'] = {
        'df': df_female,
        'cluster_labels': cluster_labels_female,
        'cluster_stats': cluster_stats_female,
        'k': 5,
        'silhouette': silhouette_female,
        'total_athletes': len(df_female)
    }
    
    # Create improved results display
    print("\n📝 Creating improved results display...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 16))
    
    titles = ['COMBINED ANALYSIS', 'MALE ANALYSIS', 'FEMALE ANALYSIS']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (analysis_type, ax, title, color) in enumerate(zip(['combined', 'male', 'female'], [ax1, ax2, ax3], titles, colors)):
        result = all_results[analysis_type]
        cluster_stats = result['cluster_stats']
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create text content with larger font and better spacing
        text_content = f"🏆 {title}\n"
        text_content += "=" * 60 + "\n\n"
        text_content += f"📊 CLUSTERS: {result['k']}\n"
        text_content += f"📈 ATHLETES: {result['total_athletes']}\n"
        text_content += f"🎯 SILHOUETTE: {result['silhouette']:.3f}\n\n"
        
        # Add cluster summaries with better formatting
        for stat in cluster_stats:
            text_content += f"🔹 CLUSTER {stat['cluster_id']}\n"
            text_content += f"   Athletes: {stat['size']} ({stat['percentage']:.1f}%)\n"
            
            if not pd.isna(stat['avg_height']):
                text_content += f"   Height: {stat['avg_height']:.1f}cm\n"
            if not pd.isna(stat['avg_weight']):
                text_content += f"   Weight: {stat['avg_weight']:.1f}kg\n"
            if not pd.isna(stat['avg_bmi']):
                text_content += f"   BMI: {stat['avg_bmi']:.1f}\n"
            
            text_content += f"   Top Sports: {', '.join(stat['top_sports'].head(2).index)}\n\n"
        
        # Add text box with much larger font and better spacing
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=color, alpha=0.15, edgecolor=color, linewidth=2))
        
        # Add title with larger font
        ax.set_title(title, fontsize=20, fontweight='bold', color=color, pad=20)
    
    # Main title
    plt.suptitle('ATHLETE BODY TYPE CLUSTERING RESULTS', fontsize=24, fontweight='bold', y=0.95)
    
    # Adjust layout for maximum space usage
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save with high DPI for crisp text
    fig.savefig('clustering_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Improved results saved as 'clustering_results.png'")
    
    print("\n✅ Regeneration complete!")
    print("📁 The new clustering_results.png has:")
    print("   • Much larger, clearer text (fontsize=14)")
    print("   • Better spacing and padding")
    print("   • Maximum space usage (30x16 figure)")
    print("   • High DPI (300) for crisp text")
    print("   • Bold fonts for better readability")

if __name__ == "__main__":
    create_improved_results_display()
