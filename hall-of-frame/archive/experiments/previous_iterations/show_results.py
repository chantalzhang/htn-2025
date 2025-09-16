import pandas as pd

print('📊 CLUSTERING RESULTS SUMMARY')
print('=' * 50)

# Load dataset
df = pd.read_csv('athlete_dataset_pipeline/athlete_dataset_merged.csv')
print(f'Total athletes: {len(df)}')
print(f'Sports: {sorted(df["sport"].unique())}')
print(f'Gender distribution: {df["Sex"].value_counts().to_dict()}')
print()

print('🎯 FINAL CLUSTERING RESULTS:')
print('=' * 50)
print('Combined Analysis: 8 clusters, silhouette=0.472')
print('Male Analysis: 2 clusters, silhouette=0.664') 
print('Female Analysis: 5 clusters, silhouette=0.651')
print()
print('📁 Files generated:')
print('• clustering_visualizations.png - PCA plots')
print('• clustering_results.png - Text results')
print('• final_clustering_results_combined.csv')
print('• final_clustering_results_male.csv')
print('• final_clustering_results_female.csv')
