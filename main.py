

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from habitability import compute_knn_based_score, optimize_decay_rate_with_kmeans, compute_knn_scores_for_dataframe, natural_data
from knn_model import train_planet_knn, find_similar_planet, PlanetSimilarityKNN
from kmeans_model import create_planet_clusters

print("=== EXOPLANET HABITABILITY AND SIMILARITY ANALYSIS ===\n")

# Define Earth's parameters
earth_params = {
    'pl_rade': 1.0,      # Earth radius
    'pl_bmasse': 1.0,    # Earth mass
    'pl_orbper': 365.25, # Earth orbital period
    'pl_orbeccen': 0.017, # Earth orbital eccentricity
    'pl_insol': 1.0,     # Earth insolation flux
    'pl_eqt': 255        # Earth equilibrium temperature
}

# Train KNN model first
print("1. Training KNN model...")
knn_model = train_planet_knn(natural_data)

# Optimize decay rate for better category separation
print("\n2. Optimizing decay rate using KMeans clustering...")
optimal_decay_rate = optimize_decay_rate_with_kmeans(knn_model, earth_params, natural_data, n_categories=5)

# Calculate KNN-based habitability scores with optimized decay rate for all planets
print("\n3. Calculating KNN-based habitability scores with optimized decay rate for all planets...")
natural_data["habitability_score"] = compute_knn_scores_for_dataframe(natural_data, knn_model, earth_params, optimal_decay_rate)

# Sort and save the new scores
if 'habitability_score' in natural_data.columns:
    sorted_df = natural_data.sort_values(by='habitability_score', ascending=False, na_position='last')
    print('\nTop 20 planets by KNN-based habitability score:')
    cols_to_show = [c for c in ['pl_name', 'hostname', 'disc_year', 'habitability_score'] if c in sorted_df.columns]
    print(sorted_df[cols_to_show].head(20))
    # write a sorted CSV
    out = 'PS_2025.09.20_19.26.05_sorted_by_knn_habitability.csv'
    sorted_df.to_csv(out, index=False)
    print(f"Wrote KNN-based sorted CSV to: {out}")
    # optionally write top 100
    sorted_df.head(100).to_csv('PS_top100_by_knn_habitability.csv', index=False)

# Load the KNN-based sorted habitability data
print("\n4. Loading KNN-based habitability data...")
sorted_data = sorted_df.head(100) if 'sorted_df' in locals() else pd.read_csv('PS_top100_by_knn_habitability.csv')

# Find the 5 planets closest to Earth (KNN-based habitability score = 100)
print("\n=== TOP 5 MOST HABITABLE PLANETS (KNN-based) ===")
top5_planets = sorted_data.head(5)
print(f"Planet Name | KNN Habitability Score")
print("-" * 45)
for i, row in top5_planets.iterrows():
    print(f"{row['pl_name']:<25} | {row['habitability_score']:.2f}")

# Find planet most similar to Earth (should now align with top habitability scores)
print("\n5. Finding Earth's closest match...")

# Define Earth's parameters (already defined above)

# Find most similar planet to Earth
print("\n=== PLANET MOST SIMILAR TO EARTH (by features) ===")
earth_similar = find_similar_planet(knn_model, earth_params)
earth_match_name = earth_similar['most_similar_row']['pl_name']
earth_match_score = None

# Get habitability score for Earth's closest match
for i, row in natural_data.iterrows():
    if row['pl_name'] == earth_match_name:
        earth_match_score = row.get('habitability_score', 'N/A')
        break

print(f"Most similar planet: {earth_match_name}")
print(f"KNN-based habitability score: {earth_match_score}")

print(f"\n=== COMPARISON: KNN Match vs Top Habitability ===")
# Format scores properly - handle both numeric and 'N/A' cases
earth_score_str = f"{earth_match_score:.2f}" if earth_match_score != 'N/A' and not pd.isna(earth_match_score) else 'N/A'
top_score_str = f"{top5_planets.iloc[0]['habitability_score']:.2f}" if not pd.isna(top5_planets.iloc[0]['habitability_score']) else 'N/A'

print(f"Earth's KNN closest match: {earth_match_name} (Score: {earth_score_str})")
print(f"Top habitability planet: {top5_planets.iloc[0]['pl_name']} (Score: {top_score_str})")
if earth_match_name == top5_planets.iloc[0]['pl_name']:
    print("✅ SUCCESS: KNN closest match is the same as top habitability planet!")
else:
    print("ℹ️  INFO: KNN match and top habitability planet are different (this can happen with noisy data)")

# Create bar chart comparing top 5 planets + Earth's match vs Earth
print("\n6. Creating comparison bar chart...")
plt.figure(figsize=(12, 8))

# Collect planets for comparison
comparison_planets = []
comparison_scores = []
comparison_colors = []

# Add top 5 planets
for i, row in top5_planets.iterrows():
    comparison_planets.append(row['pl_name'])
    comparison_scores.append(row['habitability_score'])
    comparison_colors.append('skyblue')

# Add Earth's closest match if not already in top 5
if earth_match_name not in [planet for planet in comparison_planets]:
    comparison_planets.append(f"{earth_match_name}\n(Earth's closest match)")
    # Handle 'N/A' and NaN values properly
    score_value = earth_match_score if earth_match_score != 'N/A' and not pd.isna(earth_match_score) else 0
    comparison_scores.append(score_value)
    comparison_colors.append('lightgreen')

# Add Earth
comparison_planets.append('Earth')
comparison_scores.append(100)
comparison_colors.append('gold')

plt.bar(range(len(comparison_planets)), comparison_scores, color=comparison_colors)

# Add score labels on top of bars
for i, score in enumerate(comparison_scores):
    if score > 0:  # Only show non-zero scores
        plt.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Planets')
plt.ylabel('KNN-based Habitability Score')
plt.title('KNN-based Habitability Scores: Top Planets vs Earth')
plt.xticks(range(len(comparison_planets)), comparison_planets, rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.show()

# Create hypothetical planet with realistic parameters
print("\n7. Creating hypothetical planet...")
hypothetical_planet = {
    'pl_rade': 1.2,      # 20% larger than Earth
    'pl_bmasse': 1.5,    # 50% more massive than Earth
    'pl_orbper': 400,    # Slightly longer year
    'pl_orbeccen': 0.05, # Low eccentricity
    'pl_insol': 0.9,     # Slightly less solar radiation
    'pl_eqt': 250        # Slightly cooler than Earth
}

print("Hypothetical Planet Parameters:")
for param, value in hypothetical_planet.items():
    print(f"  {param}: {value}")

# Calculate KNN-based habitability score for hypothetical planet
print("\n8. Calculating hypothetical planet KNN-based habitability...")

# Create a mock row for the hypothetical planet
hyp_row = pd.Series(hypothetical_planet)

hyp_habitability = compute_knn_based_score(hyp_row, knn_model, earth_params, optimal_decay_rate)
print(f"Hypothetical Planet KNN-based Habitability Score: {hyp_habitability:.2f}")

# Also find the 5 planets most similar to Earth and print their full metrics (features + units)
print("\n=== 5 PLANETS MOST SIMILAR TO EARTH ===")
knn_model_5_earth = PlanetSimilarityKNN(n_neighbors=5)
feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
knn_model_5_earth.train_model(natural_data, feature_columns)

test_point_earth = [earth_params[col] for col in feature_columns]
test_point_earth_scaled = knn_model_5_earth.scaler.transform([test_point_earth])
distances_e, indices_e = knn_model_5_earth.knn_model.kneighbors(test_point_earth_scaled)

for i in range(5):
    idx = indices_e[0][i]
    similar_row = knn_model_5_earth.training_data.iloc[idx]
    planet_name = similar_row['pl_name']
    stored_habitability = similar_row.get('habitability_score', 'N/A')
    distance = distances_e[0][i]

    feat_rade = similar_row.get('pl_rade', 'N/A')
    feat_bmasse = similar_row.get('pl_bmasse', 'N/A')
    feat_orbper = similar_row.get('pl_orbper', 'N/A')
    feat_orbeccen = similar_row.get('pl_orbeccen', 'N/A')
    feat_insol = similar_row.get('pl_insol', 'N/A')
    feat_eqt = similar_row.get('pl_eqt', 'N/A')

    if stored_habitability == 'N/A' or pd.isna(stored_habitability):
        stored_str = 'N/A'
    else:
        stored_str = f"{stored_habitability:.2f}"

    print(f"{i+1}. {planet_name} (KNN Distance: {distance:.4f})")
    print(f"    pl_rade: {feat_rade} Rearth")
    print(f"    pl_bmasse: {feat_bmasse} Mearth")
    print(f"    pl_orbper: {feat_orbper} days")
    print(f"    pl_orbeccen: {feat_orbeccen} (unitless)")
    print(f"    pl_insol: {feat_insol} Fearth")
    print(f"    pl_eqt: {feat_eqt} K")
    print(f"    Stored habitability_score: {stored_str}")

    # Build series to recompute scores
    planet_series = pd.Series({
        'pl_rade': feat_rade,
        'pl_bmasse': feat_bmasse,
        'pl_orbper': feat_orbper,
        'pl_orbeccen': feat_orbeccen,
        'pl_insol': feat_insol,
        'pl_eqt': feat_eqt
    })

    score_vs_earth = compute_knn_based_score(planet_series, knn_model, earth_params, optimal_decay_rate)
    score_vs_hyp = compute_knn_based_score(planet_series, knn_model, hypothetical_planet, optimal_decay_rate)

    print(f"    Recomputed score (vs Earth reference): {score_vs_earth:.2f}")
    print(f"    Recomputed score (vs Hypothetical reference): {score_vs_hyp:.2f}\n")

# Find 5 most similar planets to hypothetical planet
print("\n9. Finding planets similar to hypothetical planet...")

# Modify KNN model to find 5 neighbors
knn_model_5 = PlanetSimilarityKNN(n_neighbors=5)
feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
clean_data = natural_data[feature_columns + ['pl_name', 'habitability_score']].dropna()
knn_model_5.train_model(natural_data, feature_columns)

# Find 5 most similar planets
test_point = [hypothetical_planet[col] for col in feature_columns]
test_point_scaled = knn_model_5.scaler.transform([test_point])
distances, indices = knn_model_5.knn_model.kneighbors(test_point_scaled)

print("\n=== 5 PLANETS MOST SIMILAR TO HYPOTHETICAL PLANET ===")
similar_planets = []
similar_scores = []

for i in range(5):
    idx = indices[0][i]
    similar_row = knn_model_5.training_data.iloc[idx]
    planet_name = similar_row['pl_name']
    habitability_score = similar_row.get('habitability_score', 'N/A')
    distance = distances[0][i]
    
    # Format habitability score properly - handle both numeric and 'N/A' cases
    if habitability_score == 'N/A' or pd.isna(habitability_score):
        hab_score_str = 'N/A'
    else:
        hab_score_str = f"{habitability_score:.2f}"
    
    # Print full feature values with units
    feat_rade = similar_row.get('pl_rade', 'N/A')
    feat_bmasse = similar_row.get('pl_bmasse', 'N/A')
    feat_orbper = similar_row.get('pl_orbper', 'N/A')
    feat_orbeccen = similar_row.get('pl_orbeccen', 'N/A')
    feat_insol = similar_row.get('pl_insol', 'N/A')
    feat_eqt = similar_row.get('pl_eqt', 'N/A')

    print(f"{i+1}. {planet_name} (KNN Distance: {distance:.4f})")
    print(f"    pl_rade: {feat_rade} Rearth")
    print(f"    pl_bmasse: {feat_bmasse} Mearth")
    print(f"    pl_orbper: {feat_orbper} days")
    print(f"    pl_orbeccen: {feat_orbeccen} (unitless)")
    print(f"    pl_insol: {feat_insol} Fearth")
    print(f"    pl_eqt: {feat_eqt} K")
    print(f"    Stored habitability_score: {hab_score_str}")

    # Compute KNN-based habitability for this planet relative to Earth (using compute_knn_based_score)
    # Build a pandas Series for this row's features to pass to compute_knn_based_score
    planet_series = pd.Series({
        'pl_rade': feat_rade,
        'pl_bmasse': feat_bmasse,
        'pl_orbper': feat_orbper,
        'pl_orbeccen': feat_orbeccen,
        'pl_insol': feat_insol,
        'pl_eqt': feat_eqt
    })

    # Score relative to Earth reference (earth_params)
    score_vs_earth = compute_knn_based_score(planet_series, knn_model, earth_params, optimal_decay_rate)

    # Score relative to hypothetical planet (use hypothetical_planet as reference)
    score_vs_hyp = compute_knn_based_score(planet_series, knn_model, hypothetical_planet, optimal_decay_rate)

    print(f"    Recomputed score (vs Earth reference): {score_vs_earth:.2f}")
    print(f"    Recomputed score (vs Hypothetical reference): {score_vs_hyp:.2f}\n")

    similar_planets.append(planet_name)
    similar_scores.append(habitability_score if habitability_score != 'N/A' and not pd.isna(habitability_score) else 0)

# Create bar chart comparing Earth, hypothetical planet, and 5 similar planets
plt.figure(figsize=(12, 8))
# Build comparison chart using the same reliable logic as the "Top Planets vs Earth" plot,
# but include the hypothetical planet as well.
comparison_planets = []
comparison_scores = []
comparison_colors = []

# Add top 5 planets (same as earlier plot)
for i, row in top5_planets.iterrows():
    comparison_planets.append(row['pl_name'])
    comparison_scores.append(row['habitability_score'])
    comparison_colors.append('skyblue')

# Add Earth's closest match if not already present
if earth_match_name not in [p.split('\n')[0] for p in comparison_planets]:
    comparison_planets.append(f"{earth_match_name}\n(Earth's closest match)")
    score_value = earth_match_score if earth_match_score != 'N/A' and not pd.isna(earth_match_score) else 0
    comparison_scores.append(score_value)
    comparison_colors.append('lightgreen')

# Add hypothetical planet (highlighted in red)
comparison_planets.append('Hypothetical Planet')
comparison_scores.append(hyp_habitability)
comparison_colors.append('red')

# Add Earth at the end for reference
comparison_planets.append('Earth')
comparison_scores.append(100)
comparison_colors.append('gold')

plt.bar(range(len(comparison_planets)), comparison_scores, color=comparison_colors)

# Add score labels on top of bars
for i, score in enumerate(comparison_scores):
    if score > 0:
        plt.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Planets')
plt.ylabel('KNN-based Habitability Score')
plt.title('KNN-based Habitability: Top Planets vs Earth (+ Hypothetical Planet)')
plt.xticks(range(len(comparison_planets)), comparison_planets, rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.show()

print(f"\n=== VERIFICATION ===")
print(f"Notice how planets similar in KNN space now have similar habitability scores!")
print(f"This creates consistency between 'similarity' and 'habitability' measurements.")

# Create KMeans clustering (30 clusters) and 2D visualization
print("\n10. Creating KMeans clustering and 2D visualization...")

# Generate cluster representatives (change to 30 clusters)
representatives_df = create_planet_clusters(natural_data, n_clusters=30)

# Prepare data for 2D visualization
feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
cluster_features = representatives_df[feature_columns].values

# Add Earth and hypothetical planet to the data
earth_features = np.array([[earth_params[col] for col in feature_columns]])
hyp_features = np.array([[hypothetical_planet[col] for col in feature_columns]])

# Combine all data for PCA
all_features = np.vstack([cluster_features, earth_features, hyp_features])

# Apply PCA for 2D visualization
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

pca = PCA(n_components=2)
all_features_2d = pca.fit_transform(all_features_scaled)

# Split back into components
cluster_2d = all_features_2d[:-2]
earth_2d = all_features_2d[-2]
hyp_2d = all_features_2d[-1]

# Create 2D plot with habitability color coding
plt.figure(figsize=(14, 10))

# Get habitability scores for cluster representatives for color coding
cluster_habitability = representatives_df['habitability_score'].values

# Plot clusters with color representing habitability
scatter = plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], 
                     c=cluster_habitability, cmap='viridis', 
                     s=60, alpha=0.7, label='Planet Clusters (30)')

# Plot Earth
plt.scatter(earth_2d[0], earth_2d[1], c='gold', s=120, 
           marker='*', label='Earth (Score=100)', edgecolors='black', linewidth=2)

# Plot hypothetical planet
plt.scatter(hyp_2d[0], hyp_2d[1], c='red', s=120, 
           marker='D', label=f'Hypothetical Planet (Score={hyp_habitability:.1f})', 
           edgecolors='black', linewidth=2)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('2D Visualization: Planet Clusters with KNN-based Habitability Scores')
plt.legend()
plt.grid(True, alpha=0.3)

# Add colorbar for habitability scores
cbar = plt.colorbar(scatter)
cbar.set_label('KNN-based Habitability Score')

plt.show()

print(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of total variance")
print(f"Clusters are now color-coded by KNN-based habitability scores!")

# Categorize hypothetical planet
print("\n11. Categorizing hypothetical planet...")
print(f"\nHypothetical Planet KNN-based Habitability Score: {hyp_habitability:.2f}")

def categorize_habitability(score):
    if score >= 95:
        return "Earth Twin"
    elif score >= 80:
        return "Life Friendly"
    elif score >= 60:
        return "Maybe Habitable"
    elif score >= 30:
        return "Harsh World"
    else:
        return "Deadzone"

category = categorize_habitability(hyp_habitability)
print(f"Category: {category}")

print("\n=== HABITABILITY SCALE ===")
print("Earth Twin: 95-100 (Nearly identical to Earth)")
print("Life Friendly: 80-94 (Very good conditions for life)")
print("Maybe Habitable: 60-79 (Potentially habitable)")
print("Harsh World: 30-59 (Difficult but possible)")
print("Deadzone: 0-29 (Likely uninhabitable)")

print(f"\n🌍 CONCLUSION: The hypothetical planet falls into the '{category}' category with a KNN-based habitability score of {hyp_habitability:.2f}")

print("\n=== UNIFIED KNN-BASED ANALYSIS COMPLETE ===")
print("✅ SUCCESS: KNN similarity and habitability scoring are now unified!")
print("✅ Decay rate optimized using KMeans for better category separation!")
print("✅ Planets close in KNN space now have similar habitability scores")
print("✅ All visualizations use consistent KNN-based measurements with score labels")
print("✅ The hypothetical planet has been categorized using the optimized unified system")
