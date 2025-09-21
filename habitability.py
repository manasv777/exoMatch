import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# The file contains a NASA-style commented header (lines starting with '#').
# Use engine='python' and comment='#' so pandas skips those lines and
# parses the actual CSV header correctly.

csv_path = 'PS_2025.09.20_19.26.05.csv'
try:
    natural_data = pd.read_csv(csv_path, comment='#', engine='python')
except Exception:
	# fallback: show a helpful diagnostic and try a permissive read
	print('Initial read failed — attempting permissive read to locate bad lines')
	natural_data = pd.read_csv(csv_path, comment='#', engine='python', on_bad_lines='warn')

#print(natural_data.columns)

# --- KNN-based habitability score calculator with optimized decay rate ---
def compute_knn_based_score(row, knn_model=None, earth_params=None, decay_rate=0.1):
    """
    Calculate habitability score based on KNN distance to Earth
    Lower KNN distance = Higher habitability score
    """
    if knn_model is None or earth_params is None:
        return np.nan
    
    # Extract the same features used in KNN model
    feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
    
    # Check if all required features are available
    planet_features = {}
    for feature in feature_columns:
        value = row.get(feature)
        if pd.isna(value):
            return np.nan  # Cannot compute score without all features
        planet_features[feature] = value
    
    try:
        # Calculate KNN distance to Earth
        test_point = [planet_features[col] for col in feature_columns]
        earth_point = [earth_params[col] for col in feature_columns]
        
        # Scale both points using the same scaler as the KNN model
        test_point_scaled = knn_model.scaler.transform([test_point])[0]
        earth_point_scaled = knn_model.scaler.transform([earth_point])[0]
        
        # Calculate Euclidean distance in scaled space
        distance = np.linalg.norm(test_point_scaled - earth_point_scaled)
        
        # Convert distance to similarity score (0-100)
        # Use exponential decay: closer planets get higher scores
        # decay_rate is now optimized for better category separation
        score = 100 * np.exp(-distance * decay_rate)
        
        return score
        
    except Exception as e:
        return np.nan

def optimize_decay_rate_with_kmeans(knn_model, earth_params, natural_data, n_categories=5):
    """
    Optimize decay rate to create well-separated habitability categories using KMeans
    
    Args:
        knn_model: trained KNN model
        earth_params: Earth's parameters
        natural_data: planet dataset
        n_categories: number of habitability categories (default 5)
    
    Returns:
        optimized_decay_rate: float value that maximizes category separation
    """
    print("Optimizing decay rate for better category separation...")
    
    # Get clean data for optimization
    feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
    clean_data = natural_data[feature_columns + ['pl_name']].dropna()
    
    # Calculate distances for all planets
    distances = []
    earth_point = [earth_params[col] for col in feature_columns]
    earth_point_scaled = knn_model.scaler.transform([earth_point])[0]
    
    for _, row in clean_data.iterrows():
        test_point = [row[col] for col in feature_columns]
        test_point_scaled = knn_model.scaler.transform([test_point])[0]
        distance = np.linalg.norm(test_point_scaled - earth_point_scaled)
        distances.append(distance)
    
    distances = np.array(distances)
    
    # Test different decay rates
    decay_rates = np.linspace(0.05, 1.0, 20)  # Test 20 different decay rates
    best_decay_rate = 0.1
    best_silhouette = -1
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    for decay_rate in decay_rates:
        # Calculate scores with this decay rate
        scores = 100 * np.exp(-distances * decay_rate)
        
        # Reshape for KMeans (needs 2D array)
        scores_reshaped = scores.reshape(-1, 1)
        
        # Apply KMeans to create categories
        kmeans = KMeans(n_clusters=n_categories, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scores_reshaped)
        
        # Calculate silhouette score (measure of cluster separation)
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette score
            silhouette = silhouette_score(scores_reshaped, labels)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_decay_rate = decay_rate
    
    print(f"Optimized decay rate: {best_decay_rate:.3f}")
    print(f"Best silhouette score: {best_silhouette:.3f}")
    
    return best_decay_rate

# --- Original Gaussian similarity function ---
def similarity(x, x0, sigma):
    if pd.isna(x):
        return np.nan
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# --- Original composite score calculator for comparison ---
def compute_score(row):
    """
    Calculate habitability score based on the same features used in KNN model:
    pl_rade, pl_bmasse, pl_orbper, pl_orbeccen, pl_insol, pl_eqt
    """
    
    # Planet radius similarity (Earth = 1.0)
    S_rade = similarity(row.get("pl_rade"), 1.0, 0.3)
    
    # Planet mass similarity (Earth = 1.0)
    S_bmasse = similarity(row.get("pl_bmasse"), 1.0, 0.5)
    
    # Orbital period similarity (Earth = 365.25 days)
    S_orbper = similarity(row.get("pl_orbper"), 365.25, 100)
    
    # Orbital eccentricity similarity (Earth = 0.017, lower is better)
    S_orbeccen = np.nan
    if not pd.isna(row.get("pl_orbeccen")):
        try:
            # Lower eccentricity is better for habitability
            S_orbeccen = np.exp(- (float(row.get("pl_orbeccen")) ** 2) / (2 * 0.1 ** 2))
        except Exception:
            S_orbeccen = np.nan
    
    # Insolation flux similarity (Earth = 1.0)
    S_insol = similarity(row.get("pl_insol"), 1.0, 0.5)
    
    # Equilibrium temperature similarity (Earth = 255K)
    S_eqt = similarity(row.get("pl_eqt"), 255, 50)
    
    # Define weights for each feature (must sum to 1.0)
    weights = {
        "rade": 0.20,      # Planet radius
        "bmasse": 0.20,    # Planet mass
        "orbper": 0.15,    # Orbital period
        "orbeccen": 0.15,  # Orbital eccentricity
        "insol": 0.15,     # Insolation flux
        "eqt": 0.15        # Equilibrium temperature
    }
    
    # Collect all feature scores
    feature_scores = {
        "rade": S_rade,
        "bmasse": S_bmasse,
        "orbper": S_orbper,
        "orbeccen": S_orbeccen,
        "insol": S_insol,
        "eqt": S_eqt
    }
    
    # Keep only available features
    available = {k: v for k, v in feature_scores.items() if not pd.isna(v)}
    
    # Require at least 3 out of 6 features to compute a score
    if len(available) < 3:
        return np.nan
    
    # Normalize weights to available features
    W = sum(weights[k] for k in available.keys())
    if W == 0:
        return np.nan
    
    # Calculate weighted average score
    score = sum(weights[k] * available[k] for k in available.keys()) / W
    return score * 100
    

def compute_knn_scores_for_dataframe(df, knn_model, earth_params, decay_rate):
    """
    Compute KNN-based habitability scores for all rows in `df` and return a Series.
    Rows with missing features will receive NaN.
    """
    feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
    scores = []
    for _, row in df.iterrows():
        try:
            score = compute_knn_based_score(row, knn_model=knn_model, earth_params=earth_params, decay_rate=decay_rate)
        except Exception:
            score = np.nan
        scores.append(score)
    return pd.Series(scores, index=df.index)



     
