import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_planet_clusters(data, n_clusters=100):
    """
    Create K-means clusters and return representative points
    
    Args:
        data: pandas DataFrame with planet data
        n_clusters: number of clusters to create (default 100)
    
    Returns:
        DataFrame with cluster representative planets
    """
    feature_columns = [
        'pl_rade',      # planet radius [Earth Radius]
        'pl_bmasse',    # planet mass [Earth Mass]
        'pl_orbper',    # orbital period [days]
        'pl_orbeccen',  # orbital eccentricity
        'pl_insol',     # insolation flux [Earth Flux]
        'pl_eqt'        # equilibrium temperature [K]
    ]
    
    # Remove rows with missing values
    clean_data = data[feature_columns + ['pl_name', 'habitability_score']].dropna()
    
    # use a plain numpy array so the scaler does not record feature names
    X = clean_data[feature_columns].values
    
    # Scale features and perform clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    clean_data = clean_data.copy()
    clean_data['cluster'] = cluster_labels
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Find representative planet for each cluster (closest to center)
    representatives = []
    
    for cluster_id in range(n_clusters):
        cluster_planets = clean_data[clean_data['cluster'] == cluster_id]
        
        if len(cluster_planets) == 0:
            continue
            
        cluster_center = cluster_centers[cluster_id]
        
        # Find closest planet to center
        distances = []
        for idx, planet in cluster_planets.iterrows():
            planet_features = planet[feature_columns].values
            planet_scaled = scaler.transform([planet_features])[0]
            center_scaled = scaler.transform([cluster_center])[0]
            distance = np.linalg.norm(planet_scaled - center_scaled)
            distances.append(distance)
        
        closest_idx = np.argmin(distances)
        representative = cluster_planets.iloc[closest_idx]
        representatives.append(representative)
    
    representatives_df = pd.DataFrame(representatives)
    
    # Save to CSV
    representatives_df.to_csv('planet_cluster_representatives.csv', index=False)
    
    print(f"Created {len(representatives_df)} cluster representatives from {len(clean_data)} planets")
    print(f"Saved to: planet_cluster_representatives.csv")
    
    return representatives_df