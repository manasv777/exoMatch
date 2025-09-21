"""
Backend API module for the Exoplanet Matcher frontend.
This module provides the get_matches function that the frontend calls.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Any
import warnings
# Try to import matplotlib, but make it optional for deployment
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    matplotlib = None
warnings.filterwarnings('ignore')

# Import our custom modules
from habitability import compute_knn_based_score, optimize_decay_rate_with_kmeans, compute_knn_scores_for_dataframe, natural_data
from knn_model import train_planet_knn, PlanetSimilarityKNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Global variables to store trained models
_knn_model = None
_earth_params = None
_optimal_decay_rate = None
_training_data = None

def _initialize_models():
    """Initialize the KNN model and other required components."""
    global _knn_model, _earth_params, _optimal_decay_rate, _training_data
    
    if _knn_model is not None:
        return  # Already initialized
    
    print("Initializing backend models...")
    
    # Define Earth's parameters
    _earth_params = {
        'pl_rade': 1.0,      # Earth radius
        'pl_bmasse': 1.0,    # Earth mass
        'pl_orbper': 365.25, # Earth orbital period
        'pl_orbeccen': 0.017, # Earth orbital eccentricity
        'pl_insol': 1.0,     # Earth insolation flux
        'pl_eqt': 255        # Earth equilibrium temperature
    }
    
    # Try to load existing model, otherwise train new one
    try:
        # Load the existing model (trained with n_neighbors=1)
        existing_model = PlanetSimilarityKNN(n_neighbors=1)
        existing_model.load_model('planet_similarity_knn_model.pkl')
        
        # Create a new model with 5 neighbors for API use
        _knn_model = PlanetSimilarityKNN(n_neighbors=5)
        _knn_model.train_model(natural_data, existing_model.feature_columns)
        print("Loaded existing KNN model and created 5-neighbor version")
    except:
        print("Training new KNN model...")
        _knn_model = train_planet_knn(natural_data)
    
    # Optimize decay rate for better category separation
    print("Optimizing decay rate...")
    _optimal_decay_rate = optimize_decay_rate_with_kmeans(_knn_model, _earth_params, natural_data, n_categories=5)
    
    # Store training data
    _training_data = natural_data.copy()
    
    print("Backend models initialized successfully!")

def generate_scatterplot(user_planet_params: Dict[str, Any]) -> str:
    """
    Generate a 2D PCA scatterplot showing planet clusters with habitability scores.
    Returns the plot as a base64 encoded string for display in Streamlit.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping scatterplot generation")
        return None
    
    try:
        # Use a subset of natural data for clustering (sample for performance)
        sample_size = min(1000, len(natural_data))
        sample_data = natural_data.sample(n=sample_size, random_state=42)
        
        # Prepare data for clustering
        feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
        clean_data = sample_data[feature_columns + ['pl_name']].dropna()
        
        # Compute habitability scores for the sample
        habitability_scores = []
        for _, row in clean_data.iterrows():
            planet_params = {
                'pl_rade': row['pl_rade'],
                'pl_bmasse': row['pl_bmasse'],
                'pl_orbper': row['pl_orbper'],
                'pl_orbeccen': row['pl_orbeccen'],
                'pl_insol': row['pl_insol'],
                'pl_eqt': row['pl_eqt']
            }
            score = compute_knn_based_score(planet_params, _knn_model, _earth_params, _optimal_decay_rate)
            habitability_scores.append(score)
        
        # Add habitability scores to the dataframe
        clean_data = clean_data.copy()
        clean_data['habitability_score'] = habitability_scores
        
        # Prepare features for PCA
        cluster_features = clean_data[feature_columns].values
        
        # Add Earth and user planet to the data
        earth_features = np.array([[1.0, 1.0, 365.25, 0.017, 1.0, 255]])
        user_features = np.array([[
            user_planet_params.get('pl_rade', 1.0),
            user_planet_params.get('pl_bmasse', 1.0),
            user_planet_params.get('pl_orbper', 365.0),
            0.017,  # Earth's eccentricity as default
            1.0,    # Earth's insolation as default
            user_planet_params.get('pl_eqt', 255.0)
        ]])
        
        # Combine all data for PCA
        all_features = np.vstack([cluster_features, earth_features, user_features])
        
        # Apply PCA for 2D visualization
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)
        
        pca = PCA(n_components=2)
        all_features_2d = pca.fit_transform(all_features_scaled)
        
        # Split back into components
        cluster_2d = all_features_2d[:-2]
        earth_2d = all_features_2d[-2]
        user_2d = all_features_2d[-1]
        
        # Create 2D plot with habitability color coding
        plt.figure(figsize=(12, 8))
        
        # Get habitability scores for color coding
        cluster_habitability = clean_data['habitability_score'].values
        
        # Plot clusters with color representing habitability
        scatter = plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], 
                             c=cluster_habitability, cmap='viridis', 
                             s=60, alpha=0.7, label=f'Planet Sample ({len(cluster_2d)})')
        
        # Plot Earth
        plt.scatter(earth_2d[0], earth_2d[1], c='gold', s=120, 
                   marker='*', label='Earth (Score=100)', edgecolors='black', linewidth=2)
        
        # Plot user planet
        plt.scatter(user_2d[0], user_2d[1], c='red', s=120, 
                   marker='D', label='Your Planet', edgecolors='black', linewidth=2)
        
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('2D Visualization: Planet Sample with Habitability Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add colorbar for habitability scores
        cbar = plt.colorbar(scatter)
        cbar.set_label('KNN-based Habitability Score')
        
        # Convert plot to base64 string
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"Error generating scatterplot: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_matches(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find exoplanets similar to the given planet parameters.
    
    Args:
        payload: Dictionary containing planet parameters:
            - planet_name: str
            - pl_rade: float (Earth radii)
            - pl_bmasse: float (Earth masses) 
            - pl_eqt: float (Kelvin)
            - pl_orbper: float (days)
    
    Returns:
        Dictionary with:
            - similar: List of up to 5 similar planets with scores
            - meta: Model metadata
    """
    # Initialize models if needed
    _initialize_models()
    
    # Extract parameters from payload
    planet_name = payload.get('planet_name', 'Unknown Planet')
    pl_rade = payload.get('pl_rade', 1.0)
    pl_bmasse = payload.get('pl_bmasse', 1.0)
    pl_eqt = payload.get('pl_eqt', 255.0)
    pl_orbper = payload.get('pl_orbper', 365.0)
    
    # Create test planet parameters (using Earth defaults for missing features)
    test_planet = {
        'pl_rade': pl_rade,
        'pl_bmasse': pl_bmasse,
        'pl_orbper': pl_orbper,
        'pl_orbeccen': 0.017,  # Earth's eccentricity as default
        'pl_insol': 1.0,       # Earth's insolation as default
        'pl_eqt': pl_eqt
    }
    
    # Find similar planets using KNN
    feature_columns = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
    
    # Prepare test point
    test_point = [test_planet[col] for col in feature_columns]
    test_point_scaled = _knn_model.scaler.transform([test_point])
    
    # Find 5 most similar planets
    distances, indices = _knn_model.knn_model.kneighbors(test_point_scaled)
    
    similar_planets = []
    
    for i in range(min(5, len(indices[0]))):
        idx = indices[0][i]
        similar_row = _knn_model.training_data.iloc[idx]
        distance = distances[0][i]
        
        # Get planet data
        planet_data = {
            'pl_name': similar_row.get('pl_name', 'Unknown'),
            'pl_rade': float(similar_row.get('pl_rade', 0)),
            'pl_bmasse': float(similar_row.get('pl_bmasse', 0)),
            'pl_eqt': float(similar_row.get('pl_eqt', 0)),
            'pl_orbper': float(similar_row.get('pl_orbper', 0)),
            'similarity_score': float(distance),  # Use raw KNN distance (lower = better)
            'habitability_score': 0.0  # Will calculate below
        }
        
        # Calculate habitability score
        try:
            habitability = compute_knn_based_score(
                similar_row, 
                _knn_model, 
                _earth_params, 
                _optimal_decay_rate
            )
            planet_data['habitability_score'] = float(habitability / 100.0)  # Convert to 0-1 scale
        except:
            planet_data['habitability_score'] = 0.0
        
        similar_planets.append(planet_data)
    
    # Sort by similarity score (lower is better)
    similar_planets.sort(key=lambda x: x['similarity_score'])
    
    # Use raw KNN distances as similarity scores (lower = better)
    # No normalization needed - the original code used raw distances
    
    # Generate scatterplot
    scatterplot_data = generate_scatterplot(test_planet)
    
    return {
        'similar': similar_planets,
        'scatterplot': scatterplot_data,
        'meta': {
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
    }

# Initialize models when module is imported
try:
    _initialize_models()
except Exception as e:
    print(f"Warning: Could not initialize models on import: {e}")
    print("Models will be initialized on first get_matches call.")

