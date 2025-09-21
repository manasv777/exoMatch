import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle

class PlanetSimilarityKNN:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.training_data = None
        
    def train_model(self, data, feature_columns):
        """
        Train the KNN model on the provided data
        
        Args:
            data: pandas DataFrame with training data
            feature_columns: list of column names to use as features
        """
        self.feature_columns = feature_columns
        
        # Remove rows with missing values in features
        clean_data = data[feature_columns + ['pl_name']].dropna()
        
        # use numpy array to avoid scaler storing feature names
        X = clean_data[feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.knn_model.fit(X_scaled)
        
        # Store training data for similarity lookup
        self.training_data = clean_data.copy()
        
        print(f"Model trained on {len(clean_data)} samples")
        print(f"Features used: {feature_columns}")
        
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        model_data = {
            'knn_model': self.knn_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_data': self.training_data,
            'n_neighbors': self.n_neighbors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a previously trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.knn_model = model_data['knn_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.training_data = model_data['training_data']
        self.n_neighbors = model_data['n_neighbors']
        
        print(f"Model loaded from {filepath}")
    
    def find_most_similar(self, test_values):
        """
        Find the most similar planet in training data to the test values
        
        Args:
            test_values: dict with feature names as keys and values as the test point
        
        Returns:
            dict with:
                - 'most_similar_row': the most similar training data row
                - 'distance': distance to most similar point
        """
        if self.knn_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Convert test_values to the right format
        test_point = [test_values[col] for col in self.feature_columns]
        
        # Check for missing values
        if any(pd.isna(val) for val in test_point):
            raise ValueError("Test point contains missing values")
        
        # Scale the test point
        test_point_scaled = self.scaler.transform([test_point])
        
        # Find the most similar point (nearest neighbor)
        distances, indices = self.knn_model.kneighbors(test_point_scaled)
        
        most_similar_idx = indices[0][0]
        distance = distances[0][0]
        
        # Get the original training data row
        most_similar_row = self.training_data.iloc[most_similar_idx]
        
        return {
            'most_similar_row': most_similar_row,
            'distance': distance
        }

# Training function
def train_planet_knn(natural_data):
    """Train KNN model on exoplanet data"""
    
    # Define features to use
    feature_columns = [
        'pl_rade',      # planet radius [Earth Radius]
        'pl_bmasse',    # planet mass [Earth Mass]
        'pl_orbper',    # orbital period [days]
        'pl_orbeccen',  # orbital eccentricity
        'pl_insol',     # insolation flux [Earth Flux]
        'pl_eqt'        # equilibrium temperature [K]
    ]
    
    # Initialize and train model
    knn_model = PlanetSimilarityKNN(n_neighbors=1)
    knn_model.train_model(natural_data, feature_columns)
    
    # Save the model
    knn_model.save_model('planet_similarity_knn_model.pkl')
    
    return knn_model

# Testing function
def find_similar_planet(knn_model, test_planet_params):
    """
    Find the most similar planet to the test parameters
    
    Args:
        knn_model: trained PlanetSimilarityKNN instance
        test_planet_params: dict with planet parameters
    """
    result = knn_model.find_most_similar(test_planet_params)
    
    print("=== Most Similar Exoplanet ===")
    print(f"Planet Name: {result['most_similar_row'].get('pl_name', 'Unknown')}")
    print(f"Distance: {result['distance']:.4f}")
    print("\nSimilar Planet Details:")
    # Only show the features that are actually available in the training data
    available_features = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_insol', 'pl_eqt']
    feature_subset = []
    for feature in available_features:
        if feature in result['most_similar_row'].index:
            feature_subset.append(feature)
    
    if feature_subset:
        print(result['most_similar_row'][feature_subset].to_string())
    else:
        print("Feature details not available")
    
    return result