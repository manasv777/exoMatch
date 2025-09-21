# exoMatch

https://exomatch.streamlit.app/

**Find Your Planet’s Cosmic Twin**  
exoMatch is a Streamlit app that lets you design a hypothetical planet and discover which known exoplanets are most similar — powered by NASA’s Exoplanet Archive data and our custom similarity + habitability scoring models.

## Overview

- **Interactive Input** – Adjust key planetary features with intuitive sliders:
  - Radius (Earth radii)
  - Mass (Earth masses)
  - Equilibrium Temperature (K)
  - Orbital Period (days)

- **Dynamic Visualization** – A responsive planet graphic updates live:
  - Size = Radius  
  - Core darkness = Mass  
  - Color gradient = Temperature  
  - Orbital speed lines = Orbital Period  

- **Smart Matching** – Our backend models use KNN + clustering to:
  - Find the **Top-5 most similar exoplanets** to your design  
  - Assign each a **Habitability Score** (0–1, higher = more Earth-like)

- **Educational & Engaging** – Makes raw exoplanet data accessible to everyone through an intuitive, space-themed interface.

## Project Structure

├── app.py # Streamlit frontend
├── main.py # Backend entrypoint
├── knn_model.py # KNN similarity model
├── kmeans_model.py # KMeans clustering for habitability
├── habitability.py # Habitability scoring logic
├── planet_cluster_representatives.csv
├── PS_2025.09.20_19.26.05.csv # NASA Exoplanet Archive data
├── PS_top100_by_habitability.csv
├── PS_top100_by_knn_habitability.csv
└── README.md

## Installation

Clone the repo and install dependencies:


git clone https://github.com/manasv777/exoMatch.git
cd exoMatch
pip install -r requirements.txt
Requirements: Python 3.10+ recommended. Streamlit, pandas, scikit-learn are required.

Usage
Run the Streamlit app:

```bash
Copy code
streamlit run app.py
Then open the provided local URL in your browser.
```
Data Sources
NASA Exoplanet Archive – Planetary Systems

Derived CSVs included in this repo (cleaned + feature-selected).

Hackathon Context
exoMatch was built for the Carolina Data Challenge 2025 under the Natural Sciences track.
Our goal: turn complex exoplanet datasets into an interactive tool that inspires curiosity about space exploration and highlights why studying exoplanets matters.

Future Work
Expand feature set beyond 4 core parameters

Incorporate stellar properties into similarity scoring

Deploy as a public web app (e.g., Streamlit Cloud)

Team:
Sai Yadavalli,
Sidharth Yeramaddu,
Dhanvi Movva,
Manas Vellaturi

