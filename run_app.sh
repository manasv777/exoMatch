#!/bin/bash

# Script to run the Exoplanet Matcher Streamlit app
# This activates the virtual environment and starts the app

echo "🌌 Starting Exoplanet Matcher App..."
echo "Activating virtual environment..."

# Activate virtual environment
source venv/bin/activate

echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

# Run the Streamlit app
streamlit run app.py --server.port 8501

