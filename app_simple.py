# Simple version of the Exoplanet Matcher app for debugging
from backend_api import get_matches
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Exoplanet Matcher",
    page_icon="🌌",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def main():
    st.title("🌌 Exoplanet Matcher")
    st.subheader("Discover planets similar to your specifications")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home"):
            st.session_state.current_page = 'home'
            st.rerun()
    with col2:
        if st.button("📝 Input"):
            st.session_state.current_page = 'input'
            st.rerun()
    with col3:
        if st.button("🔍 Match"):
            st.session_state.current_page = 'match'
            st.rerun()
    
    st.write(f"Current page: {st.session_state.current_page}")
    
    if st.session_state.current_page == 'home':
        st.markdown("""
        Welcome to the Exoplanet Matcher! This tool helps you find exoplanets that are similar 
        to a planet you describe.
        """)
        
        if st.button("🚀 Start Matching"):
            st.session_state.current_page = 'input'
            st.rerun()
    
    elif st.session_state.current_page == 'input':
        st.markdown("## Enter Planet Parameters")
        
        name = st.text_input("Planet Name", value="My Planet")
        radius = st.slider("Planet Radius (Earth radii)", 0.1, 5.0, 1.0)
        mass = st.slider("Planet Mass (Earth masses)", 0.1, 20.0, 1.0)
        temp = st.number_input("Temperature (K)", 100, 1200, 255)
        period = st.number_input("Orbital Period (days)", 1, 2000, 365)
        
        if st.button("🔍 Find Matches"):
            payload = {
                "planet_name": name,
                "pl_rade": radius,
                "pl_bmasse": mass,
                "pl_eqt": temp,
                "pl_orbper": period
            }
            
            try:
                matches = get_matches(payload)
                st.session_state.matches = matches
                st.session_state.current_page = 'match'
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif st.session_state.current_page == 'match':
        st.markdown("## Matching Exoplanets")
        
        if 'matches' in st.session_state and st.session_state.matches:
            for i, planet in enumerate(st.session_state.matches['similar'], 1):
                with st.expander(f"{i}. {planet['pl_name']}"):
                    st.write(f"**Radius:** {planet['pl_rade']:.2f} Earth radii")
                    st.write(f"**Mass:** {planet['pl_bmasse']:.2f} Earth masses")
                    st.write(f"**Temperature:** {planet['pl_eqt']:.0f}K")
                    st.write(f"**Orbital Period:** {planet['pl_orbper']:.1f} days")
                    st.write(f"**Similarity Score:** {planet['similarity_score']:.3f}")
                    st.write(f"**Habitability Score:** {planet['habitability_score']:.3f}")
        else:
            st.warning("No matches found. Please go to Input page to search.")

if __name__ == "__main__":
    main()
