import streamlit as st
import pandas as pd
from typing import Dict, Tuple, Any, List

# FRONT END ONLY — do not modify backend files.
# Use the existing backend in backend_api.py. If your module name differs, change only the import line below.
from backend_api import get_matches  # <-- keep as-is unless your backend entrypoint file is different
try:
    from backend_api import get_feature_ranges  # optional helper; if absent we'll fallback to CSV
except Exception:
    get_feature_ranges = None

# CSV fallback (only used if get_feature_ranges is not available)
DATA_CSV_PATH = "PS_2025.09.20_19.26.05.csv"  # update if your file name/path differs

@st.cache_data(show_spinner=False)
def load_feature_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Try backend get_feature_ranges(); else compute from CSV.
    Returns dict with keys: pl_rade, pl_bmasse, pl_eqt, pl_orbper.
    Pads the ranges slightly to avoid slider clipping.
    """
    def pad(lo, hi, pct=0.02):
        span = hi - lo
        return float(round(lo - pct*span, 3)), float(round(hi + pct*span, 3))

    # 1) Prefer backend
    if callable(get_feature_ranges):
        try:
            rng = get_feature_ranges()
            out = {}
            for k in ["pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper"]:
                if k in rng and isinstance(rng[k], (list, tuple)) and len(rng[k]) == 2:
                    lo, hi = float(rng[k][0]), float(rng[k][1])
                    out[k] = pad(lo, hi)
            if out:
                return out
        except Exception:
            pass

    # 2) Fallback to CSV
    try:
        df = pd.read_csv(DATA_CSV_PATH, comment="#")
        if "default_flag" in df.columns:
            df = df[df["default_flag"] == 1]
        cols = ["pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper"]
        out = {}
        for c in cols:
            if c in df.columns:
                series = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(series):
                    lo, hi = float(series.min()), float(series.max())
                    out[c] = pad(lo, hi)
        # Sensible defaults if anything missing:
        out.setdefault("pl_rade", (0.1, 5.0))
        out.setdefault("pl_bmasse", (0.1, 20.0))
        out.setdefault("pl_eqt", (100.0, 1200.0))
        out.setdefault("pl_orbper", (1.0, 2000.0))
        return out
    except Exception:
        return {
            "pl_rade": (0.1, 5.0),
            "pl_bmasse": (0.1, 20.0),
            "pl_eqt": (100.0, 1200.0),
            "pl_orbper": (1.0, 2000.0),
        }

# Configure Streamlit page
st.set_page_config(
    page_title="Exoplanet Matcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'matches' not in st.session_state:
    st.session_state.matches = None
if 'planet_inputs' not in st.session_state:
    st.session_state.planet_inputs = {
        'name': '',
        'radius': 1.0,
        'mass': 1.0,
        'temp': 255,
        'period': 365
    }

def render_starfield_css() -> str:
    """
    Generate CSS for dark space theme with animated starfield background.
    Uses @st.cache_data for performance.
    """
    return """
    <style>
    /* Simple dark theme without complex animations */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Dark background */
    body {
        background: #000000;
        color: white;
    }
    
    /* Ensure content is visible */
    .stApp {
        background: #000000;
    }
    
    .main .block-container {
        background: #000000;
    }
    
    /* Make all text white for readability */
    .stMarkdown, .stText, .stSelectbox, .stSlider, .stNumberInput, .stTextInput {
        color: white !important;
    }
    
    /* Ensure Streamlit elements are visible on black background */
    .stSelectbox > div > div {
        background-color: #333333;
        color: white;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .stNumberInput > div > div > input {
        background: #333333;
        border: 1px solid #666666;
        color: white;
    }
    
    .stTextInput > div > div > input {
        background: #333333;
        border: 1px solid #666666;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        text-transform: none;
        letter-spacing: 0.5px;
        min-height: 48px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:disabled {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        color: #a0aec0;
        transform: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        cursor: not-allowed;
    }
    
    /* Card styling */
    .planet-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Planet visualization container */
    .planet-viz-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .planet-circle {
        border-radius: 50%;
        position: relative;
        box-shadow: 
            0 0 20px rgba(255, 255, 255, 0.3),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    /* Score bars */
    .score-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        margin: 0 0.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .similarity-fill {
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
    }
    
    .habitability-fill {
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
    }
    
    /* Validation messages */
    .validation-error {
        color: #ff6b6b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: rgba(255, 107, 107, 0.1);
        border-radius: 8px;
        border-left: 3px solid #ff6b6b;
    }
    </style>
    """

def render_navbar(active: str) -> None:
    """
    Render the top navigation bar with three page buttons.
    Highlights the active page.
    """
    # Navigation buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("Home", key="nav_home", help="Return to the main page"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col2:
        if st.button("Input", key="nav_input", help="Enter planet parameters"):
            st.session_state.current_page = 'input'
            st.rerun()
    
    with col3:
        if st.button("Match", key="nav_match", help="View matching planets"):
            st.session_state.current_page = 'match'
            st.rerun()

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def adaptive_step(lo: float, hi: float, target_ticks: int = 200) -> float:
    span = abs(hi - lo)
    if span <= 0: 
        return 0.1
    raw = span / target_ticks
    # round step to a friendly increment
    if span < 5:   return max(0.01, round(raw, 2))
    if span < 20:  return max(0.1, round(raw, 1))
    if span < 200: return max(1.0, round(raw))
    if span < 1000:return max(5.0, round(raw/5)*5)
    return max(10.0, round(raw/10)*10)

def compute_visual_styles(radius: float, mass: float, temp: float, ranges: Dict[str, Tuple[float,float]], stage_px: int = 640) -> Dict[str, Any]:
    """
    Map inputs to CSS:
    - size (px) from radius, capped to ~90% of stage (which itself should be <= 50vw).
    - surface color from temp (blue->red).
    - core alpha from mass (0.2..0.7).
    """
    r_lo, r_hi = ranges["pl_rade"]
    t_lo, t_hi = ranges["pl_eqt"]
    m_lo, m_hi = ranges["pl_bmasse"]

    # Normalize inputs into 0..1 safely
    r_n = (clamp(radius, r_lo, r_hi) - r_lo) / max(1e-9, (r_hi - r_lo))
    t_n = (clamp(temp,   t_lo, t_hi) - t_lo) / max(1e-9, (t_hi - t_lo))
    m_n = (clamp(mass,   m_lo, m_hi) - m_lo) / max(1e-9, (m_hi - m_lo))

    # Diameter mapping
    max_diam = int(stage_px * 0.9)
    min_diam = int(stage_px * 0.12)
    diameter = int(min_diam + r_n * (max_diam - min_diam))

    # Temperature color mix (blue -> red)
    blue = (70, 120, 255)
    red  = (255, 90, 70)
    mix = tuple(int(blue[i] + t_n * (red[i] - blue[i])) for i in range(3))
    surface_color = f"rgb({mix[0]},{mix[1]},{mix[2]})"

    # Core alpha by mass
    core_alpha = 0.2 + 0.5 * m_n

    return {
        "diameter": diameter,
        "surface_color": surface_color,
        "core_alpha": core_alpha,
    }

def compute_speed_line_length(period_days: float, ranges: Dict[str, Tuple[float,float]]) -> int:
    """
    Longer lines = faster orbit = shorter period.
    Scale inversely within observed period range.
    """
    p_lo, p_hi = ranges["pl_orbper"]
    p = clamp(period_days, p_lo, p_hi)
    inv = 1.0 / p
    inv_lo, inv_hi = 1.0/p_hi, 1.0/p_lo
    inv_n = (inv - inv_lo) / max(1e-9, (inv_hi - inv_lo))
    # Map 0..1 -> pixels
    return int(8 + inv_n * (36 - 8))  # 8..36 px

def planet_html(radius: float, mass: float, temp: float, period: float, ranges: Dict[str, Tuple[float,float]]) -> str:
    """
    Returns a self-contained HTML+CSS string that draws:
    - A planet circle with radial gradient for core darkness and surface color.
    - 8 'speed lines' around it whose length scales with orbital speed (shorter period -> longer lines).
    """
    styles = compute_visual_styles(radius, mass, temp, ranges)
    diam = styles["diameter"]
    core_a = styles["core_alpha"]
    color = styles["surface_color"]
    line_len = compute_speed_line_length(period, ranges)

    # positions for 8 lines (percent offsets around the circle)
    line_positions = [
        ("50%", "-6%","0deg"),
        ("85%","10%","90deg"),
        ("50%","92%","180deg"),
        ("-6%","50%","270deg"),
        ("20%","-2%","45deg"),
        ("92%","30%","120deg"),
        ("78%","96%","210deg"),
        ("-4%","78%","315deg"),
    ]

    return f"""
    <style>
      .stage {{
        width: min(45vw, 640px);
        height: min(45vw, 640px);
        position: relative;
        margin: 0 auto;
      }}
      .planet {{
        width: {diam}px; height: {diam}px;
        border-radius: 50%;
        margin: auto;
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 40px rgba(150,140,255,0.25), inset 0 0 30px rgba(0,0,0,0.25);
        background:
          radial-gradient(circle at 48% 42%,
            rgba(0,0,0,{core_a}) 0%,
            {color} 52%,
            rgba(255,255,255,0.10) 84%);
      }}
      .lines {{
        position: absolute; inset: 0; pointer-events: none;
      }}
      .line {{
        position: absolute;
        height: 3px;
        width: {line_len}px;
        background: rgba(255,255,255,0.7);
        box-shadow: 0 0 8px rgba(255,255,255,0.35);
      }}
    </style>
    <div class="stage">
      <div class="planet"></div>
      <div class="lines">
        {"".join([f'<div class="line" style="top:{t}; left:{l}; transform: rotate({deg});"></div>' for (l,t,deg) in line_positions])}
      </div>
    </div>
    <div style="text-align:center; font-size:0.9rem; opacity:0.8; margin-top:8px; color: white;">
      <em>Size = Radius · Core = Mass · Color = Temperature · Line length = Orbital speed</em>
    </div>
    """

def render_planet(radius: float, mass: float, temp: float, period: float, ranges: Dict[str, Tuple[float,float]]):
    from streamlit.components.v1 import html as st_html
    st_html(planet_html(radius, mass, temp, period, ranges), height=720, scrolling=False)

def inputs_valid(name: str, radius: float, mass: float, temp: float, period: float) -> Tuple[bool, List[str]]:
    """
    Validate all input parameters.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []
    
    if not name or not name.strip():
        errors.append("Planet name is required")
    elif len(name.strip()) < 2:
        errors.append("Planet name must be at least 2 characters")
    
    # Get ranges for validation
    ranges = load_feature_ranges()
    
    r_lo, r_hi = ranges["pl_rade"]
    if not (r_lo <= radius <= r_hi):
        errors.append(f"Planet radius must be between {r_lo} and {r_hi} Earth radii")
    
    m_lo, m_hi = ranges["pl_bmasse"]
    if not (m_lo <= mass <= m_hi):
        errors.append(f"Planet mass must be between {m_lo} and {m_hi} Earth masses")
    
    t_lo, t_hi = ranges["pl_eqt"]
    if not (t_lo <= temp <= t_hi):
        errors.append(f"Equilibrium temperature must be between {t_lo}K and {t_hi}K")
    
    p_lo, p_hi = ranges["pl_orbper"]
    if not (p_lo <= period <= p_hi):
        errors.append(f"Orbital period must be between {p_lo} and {p_hi} days")
    
    return len(errors) == 0, errors


def screen_home() -> None:
    """
    Render the home page with welcome message and start button.
    """
    st.markdown("# Exoplanet Matcher")
    st.markdown("### Discover planets similar to your specifications")
    
    st.markdown("""
    Welcome to the Exoplanet Matcher! This tool helps you find exoplanets that are similar 
    to a planet you describe. Simply enter the characteristics of your planet, and we'll 
    find the most similar exoplanets from our database.
    
    **Features:**
    - Interactive planet visualization
    - Advanced similarity matching
    - Habitability scoring
    - Detailed comparison metrics
    """)
    
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Matching", key="start_button", help="Begin the planet matching process"):
            st.session_state.current_page = 'input'
            st.rerun()

def screen_input() -> None:
    """
    Render the input page with form controls and planet visualization.
    """
    st.markdown("# Enter Planet Parameters")
    st.markdown("Fill in the details of the planet you want to find matches for:")
    
    # Load ranges once
    ranges = load_feature_ranges()
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Planet Characteristics")
        
        st.text_input(
            "Planet Name", key="planet_name", placeholder="e.g., My Test Planet",
            help="A label for your target planet (required)."
        )

        # Planet Radius (R⊕)
        r_lo, r_hi = ranges["pl_rade"]
        st.slider(
            "Planet Radius (Earth radii)",
            min_value=float(r_lo), max_value=float(r_hi),
            value=float(st.session_state.get("radius", clamp(1.0, r_lo, r_hi))),
            step=float(adaptive_step(r_lo, r_hi)),
            key="radius",
            help="How large is the planet compared to Earth? 1.0 = Earth-sized.",
        )

        # Planet Mass (M⊕)
        m_lo, m_hi = ranges["pl_bmasse"]
        st.slider(
            "Planet Mass (Earth masses)",
            min_value=float(m_lo), max_value=float(m_hi),
            value=float(st.session_state.get("mass", clamp(1.0, m_lo, m_hi))),
            step=float(adaptive_step(m_lo, m_hi)),
            key="mass",
            help="How massive is the planet compared to Earth? 1.0 = Earth's mass.",
        )

        # Equilibrium Temp (K)
        t_lo, t_hi = ranges["pl_eqt"]
        st.slider(
            "Equilibrium Temperature (K)",
            min_value=float(t_lo), max_value=float(t_hi),
            value=float(st.session_state.get("temp", clamp(255.0, t_lo, t_hi))),
            step=float(adaptive_step(t_lo, t_hi)),
            key="temp",
            help="Approximate blackbody temperature in Kelvin; higher is hotter (blue→red).",
        )

        # Orbital Period (days)
        p_lo, p_hi = ranges["pl_orbper"]
        st.slider(
            "Orbital Period (days)",
            min_value=float(p_lo), max_value=float(p_hi),
            value=float(st.session_state.get("period", clamp(365.0, p_lo, p_hi))),
            step=float(adaptive_step(p_lo, p_hi)),
            key="period",
            help="Days to orbit its star once; shorter period implies faster orbital speed (longer 'speed lines').",
        )
        
        # Validation and submit button
        is_valid, errors = inputs_valid(
            st.session_state.get("planet_name", ""),
            st.session_state.get("radius", 1.0),
            st.session_state.get("mass", 1.0),
            st.session_state.get("temp", 255.0),
            st.session_state.get("period", 365.0)
        )
        
        if errors:
            for error in errors:
                st.markdown(f'<div class="validation-error">{error}</div>', unsafe_allow_html=True)
        
        # Find matches button
        if st.button("Find Your Match", disabled=not is_valid, help="Search for similar exoplanets"):
            if is_valid:
                try:
                    payload = {
                        "planet_name": st.session_state.get("planet_name", ""),
                        "pl_rade": st.session_state.get("radius", 1.0),
                        "pl_bmasse": st.session_state.get("mass", 1.0),
                        "pl_eqt": st.session_state.get("temp", 255.0),
                        "pl_orbper": st.session_state.get("period", 365.0)
                    }
                    with st.spinner("Searching for similar planets..."):
                        matches = get_matches(payload)
                        st.session_state.matches = matches
                        st.session_state.current_page = 'match'
                        st.rerun()
                except Exception as e:
                    st.error(f"Error finding matches: {str(e)}")
    
    with col2:
        st.markdown("### Planet Visualization")
        st.markdown("*Visual representation of your planet based on the parameters above*")
        
        # Render the planet visualization
        render_planet(
            radius=st.session_state.get("radius", 1.0),
            mass=st.session_state.get("mass", 1.0),
            temp=st.session_state.get("temp", 255.0),
            period=st.session_state.get("period", 365.0),
            ranges=ranges,
        )
        
        # Show current parameters
        st.markdown("**Current Parameters:**")
        st.markdown(f"- **Name:** {st.session_state.get('planet_name', 'Unnamed')}")
        st.markdown(f"- **Radius:** {st.session_state.get('radius', 1.0):.1f} Earth radii")
        st.markdown(f"- **Mass:** {st.session_state.get('mass', 1.0):.1f} Earth masses")
        st.markdown(f"- **Temperature:** {st.session_state.get('temp', 255.0)}K")
        st.markdown(f"- **Orbital Period:** {st.session_state.get('period', 365.0)} days")

def screen_match() -> None:
    """
    Render the match page with top-5 results and planet visualization.
    """
    st.markdown("# Matching Exoplanets")
    
    if st.session_state.matches is None:
        st.warning("No matches found. Please go to the Input page to search for planets.")
        return
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Your Planet")
        
        # Load ranges for planet visualization
        ranges = load_feature_ranges()
        
        # Show planet visualization
        render_planet(
            radius=st.session_state.get("radius", 1.0),
            mass=st.session_state.get("mass", 1.0),
            temp=st.session_state.get("temp", 255.0),
            period=st.session_state.get("period", 365.0),
            ranges=ranges,
        )
        
        # Inputs summary expander
        with st.expander("📋 Inputs Summary", expanded=False):
            st.markdown(f"**Name:** {st.session_state.get('planet_name', 'Unnamed')}")
            st.markdown(f"**Radius:** {st.session_state.get('radius', 1.0):.1f} Earth radii")
            st.markdown(f"**Mass:** {st.session_state.get('mass', 1.0):.1f} Earth masses")
            st.markdown(f"**Temperature:** {st.session_state.get('temp', 255.0)}K")
            st.markdown(f"**Orbital Period:** {st.session_state.get('period', 365.0)} days")
    
    with col2:
        st.markdown("### Top 5 Most Similar Exoplanets")
        
        # Show model metadata
        if 'meta' in st.session_state.matches:
            meta = st.session_state.matches['meta']
            st.caption(f"Model: {meta.get('model_version', 'Unknown')} | Generated: {meta.get('timestamp', 'Unknown')}")
        
        # Display scatterplot if available
        if 'scatterplot' in st.session_state.matches and st.session_state.matches['scatterplot']:
            st.markdown("#### Planet Clusters Visualization")
            st.markdown("*Your planet (red diamond) compared to planet clusters (colored by habitability)*")
            
            import base64
            from io import BytesIO
            
            # Decode and display the scatterplot
            plot_data = base64.b64decode(st.session_state.matches['scatterplot'])
            st.image(plot_data, width='stretch')
        
        # Display each match
        similar_planets = st.session_state.matches.get('similar', [])
        
        if not similar_planets:
            st.warning("No similar planets found in the database.")
            return
        
        for i, planet in enumerate(similar_planets, 1):
            with st.container():
                st.markdown(f'<div class="planet-card">', unsafe_allow_html=True)
                
                # Planet name and basic info
                st.markdown(f"### {i}. {planet['pl_name']}")
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"**Radius:** {planet['pl_rade']:.2f} Earth radii")
                    st.markdown(f"**Mass:** {planet['pl_bmasse']:.2f} Earth masses")
                
                with col_info2:
                    st.markdown(f"**Temperature:** {planet['pl_eqt']:.0f}K")
                    st.markdown(f"**Orbital Period:** {planet['pl_orbper']:.1f} days")
                
                # Scores section
                st.markdown("**Scores:**")
                
                # Similarity score (lower is better) - now using raw KNN distances
                similarity_score = planet['similarity_score']
                # For raw distances, we need to invert the scale properly
                # Assuming typical distance range is 0-5, normalize and invert
                max_distance = 5.0  # Typical max distance for KNN
                normalized_distance = min(similarity_score / max_distance, 1.0)
                similarity_width = (1 - normalized_distance) * 100
                
                col_sim1, col_sim2, col_sim3 = st.columns([2, 3, 1])
                with col_sim1:
                    st.markdown("Similarity Match:")
                with col_sim2:
                    st.markdown(f'<div class="score-bar"><div class="score-bar-fill similarity-fill" style="width: {similarity_width}%"></div></div>', unsafe_allow_html=True)
                with col_sim3:
                    st.markdown(f"{similarity_score:.3f}")
                
                # Habitability score (higher is better)
                habitability_score = planet['habitability_score']
                habitability_width = habitability_score * 100
                
                col_hab1, col_hab2, col_hab3 = st.columns([2, 3, 1])
                with col_hab1:
                    st.markdown("Habitability:")
                with col_hab2:
                    st.markdown(f'<div class="score-bar"><div class="score-bar-fill habitability-fill" style="width: {habitability_width}%"></div></div>', unsafe_allow_html=True)
                with col_hab3:
                    st.markdown(f"{habitability_score:.3f}")
                
                # Tooltip for scores
                st.markdown("**Scores Help:**")
                st.markdown("""
                - **Similarity Match:** Closeness to your inputs on key features (0=identical; lower is better)
                - **Habitability (vs Earth):** Model-estimated Earth-similar habitability potential (0–1; higher is better). Not evidence of life.
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

def main() -> None:
    """
    Main application function that handles page routing.
    """
    # Apply CSS styling first
    st.markdown(render_starfield_css(), unsafe_allow_html=True)
    
    # Render navigation
    page_names = {'home': 'Home', 'input': 'Input', 'match': 'Match'}
    render_navbar(page_names.get(st.session_state.current_page, 'Home'))
    
    # Route to appropriate page
    if st.session_state.current_page == 'home':
        screen_home()
    elif st.session_state.current_page == 'input':
        screen_input()
    elif st.session_state.current_page == 'match':
        screen_match()
    else:
        st.error("Invalid page state. Redirecting to home.")
        st.session_state.current_page = 'home'
        st.rerun()

if __name__ == "__main__":
    main()

