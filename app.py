"""
Sundex Interactive App
Explore winter weather quality across Washington State with adjustable thresholds.
"""

import streamlit as st
import numpy as np
import os
import rasterio
from rasterio.windows import from_bounds
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from io import BytesIO
import base64
import math

# Page config
st.set_page_config(
    page_title="Sundex - Winter Weather Experience",
    page_icon="🌤️",
    layout="wide"
)

# Constants
DATA_DIR = "Data/prism_normals"
WA_BOUNDS = {"west": -124.85, "east": -116.90, "south": 45.50, "north": 49.05}
WINTER_MONTHS = {10: 31, 11: 30, 12: 31, 1: 31, 2: 28, 3: 31, 4: 30}
TOTAL_DAYS = sum(WINTER_MONTHS.values())


def norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


def find_raster_file(pattern_dir):
    if not os.path.isdir(pattern_dir):
        return None
    for f in os.listdir(pattern_dir):
        if f.endswith('.bil') or f.endswith('.tif'):
            return os.path.join(pattern_dir, f)
    return None


@st.cache_data
def get_file_paths():
    """Build dictionary of all available data file paths."""
    paths = {'ppt': {}, 'soltrans': {}, 'tmean': {}, 'tdmean': {}}

    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if not os.path.isdir(item_path):
            continue

        item_lower = item.lower()
        if '800m' in item_lower:
            continue

        for var in paths.keys():
            if var in item_lower:
                month = None
                if '_bil' in item_lower:
                    parts = item.split('_')
                    for i, p in enumerate(parts):
                        if p == 'bil' and i > 0:
                            month_str = parts[i-1]
                            if month_str.isdigit() and len(month_str) == 2:
                                month = int(month_str)
                                break
                elif 'avg_30y' in item_lower:
                    parts = item.split('_')
                    for p in parts:
                        if len(p) == 6 and p.isdigit():
                            month = int(p[4:6])
                            break

                if month:
                    raster_file = find_raster_file(item_path)
                    if raster_file:
                        paths[var][month] = raster_file
                break

    return paths


@st.cache_data
def load_monthly_data():
    """Load all monthly PRISM data for winter months."""
    file_paths = get_file_paths()
    monthly_data = {var: {} for var in file_paths.keys()}
    transform = None
    target_shape = None

    for var, month_files in file_paths.items():
        for month in WINTER_MONTHS.keys():
            if month in month_files:
                with rasterio.open(month_files[month]) as src:
                    window = from_bounds(
                        WA_BOUNDS['west'], WA_BOUNDS['south'],
                        WA_BOUNDS['east'], WA_BOUNDS['north'],
                        src.transform
                    )
                    window = window.round_offsets().round_lengths()
                    data = src.read(1, window=window)

                    if transform is None:
                        transform = src.window_transform(window)
                        target_shape = data.shape

                    nodata = src.nodata if src.nodata else -9999
                    data = np.where(data == nodata, np.nan, data)

                    if data.shape != target_shape:
                        new_data = np.full(target_shape, np.nan)
                        min_rows = min(data.shape[0], target_shape[0])
                        min_cols = min(data.shape[1], target_shape[1])
                        new_data[:min_rows, :min_cols] = data[:min_rows, :min_cols]
                        data = new_data

                    monthly_data[var][month] = data

    return monthly_data, transform


def calculate_good_days(monthly_data, solar_min, precip_max, temp_min):
    """Calculate good weather days with given thresholds."""
    shape = None
    for var_data in monthly_data.values():
        for arr in var_data.values():
            shape = arr.shape
            break
        if shape:
            break

    good_days_combined = np.zeros(shape)
    good_days_solar = np.zeros(shape)
    good_days_dry = np.zeros(shape)

    solar_std = 0.12
    temp_std = 5

    for month, days_in_month in WINTER_MONTHS.items():
        soltrans = monthly_data['soltrans'].get(month)
        ppt = monthly_data['ppt'].get(month)
        tmean = monthly_data['tmean'].get(month)

        if soltrans is None or ppt is None or tmean is None:
            continue

        # Solar
        z_score = (solar_min - soltrans) / solar_std
        frac_good_solar = 1 - norm_cdf(z_score)
        good_days_solar += frac_good_solar * days_in_month

        # Precipitation
        daily_precip_avg = ppt / days_in_month
        # Adjust model based on threshold
        frac_dry = np.clip(0.95 - (daily_precip_avg / (precip_max * 6)), 0.10, 0.95)
        good_days_dry += frac_dry * days_in_month

        # Temperature
        z_temp = (temp_min - tmean) / temp_std
        frac_warm_enough = np.clip(1 - norm_cdf(z_temp), 0.1, 1.0)

        # Combined
        frac_combined = frac_good_solar * frac_dry * frac_warm_enough
        good_days_combined += frac_combined * days_in_month

    return good_days_combined, good_days_solar, good_days_dry


def raster_to_png(data, vmin, vmax, cmap_name='RdYlGn'):
    """Convert numpy array to base64 PNG."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(data))
    rgba[np.isnan(data), 3] = 0
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, 'RGBA')
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_folium_map(good_days, transform, vmin, vmax):
    """Create a Folium map with the good days overlay."""
    rows, cols = good_days.shape
    west, north = transform * (0, 0)
    east, south = transform * (cols, rows)
    bounds = [[south, west], [north, east]]

    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='cartodbpositron'
    )

    img_data = raster_to_png(good_days, vmin, vmax, 'RdYlGn')

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_data}",
        bounds=bounds,
        opacity=0.75,
        name='Good Weather Days'
    ).add_to(m)

    return m


# Main app
st.title("🌤️ Sundex: Winter Weather Experience")
st.markdown("""
How many days during the dark months (Oct-Apr) have weather good enough to support mental health?

Adjust the thresholds below to define what counts as a "good day" for you.
Based on [SAD research](https://pmc.ncbi.nlm.nih.gov/articles/PMC8892387/) showing ≥1hr outdoor daylight reduces depression risk.
""")

# Load data
with st.spinner("Loading PRISM climate data..."):
    monthly_data, transform = load_monthly_data()

# Sidebar with threshold controls
st.sidebar.header("Define a 'Good Day'")
st.sidebar.markdown("Adjust thresholds to see how the map changes.")

solar_min = st.sidebar.slider(
    "☀️ Minimum Solar Transmittance",
    min_value=30,
    max_value=70,
    value=50,
    step=5,
    help="% of sunlight that reaches the surface through clouds. Higher = clearer skies required."
)
solar_min_frac = solar_min / 100

precip_max = st.sidebar.slider(
    "🌧️ Maximum Daily Precipitation",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
    format="%.1f mm",
    help="Max mm of rain/snow for a day to count as 'dry enough'. Lower = stricter."
)

temp_min = st.sidebar.slider(
    "🌡️ Minimum Temperature",
    min_value=-15,
    max_value=10,
    value=0,
    step=1,
    format="%d°C",
    help="Minimum temp to comfortably spend time outside."
)

# Calculate with current thresholds
good_days, good_days_solar, good_days_dry = calculate_good_days(
    monthly_data, solar_min_frac, precip_max, temp_min
)

# Stats
valid = good_days[~np.isnan(good_days)]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Winter Days", TOTAL_DAYS)
col2.metric("Min Good Days", f"{valid.min():.0f}")
col3.metric("Max Good Days", f"{valid.max():.0f}")
col4.metric("WA Average", f"{valid.mean():.0f}")

# Dynamic scale based on data
vmin = max(0, valid.min() - 10)
vmax = min(TOTAL_DAYS, valid.max() + 10)

# Map
st.subheader("Good Weather Days Map")

# Create colorbar legend
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <span style="margin-right: 10px;">{vmin:.0f} days</span>
    <div style="flex-grow: 1; height: 15px; background: linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850); border-radius: 3px;"></div>
    <span style="margin-left: 10px;">{vmax:.0f} days</span>
</div>
""", unsafe_allow_html=True)

m = create_folium_map(good_days, transform, vmin, vmax)
st_folium(m, width=None, height=500, returned_objects=[])

# Threshold explanation
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### Current Definition
A **good day** requires ALL of:
- Solar transmittance ≥ **{solar_min}%**
- Precipitation < **{precip_max} mm/day**
- Temperature ≥ **{temp_min}°C** ({temp_min * 9/5 + 32:.0f}°F)
""")

# Research context
with st.expander("📚 Research Background"):
    st.markdown("""
    ### What the science says about light and mood:

    **Light Therapy Thresholds:**
    - Standard SAD treatment: 10,000 lux for 30 min/day
    - Natural outdoor light: 32,000-100,000 lux (sunny), 10,000-32,000 lux (cloudy)
    - Indoor lighting: only 300-500 lux

    **Outdoor Exposure Benefits:**
    - ≥1 hour of daylight in winter → 28% lower odds of depression ([source](https://www.sciencedirect.com/science/article/pii/S0160412023006864))
    - Each additional hour outdoors → progressively lower depression risk ([UK Biobank study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8892387/))

    **Key Insight:** Even cloudy days provide 20-100x more light than indoors. The question is whether conditions are pleasant enough that you'll actually go outside.
    """)

# Footer
st.markdown("---")
st.markdown("""
*Data: PRISM 30-year climate normals (1991-2020) at 4km resolution*
*Region: Washington State · Period: October through April*
""")
