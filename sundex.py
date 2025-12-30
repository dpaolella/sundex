"""
Sundex: Winter Weather Experience Index
An interactive map showing how winter weather *feels* across Washington State
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import folium
from folium import plugins
import branca.colormap as cm
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64

# Configuration
DATA_DIR = "Data/prism_normals"

# Washington State bounding box (with small buffer)
WA_BOUNDS = {
    "west": -124.85,
    "east": -116.90,
    "south": 45.50,
    "north": 49.05
}

# Winter months for Sundex calculation (Oct-Apr)
WINTER_MONTHS = [10, 11, 12, 1, 2, 3, 4]


def find_raster_file(pattern_dir):
    """Find the main raster file (.bil or .tif) in a directory."""
    if not os.path.isdir(pattern_dir):
        return None
    for f in os.listdir(pattern_dir):
        if f.endswith('.bil') or f.endswith('.tif'):
            return os.path.join(pattern_dir, f)
    return None


def get_file_paths():
    """Build dictionary of all available data file paths."""
    paths = {
        'ppt': {},      # precipitation (mm)
        'soltrans': {}, # solar transmittance (%)
        'tmean': {},    # mean temp (C)
        'tdmean': {}    # mean dew point (C)
    }

    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if not os.path.isdir(item_path):
            continue

        # Parse the directory name to extract variable and month
        item_lower = item.lower()

        for var in paths.keys():
            if var in item_lower:
                # Extract month - look for 2-digit month pattern
                # Old format: PRISM_ppt_30yr_normal_4kmM4_01_bil
                # New format: prism_tmean_us_25m_202001_avg_30y
                month = None

                if '_bil' in item_lower:
                    # Old format: month is before _bil
                    parts = item.split('_')
                    for i, p in enumerate(parts):
                        if p == 'bil' and i > 0:
                            month_str = parts[i-1]
                            if month_str.isdigit() and len(month_str) == 2:
                                month = int(month_str)
                                break
                elif 'avg_30y' in item_lower:
                    # New format: YYYYMM pattern
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


def load_raster_wa(filepath, bounds=WA_BOUNDS, target_shape=None):
    """Load a raster file and clip to Washington State bounds."""
    with rasterio.open(filepath) as src:
        # Calculate the window for WA bounds
        window = from_bounds(
            bounds['west'], bounds['south'],
            bounds['east'], bounds['north'],
            src.transform
        )

        # Round window to integer pixels for consistency
        window = window.round_offsets().round_lengths()

        # Read the windowed data
        data = src.read(1, window=window)

        # Get the transform for the windowed data
        transform = src.window_transform(window)

        # Handle nodata
        nodata = src.nodata if src.nodata else -9999
        data = np.where(data == nodata, np.nan, data)

        # Ensure consistent shape if target provided
        if target_shape is not None and data.shape != target_shape:
            # Trim or pad to match target shape
            new_data = np.full(target_shape, np.nan)
            min_rows = min(data.shape[0], target_shape[0])
            min_cols = min(data.shape[1], target_shape[1])
            new_data[:min_rows, :min_cols] = data[:min_rows, :min_cols]
            data = new_data

        return data, transform, src.crs


def compute_winter_averages(file_paths, months=WINTER_MONTHS):
    """Compute winter-period averages for each variable."""
    averages = {}
    transform = None
    crs = None
    target_shape = None

    for var, month_files in file_paths.items():
        monthly_data = []

        for month in months:
            if month in month_files:
                data, t, c = load_raster_wa(month_files[month], target_shape=target_shape)
                monthly_data.append(data)
                if transform is None:
                    transform = t
                    crs = c
                    target_shape = data.shape  # Use first loaded shape as reference

        if monthly_data:
            # Stack and compute mean, ignoring NaN
            stacked = np.stack(monthly_data, axis=0)
            averages[var] = np.nanmean(stacked, axis=0)
            print(f"  {var}: {len(monthly_data)} months, shape {averages[var].shape}")

    return averages, transform, crs


def compute_sundex(averages):
    """
    Compute the Sundex composite metric.

    Components (all normalized 0-100, higher = better):
    1. Solar Index: Based on solar transmittance (more sun = better)
    2. Precipitation Index: Less precip = better, but also consider dampness
    3. Comfort Index: Based on temp and humidity (temp-dewpoint spread)

    The Sundex formula weights these factors based on their psychological impact.
    """

    # Get the data arrays
    soltrans = averages.get('soltrans')  # % (0-100 typically)
    ppt = averages.get('ppt')             # mm/month
    tmean = averages.get('tmean')         # degrees C
    tdmean = averages.get('tdmean')       # degrees C

    # === SOLAR INDEX (0-100) ===
    # Solar transmittance is 0-1 ratio, convert to 0-100 scale
    # Higher = more sun reaching surface = better
    if soltrans is not None:
        solar_idx = np.clip(soltrans * 100, 0, 100)
    else:
        solar_idx = np.full_like(ppt, 50)

    # === PRECIPITATION INDEX (0-100) ===
    # Lower precip = higher score
    # Seattle winter avg ~150mm/month, Eastern WA ~25mm/month
    if ppt is not None:
        # Normalize: 0mm -> 100, 200mm -> 0
        precip_idx = 100 - np.clip(ppt / 2, 0, 100)
    else:
        precip_idx = np.full_like(solar_idx, 50)

    # === COMFORT INDEX (0-100) ===
    # Based on temperature and humidity
    # Damp cold (small temp-dewpoint spread) feels worse
    if tmean is not None and tdmean is not None:
        # Temperature-dewpoint spread (larger = drier, feels better)
        dewpoint_spread = tmean - tdmean

        # Normalize spread: 0C spread (saturated) -> 0, 15C spread (dry) -> 50
        humidity_factor = np.clip(dewpoint_spread / 15 * 50, 0, 50)

        # Temperature factor: freezing rain zone (0-5C) is worst
        # Below freezing with sun can feel okay, mild temps feel okay
        # Penalty for the "damp chill" zone
        temp_factor = np.where(
            (tmean > 0) & (tmean < 7),
            np.clip(30 - (7 - np.abs(tmean - 3.5)) * 5, 0, 50),  # Penalty for 0-7C
            np.clip(50 - np.abs(tmean - 10) * 2, 0, 50)  # Optimal around 10C
        )

        comfort_idx = humidity_factor + temp_factor
    else:
        comfort_idx = np.full_like(solar_idx, 50)

    # === SUNDEX COMPOSITE ===
    # Weighted combination emphasizing solar (gloom is the main issue)
    sundex = (
        0.45 * solar_idx +      # Solar is most important
        0.30 * precip_idx +     # Precipitation frequency matters
        0.25 * comfort_idx      # Comfort from temp/humidity
    )

    return {
        'sundex': sundex,
        'solar_idx': solar_idx,
        'precip_idx': precip_idx,
        'comfort_idx': comfort_idx,
        'soltrans': soltrans,
        'ppt': ppt,
        'tmean': tmean,
        'tdmean': tdmean
    }


def raster_to_image(data, vmin, vmax, cmap_name='RdYlGn'):
    """Convert a numpy array to a PNG image with transparency for NaN values."""
    # Normalize data to 0-1 range
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    # Apply colormap (returns RGBA)
    rgba = cmap(norm(data))

    # Set alpha to 0 where data is NaN
    rgba[np.isnan(data), 3] = 0

    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Save to PNG in memory
    fig, ax = plt.subplots(figsize=(data.shape[1]/10, data.shape[0]/10), dpi=100)
    ax.imshow(rgba_uint8, interpolation='bilinear')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def create_map(metrics, transform, output_path="sundex_map.html"):
    """Create an interactive Folium map with raster overlays."""

    sundex = metrics['sundex']
    rows, cols = sundex.shape

    # Calculate bounds from transform
    # Top-left corner
    west, north = transform * (0, 0)
    # Bottom-right corner
    east, south = transform * (cols, rows)

    bounds = [[south, west], [north, east]]

    print(f"  Raster bounds: S={south:.2f}, N={north:.2f}, W={west:.2f}, E={east:.2f}")

    # Create the map centered on Washington
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='cartodbpositron'
    )

    # Define metrics to display with their settings
    layer_configs = [
        {
            'name': 'Sundex Score',
            'data': metrics['sundex'],
            'vmin': 25, 'vmax': 65,
            'cmap': 'RdYlGn',
            'description': 'Composite winter weather experience (higher = better)',
            'show': True
        },
        {
            'name': 'Solar Index',
            'data': metrics['solar_idx'],
            'vmin': 30, 'vmax': 65,
            'cmap': 'YlOrRd',
            'description': 'Sunlight reaching surface through clouds',
            'show': False
        },
        {
            'name': 'Precipitation Index',
            'data': metrics['precip_idx'],
            'vmin': 0, 'vmax': 100,
            'cmap': 'BrBG',
            'description': 'Inverse of monthly precipitation (higher = drier)',
            'show': False
        },
        {
            'name': 'Comfort Index',
            'data': metrics['comfort_idx'],
            'vmin': 20, 'vmax': 80,
            'cmap': 'coolwarm',
            'description': 'Temperature and humidity comfort',
            'show': False
        },
    ]

    # Create image overlay for each metric
    for config in layer_configs:
        print(f"  Rendering {config['name']} layer...")

        img_data = raster_to_image(
            config['data'],
            config['vmin'],
            config['vmax'],
            config['cmap']
        )

        img_overlay = folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_data}",
            bounds=bounds,
            opacity=0.7,
            name=config['name'],
            show=config['show']
        )
        img_overlay.add_to(m)

    # Add colorbar legend for Sundex
    colormap = cm.LinearColormap(
        colors=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'],
        vmin=25, vmax=65,
        caption='Sundex Score (higher = better winter experience)'
    )
    colormap.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add title and description panel
    info_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 15px 20px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial, sans-serif;
                max-width: 320px;">
        <h2 style="margin: 0 0 10px 0; color: #333;">Sundex</h2>
        <p style="margin: 0 0 10px 0; font-size: 13px; color: #333; font-weight: 500;">
            Winter Weather Experience Index
        </p>
        <p style="margin: 0 0 10px 0; font-size: 12px; color: #666; line-height: 1.4;">
            How does winter <em>feel</em> in different places? This map captures the
            experienced character of winter—not just temperature or rain days, but
            the combination of gloom, dampness, and chill that affects wellbeing.
        </p>
        <div style="font-size: 11px; color: #888; border-top: 1px solid #eee; padding-top: 8px; margin-top: 8px;">
            <b>Data:</b> PRISM 30-year normals (1991-2020)<br>
            <b>Period:</b> October – April average<br>
            <b>Resolution:</b> ~4km grid<br>
            <b>Region:</b> Washington State
        </div>
        <div style="font-size: 11px; color: #888; border-top: 1px solid #eee; padding-top: 8px; margin-top: 8px;">
            <b>Components:</b><br>
            • Solar (45%): Cloud transmittance<br>
            • Precip (30%): Monthly rainfall<br>
            • Comfort (25%): Temp + humidity
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))

    # Save the map
    m.save(output_path)
    print(f"  Map saved to {output_path}")

    return m


def main():
    print("=" * 60)
    print("SUNDEX: Winter Weather Experience Index")
    print("=" * 60)

    # Step 1: Find all data files
    print("\n[1/4] Scanning for PRISM data files...")
    file_paths = get_file_paths()

    for var, months in file_paths.items():
        print(f"  {var}: {len(months)} months found")

    # Step 2: Load and average winter months
    print("\n[2/4] Computing winter averages (Oct-Apr)...")
    averages, transform, crs = compute_winter_averages(file_paths)

    # Step 3: Compute Sundex
    print("\n[3/4] Computing Sundex composite metric...")
    metrics = compute_sundex(averages)

    # Print some stats
    sundex = metrics['sundex']
    valid = sundex[~np.isnan(sundex)]
    print(f"  Sundex range: {valid.min():.1f} - {valid.max():.1f}")
    print(f"  Sundex mean: {valid.mean():.1f}")

    # Step 4: Create the map
    print("\n[4/4] Creating interactive map...")
    create_map(metrics, transform, "sundex_map.html")

    print("\n" + "=" * 60)
    print("Done! Open sundex_map.html in your browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
