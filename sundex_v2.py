"""
Sundex v2: Winter Weather Experience Index
Estimates the number of "good weather days" during fall-spring that support mental health.

Based on SAD research:
- ≥1 hour outdoor daylight exposure reduces depression risk by ~28%
- Solar transmittance ≥40% provides therapeutic light levels outdoors
- Dry days encourage outdoor activity
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64
import json
import math


def norm_cdf(x):
    """Standard normal CDF using error function (no scipy needed)."""
    return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))

# Configuration
DATA_DIR = "Data/prism_normals"

# Washington State bounding box
WA_BOUNDS = {
    "west": -124.85,
    "east": -116.90,
    "south": 45.50,
    "north": 49.05
}

# Winter months (Oct-Apr) with days per month
WINTER_MONTHS = {
    10: 31,  # October
    11: 30,  # November
    12: 31,  # December
    1: 31,   # January
    2: 28,   # February (avg)
    3: 31,   # March
    4: 30,   # April
}

# Default thresholds (based on SAD research)
DEFAULT_THRESHOLDS = {
    'solar_min': 0.40,      # Min solar transmittance (40% = enough outdoor light)
    'precip_max': 2.5,      # Max mm/day to count as "dry enough"
    'temp_min': -5,         # Min temp (C) to go outside comfortably
    'temp_max': 20,         # Max temp (C) - not relevant for winter
}


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
        'ppt': {},
        'soltrans': {},
        'tmean': {},
        'tdmean': {}
    }

    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if not os.path.isdir(item_path):
            continue

        item_lower = item.lower()

        # Skip 800m resolution files - only use 4km
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


def load_raster_wa(filepath, bounds=WA_BOUNDS, target_shape=None):
    """Load a raster file and clip to Washington State bounds."""
    with rasterio.open(filepath) as src:
        window = from_bounds(
            bounds['west'], bounds['south'],
            bounds['east'], bounds['north'],
            src.transform
        )
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window)
        transform = src.window_transform(window)

        nodata = src.nodata if src.nodata else -9999
        data = np.where(data == nodata, np.nan, data)

        if target_shape is not None and data.shape != target_shape:
            new_data = np.full(target_shape, np.nan)
            min_rows = min(data.shape[0], target_shape[0])
            min_cols = min(data.shape[1], target_shape[1])
            new_data[:min_rows, :min_cols] = data[:min_rows, :min_cols]
            data = new_data

        return data, transform, src.crs


def load_monthly_data(file_paths):
    """Load all monthly data for winter months."""
    monthly_data = {var: {} for var in file_paths.keys()}
    transform = None
    crs = None
    target_shape = None

    for var, month_files in file_paths.items():
        for month in WINTER_MONTHS.keys():
            if month in month_files:
                data, t, c = load_raster_wa(month_files[month], target_shape=target_shape)
                monthly_data[var][month] = data
                if transform is None:
                    transform = t
                    crs = c
                    target_shape = data.shape

    return monthly_data, transform, crs


def estimate_good_days_from_monthly(monthly_data, thresholds=DEFAULT_THRESHOLDS):
    """
    Estimate number of "good weather days" from monthly normals.

    Approach:
    - Solar: Assume daily transmittance varies around monthly mean with ~15% std dev
      Estimate fraction of days above threshold using normal distribution
    - Precip: Convert monthly total to avg daily, estimate dry days assuming
      precipitation follows a gamma distribution (many dry days, few wet days)
    - Temp: Use monthly mean to determine if month is "too cold"

    A "good day" requires:
    - Sufficient solar (above threshold)
    - Not too rainy (below threshold)
    - Not too cold (above threshold)
    """

    # Get shape from first available data
    shape = None
    for var_data in monthly_data.values():
        for arr in var_data.values():
            shape = arr.shape
            break
        if shape:
            break

    # Initialize results
    good_days_solar = np.zeros(shape)
    good_days_dry = np.zeros(shape)
    good_days_temp = np.zeros(shape)
    good_days_combined = np.zeros(shape)
    total_days = sum(WINTER_MONTHS.values())

    # Store monthly breakdowns for the UI
    monthly_good_days = {}

    for month, days_in_month in WINTER_MONTHS.items():
        # Get data for this month
        soltrans = monthly_data['soltrans'].get(month)
        ppt = monthly_data['ppt'].get(month)
        tmean = monthly_data['tmean'].get(month)

        if soltrans is None or ppt is None or tmean is None:
            continue

        # === SOLAR: Estimate days above threshold ===
        # Assume daily solar transmittance ~ Normal(monthly_mean, 0.12)
        # This captures day-to-day variability in cloud cover
        solar_std = 0.12  # ~12% standard deviation in daily cloud cover
        solar_threshold = thresholds['solar_min']

        # Fraction of days above threshold (using normal CDF)
        z_score = (solar_threshold - soltrans) / solar_std
        frac_good_solar = 1 - norm_cdf(z_score)
        days_good_solar = frac_good_solar * days_in_month
        good_days_solar += days_good_solar

        # === PRECIPITATION: Estimate dry days ===
        # Monthly precip (mm) / days = avg daily precip
        # Assume ~60% of days have no/minimal precip in winter
        # Scale based on total monthly precip
        daily_precip_avg = ppt / days_in_month
        precip_threshold = thresholds['precip_max']

        # Rough model: fraction of dry days decreases as avg daily precip increases
        # At 0 mm/day avg -> ~95% dry days
        # At 5 mm/day avg (150mm/mo) -> ~40% dry days
        # At 10 mm/day avg (300mm/mo) -> ~20% dry days
        frac_dry = np.clip(0.95 - (daily_precip_avg / 15), 0.15, 0.95)
        days_dry = frac_dry * days_in_month
        good_days_dry += days_dry

        # === TEMPERATURE: Is it too cold to go outside? ===
        # If monthly mean is above threshold, assume most days are OK
        # If below, scale down
        temp_threshold = thresholds['temp_min']
        temp_std = 5  # Daily temp variation around monthly mean

        z_temp = (temp_threshold - tmean) / temp_std
        frac_warm_enough = 1 - norm_cdf(z_temp)
        frac_warm_enough = np.clip(frac_warm_enough, 0.1, 1.0)
        days_warm = frac_warm_enough * days_in_month
        good_days_temp += days_warm

        # === COMBINED: Days that meet ALL criteria ===
        # Probability of all three = product of probabilities (assuming independence)
        # This is approximate but reasonable for monthly data
        frac_combined = frac_good_solar * frac_dry * frac_warm_enough
        days_combined = frac_combined * days_in_month
        good_days_combined += days_combined

        monthly_good_days[month] = {
            'solar': days_good_solar,
            'dry': days_dry,
            'temp': days_warm,
            'combined': days_combined
        }

    return {
        'good_days': good_days_combined,
        'good_days_solar': good_days_solar,
        'good_days_dry': good_days_dry,
        'good_days_temp': good_days_temp,
        'total_days': total_days,
        'monthly': monthly_good_days,
        'thresholds': thresholds
    }


def raster_to_png(data, vmin, vmax, cmap_name='RdYlGn'):
    """Convert numpy array to PNG with transparency for NaN."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(data))
    rgba[np.isnan(data), 3] = 0

    # Use PIL-style approach for cleaner output
    from PIL import Image
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, 'RGBA')

    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def create_map(results, monthly_data, transform, thresholds, output_path="sundex_map.html"):
    """Create interactive map with dynamic legend and layer switching."""

    good_days = results['good_days']
    rows, cols = good_days.shape

    # Calculate bounds
    west, north = transform * (0, 0)
    east, south = transform * (cols, rows)
    bounds = [[south, west], [north, east]]

    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    # Calculate winter averages for raw data display
    winter_avg = {}
    for var in ['soltrans', 'ppt', 'tmean', 'tdmean']:
        arrays = [monthly_data[var][m] for m in WINTER_MONTHS.keys() if m in monthly_data[var]]
        if arrays:
            winter_avg[var] = np.nanmean(np.stack(arrays), axis=0)

    # Define layers
    layers = [
        {
            'id': 'good_days',
            'name': 'Good Weather Days',
            'data': results['good_days'],
            'vmin': 30, 'vmax': 120,
            'cmap': 'RdYlGn',
            'unit': 'days',
            'description': f'Days with sun ≥{int(thresholds["solar_min"]*100)}%, precip <{thresholds["precip_max"]}mm, temp ≥{thresholds["temp_min"]}°C'
        },
        {
            'id': 'good_days_solar',
            'name': 'Sunny Days',
            'data': results['good_days_solar'],
            'vmin': 40, 'vmax': 160,
            'cmap': 'YlOrRd',
            'unit': 'days',
            'description': f'Days with solar transmittance ≥{int(thresholds["solar_min"]*100)}%'
        },
        {
            'id': 'good_days_dry',
            'name': 'Dry Days',
            'data': results['good_days_dry'],
            'vmin': 60, 'vmax': 180,
            'cmap': 'BrBG',
            'unit': 'days',
            'description': f'Days with precipitation <{thresholds["precip_max"]}mm'
        },
        {
            'id': 'soltrans',
            'name': 'Solar Transmittance',
            'data': winter_avg.get('soltrans', np.zeros_like(good_days)) * 100,
            'vmin': 30, 'vmax': 65,
            'cmap': 'YlOrRd',
            'unit': '%',
            'description': 'Average % of sunlight reaching surface (Oct-Apr)'
        },
        {
            'id': 'ppt',
            'name': 'Precipitation',
            'data': winter_avg.get('ppt', np.zeros_like(good_days)),
            'vmin': 0, 'vmax': 400,
            'cmap': 'Blues',
            'unit': 'mm/mo',
            'description': 'Average monthly precipitation (Oct-Apr)'
        },
        {
            'id': 'tmean',
            'name': 'Temperature',
            'data': winter_avg.get('tmean', np.zeros_like(good_days)),
            'vmin': -5, 'vmax': 10,
            'cmap': 'coolwarm',
            'unit': '°C',
            'description': 'Average temperature (Oct-Apr)'
        },
    ]

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='cartodbpositron'
    )

    # Generate layer data for JavaScript
    layer_info = {}

    for layer in layers:
        print(f"  Rendering {layer['name']}...")

        img_data = raster_to_png(
            layer['data'],
            layer['vmin'],
            layer['vmax'],
            layer['cmap']
        )

        layer_info[layer['id']] = {
            'name': layer['name'],
            'vmin': layer['vmin'],
            'vmax': layer['vmax'],
            'unit': layer['unit'],
            'description': layer['description'],
            'cmap': layer['cmap']
        }

        # Add as image overlay with unique name
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_data}",
            bounds=bounds,
            opacity=0.75,
            name=layer['name'],
            show=(layer['id'] == 'good_days')  # Only show first layer initially
        ).add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Create the info panel and dynamic legend with JavaScript
    total_days = results['total_days']

    custom_html = f'''
    <style>
        #sundex-panel {{
            position: fixed;
            top: 10px;
            left: 50px;
            z-index: 9999;
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 340px;
            font-size: 13px;
        }}
        #sundex-panel h2 {{
            margin: 0 0 5px 0;
            font-size: 20px;
            color: #333;
        }}
        #sundex-panel .subtitle {{
            color: #666;
            margin-bottom: 12px;
            font-size: 12px;
        }}
        #sundex-panel .description {{
            color: #444;
            line-height: 1.5;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #eee;
        }}
        #sundex-panel .layer-info {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        #sundex-panel .layer-name {{
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }}
        #sundex-panel .layer-desc {{
            font-size: 11px;
            color: #666;
        }}
        #dynamic-legend {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        #dynamic-legend .legend-bar {{
            height: 12px;
            border-radius: 3px;
            margin: 5px 0;
        }}
        #dynamic-legend .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
        }}
        .thresholds {{
            font-size: 11px;
            color: #888;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        .thresholds b {{
            color: #666;
        }}
    </style>

    <div id="sundex-panel">
        <h2>Sundex</h2>
        <div class="subtitle">Winter Weather Experience · Oct–Apr · {total_days} days</div>

        <div class="description">
            How many days during the dark months have weather good enough to
            support mental health? Based on SAD research showing outdoor daylight
            exposure ≥1hr reduces depression risk.
        </div>

        <div class="layer-info">
            <div class="layer-name" id="current-layer-name">Good Weather Days</div>
            <div class="layer-desc" id="current-layer-desc">
                Days with sun ≥{int(thresholds["solar_min"]*100)}%, precip &lt;{thresholds["precip_max"]}mm, temp ≥{thresholds["temp_min"]}°C
            </div>
        </div>

        <div id="dynamic-legend">
            <div class="legend-bar" id="legend-bar" style="background: linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850);"></div>
            <div class="legend-labels">
                <span id="legend-min">30</span>
                <span id="legend-unit">days</span>
                <span id="legend-max">120</span>
            </div>
        </div>

        <div class="thresholds">
            <b>Current thresholds:</b><br>
            ☀️ Solar ≥ {int(thresholds["solar_min"]*100)}% ·
            🌧️ Precip &lt; {thresholds["precip_max"]}mm/day ·
            🌡️ Temp ≥ {thresholds["temp_min"]}°C
        </div>
    </div>

    <script>
        var layerInfo = {json.dumps(layer_info)};

        // Color gradients for each colormap
        var cmapGradients = {{
            'RdYlGn': 'linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850)',
            'YlOrRd': 'linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #f03b20, #bd0026)',
            'BrBG': 'linear-gradient(to right, #8c510a, #d8b365, #f6e8c3, #c7eae5, #5ab4ac, #01665e)',
            'Blues': 'linear-gradient(to right, #f7fbff, #c6dbef, #6baed6, #2171b5, #084594)',
            'coolwarm': 'linear-gradient(to right, #3b4cc0, #7092c0, #aac7fd, #dddddd, #f7a789, #c24b40, #b40426)'
        }};

        // Listen for layer changes
        document.addEventListener('DOMContentLoaded', function() {{
            // Find the layer control and add listeners
            var checkInterval = setInterval(function() {{
                var inputs = document.querySelectorAll('.leaflet-control-layers-overlays input');
                if (inputs.length > 0) {{
                    clearInterval(checkInterval);
                    inputs.forEach(function(input) {{
                        input.addEventListener('change', function() {{
                            if (this.checked) {{
                                updateLegend(this.nextSibling.textContent.trim());
                            }}
                        }});
                    }});
                }}
            }}, 100);
        }});

        function updateLegend(layerName) {{
            // Find layer info by name
            var info = null;
            for (var id in layerInfo) {{
                if (layerInfo[id].name === layerName) {{
                    info = layerInfo[id];
                    break;
                }}
            }}
            if (!info) return;

            document.getElementById('current-layer-name').textContent = info.name;
            document.getElementById('current-layer-desc').textContent = info.description;
            document.getElementById('legend-min').textContent = info.vmin;
            document.getElementById('legend-max').textContent = info.vmax;
            document.getElementById('legend-unit').textContent = info.unit;
            document.getElementById('legend-bar').style.background = cmapGradients[info.cmap] || cmapGradients['RdYlGn'];
        }}
    </script>
    '''

    m.get_root().html.add_child(folium.Element(custom_html))

    # Remove the default colormap legend (we're using our custom one)

    m.save(output_path)
    print(f"  Map saved to {output_path}")

    # Print summary stats
    valid = good_days[~np.isnan(good_days)]
    print(f"\n  Summary Statistics:")
    print(f"  Total winter days: {total_days}")
    print(f"  Good days range: {valid.min():.0f} - {valid.max():.0f}")
    print(f"  Good days mean: {valid.mean():.0f}")

    return m


def main():
    print("=" * 60)
    print("SUNDEX v2: Good Weather Days Calculator")
    print("=" * 60)

    # Load data
    print("\n[1/4] Scanning for PRISM data files...")
    file_paths = get_file_paths()
    for var, months in file_paths.items():
        print(f"  {var}: {len(months)} months")

    print("\n[2/4] Loading monthly data for Oct-Apr...")
    monthly_data, transform, crs = load_monthly_data(file_paths)

    # Calculate good days
    print("\n[3/4] Estimating good weather days...")
    thresholds = DEFAULT_THRESHOLDS.copy()
    results = estimate_good_days_from_monthly(monthly_data, thresholds)

    # Create map
    print("\n[4/4] Creating interactive map...")
    create_map(results, monthly_data, transform, thresholds, "sundex_map.html")

    print("\n" + "=" * 60)
    print("Done! Open sundex_map.html in your browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
