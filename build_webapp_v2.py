"""
Build Sundex v2 - Enhanced standalone web application.
Features:
- Layer toggles (Good Days, Solar, Precip, Temp)
- Month selector with presets
- Click/hover for details
- City markers
- URL sharing
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import json

# Constants
DATA_DIR = "Data/prism_normals"
WA_BOUNDS = {"west": -124.85, "east": -116.90, "south": 45.50, "north": 49.05}

# All months with days
ALL_MONTHS = {
    1: {"name": "Jan", "days": 31},
    2: {"name": "Feb", "days": 28},
    3: {"name": "Mar", "days": 31},
    4: {"name": "Apr", "days": 30},
    5: {"name": "May", "days": 31},
    6: {"name": "Jun", "days": 30},
    7: {"name": "Jul", "days": 31},
    8: {"name": "Aug", "days": 31},
    9: {"name": "Sep", "days": 30},
    10: {"name": "Oct", "days": 31},
    11: {"name": "Nov", "days": 30},
    12: {"name": "Dec", "days": 31},
}

# Cities to mark
CITIES = [
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    {"name": "Spokane", "lat": 47.6588, "lon": -117.4260},
    {"name": "Bellingham", "lat": 48.7519, "lon": -122.4787},
    {"name": "Wenatchee", "lat": 47.4235, "lon": -120.3103},
    {"name": "Yakima", "lat": 46.6021, "lon": -120.5059},
    {"name": "Olympia", "lat": 47.0379, "lon": -122.9007},
    {"name": "Tri-Cities", "lat": 46.2856, "lon": -119.2845},
    {"name": "Vancouver", "lat": 45.6387, "lon": -122.6615},
    {"name": "Ellensburg", "lat": 46.9965, "lon": -120.5478},
    {"name": "Port Angeles", "lat": 48.1181, "lon": -123.4307},
]


def find_raster_file(pattern_dir):
    if not os.path.isdir(pattern_dir):
        return None
    for f in os.listdir(pattern_dir):
        if f.endswith('.bil') or f.endswith('.tif'):
            return os.path.join(pattern_dir, f)
    return None


def get_file_paths():
    paths = {'ppt': {}, 'soltrans': {}, 'tmean': {}}
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


def load_monthly_data():
    """Load all monthly data (not averaged)."""
    file_paths = get_file_paths()
    monthly_data = {var: {} for var in file_paths.keys()}
    transform = None
    target_shape = None

    for var, month_files in file_paths.items():
        for month in range(1, 13):
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
                        min_r = min(data.shape[0], target_shape[0])
                        min_c = min(data.shape[1], target_shape[1])
                        new_data[:min_r, :min_c] = data[:min_r, :min_c]
                        data = new_data

                    monthly_data[var][month] = data

    return monthly_data, transform, target_shape


def downsample(arr, factor=2):
    """Downsample array by averaging blocks."""
    h, w = arr.shape
    new_h, new_w = h // factor, w // factor
    result = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            block = arr[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            valid = block[~np.isnan(block)]
            if len(valid) > 0:
                result[i, j] = np.mean(valid)
            else:
                result[i, j] = np.nan

    return result


def to_json_list(arr):
    """Convert numpy array to JSON-serializable list."""
    result = []
    for row in arr:
        json_row = []
        for val in row:
            if np.isnan(val):
                json_row.append(None)
            else:
                json_row.append(round(float(val), 3))
        result.append(json_row)
    return result


def main():
    print("Loading PRISM data for all months...")
    monthly_data, transform, shape = load_monthly_data()

    # Downsample for web
    factor = 2
    downsampled = {
        var: {month: downsample(arr, factor) for month, arr in month_data.items()}
        for var, month_data in monthly_data.items()
    }

    # Get new shape
    sample_arr = list(list(downsampled.values())[0].values())[0]
    rows, cols = sample_arr.shape

    # Adjust transform
    new_transform = rasterio.Affine(
        transform.a * factor,
        transform.b,
        transform.c,
        transform.d,
        transform.e * factor,
        transform.f
    )

    west, north = new_transform * (0, 0)
    east, south = new_transform * (cols, rows)

    print(f"Data shape: {rows}x{cols}")
    print(f"Bounds: N={north:.2f}, S={south:.2f}, W={west:.2f}, E={east:.2f}")

    # Build data export
    data_export = {
        'bounds': {'north': north, 'south': south, 'west': west, 'east': east},
        'shape': {'rows': rows, 'cols': cols},
        'cellSize': abs(new_transform.a),
        'months': {str(m): info for m, info in ALL_MONTHS.items()},
        'cities': CITIES,
        'data': {}
    }

    # Export monthly data for each variable
    for var in ['soltrans', 'ppt', 'tmean']:
        data_export['data'][var] = {}
        for month in range(1, 13):
            if month in downsampled[var]:
                data_export['data'][var][str(month)] = to_json_list(downsampled[var][month])
                arr = downsampled[var][month]
                valid = arr[~np.isnan(arr)]
                print(f"  {var} month {month}: {valid.min():.2f} - {valid.max():.2f}")

    json_str = json.dumps(data_export)
    print(f"\nJSON data size: {len(json_str) / 1024:.0f} KB")

    # Generate HTML
    html_content = generate_html(json_str)

    with open('index.html', 'w') as f:
        f.write(html_content)

    print("Created index.html")


def generate_html(data_json):
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sundex - Winter Weather Experience Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
        }

        .container { display: flex; height: 100vh; }

        .sidebar {
            width: 400px;
            background: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            overflow-y: auto;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }

        .sidebar-header {
            padding: 20px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .sidebar-header h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
        .sidebar-header p { font-size: 13px; opacity: 0.9; line-height: 1.4; }

        .stats-bar {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }

        .stat {
            flex: 1;
            padding: 14px 8px;
            text-align: center;
            border-right: 1px solid #e9ecef;
        }
        .stat:last-child { border-right: none; }
        .stat-value { font-size: 22px; font-weight: 700; color: #667eea; }
        .stat-label { font-size: 10px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

        .section {
            padding: 16px 20px;
            border-bottom: 1px solid #e9ecef;
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        /* Layer Toggle */
        .layer-toggles { display: flex; gap: 6px; flex-wrap: wrap; }
        .layer-btn {
            padding: 8px 14px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            background: white;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .layer-btn:hover { border-color: #667eea; }
        .layer-btn.active {
            background: #667eea;
            border-color: #667eea;
            color: white;
        }

        /* Month Selector */
        .month-presets { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
        .preset-btn {
            padding: 6px 12px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background: #f8f9fa;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .preset-btn:hover { background: #e9ecef; }
        .preset-btn.active { background: #667eea; color: white; border-color: #667eea; }

        .month-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; }
        .month-btn {
            padding: 8px 4px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background: white;
            font-size: 11px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .month-btn:hover { border-color: #667eea; }
        .month-btn.active { background: #667eea; color: white; border-color: #667eea; }
        .month-btn.disabled { opacity: 0.4; cursor: not-allowed; }

        /* Sliders */
        .control-group { margin-bottom: 16px; }
        .control-group:last-child { margin-bottom: 0; }
        .control-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .control-label span { font-size: 13px; color: #495057; }
        .control-value {
            font-size: 12px;
            font-weight: 600;
            color: #667eea;
            background: #f0f1ff;
            padding: 3px 8px;
            border-radius: 4px;
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #e9ecef;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.4);
        }

        .control-hint { font-size: 11px; color: #868e96; margin-top: 4px; line-height: 1.3; }

        /* Legend */
        .legend-bar {
            height: 10px;
            border-radius: 5px;
            margin: 8px 0 4px 0;
        }
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #6c757d;
        }

        /* Location Details */
        .location-details {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            display: none;
        }
        .location-details.visible { display: block; }
        .location-name { font-weight: 600; font-size: 14px; margin-bottom: 8px; }
        .location-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .location-stat { font-size: 12px; }
        .location-stat-label { color: #6c757d; }
        .location-stat-value { font-weight: 600; color: #333; }

        /* Comparison */
        .compare-container { display: flex; gap: 12px; }
        .compare-location {
            flex: 1;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            position: relative;
        }
        .compare-location.loc-a { border-left: 3px solid #667eea; }
        .compare-location.loc-b { border-left: 3px solid #e91e63; }
        .compare-location .location-name { font-size: 13px; }
        .compare-clear {
            position: absolute;
            top: 8px;
            right: 8px;
            background: none;
            border: none;
            color: #adb5bd;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        }
        .compare-clear:hover { color: #666; }
        .compare-diff {
            margin-top: 12px;
            padding: 10px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        .compare-diff-title { font-size: 11px; color: #6c757d; margin-bottom: 6px; }
        .compare-diff-value {
            font-size: 18px;
            font-weight: 700;
        }
        .compare-diff-value.positive { color: #28a745; }
        .compare-diff-value.negative { color: #dc3545; }
        .compare-instructions {
            font-size: 12px;
            color: #868e96;
            text-align: center;
            padding: 20px;
        }
        .map-marker {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 3px solid;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        .map-marker.marker-a { border-color: #667eea; color: #667eea; }
        .map-marker.marker-b { border-color: #e91e63; color: #e91e63; }

        /* Share Button */
        .share-btn {
            width: 100%;
            padding: 10px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .share-btn:hover { background: #e9ecef; }
        .share-btn.copied { background: #d4edda; border-color: #c3e6cb; }

        /* Footer */
        .sidebar-footer {
            margin-top: auto;
            padding: 16px 20px;
            background: #f8f9fa;
            font-size: 11px;
            color: #6c757d;
            line-height: 1.5;
        }
        .sidebar-footer a { color: #667eea; text-decoration: none; }
        .sidebar-footer a:hover { text-decoration: underline; }

        #map { flex: 1; height: 100vh; }
        .leaflet-container { background: #e8edf2; }

        /* City Markers */
        .city-marker {
            background: white;
            border: 2px solid #667eea;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 700;
            color: #667eea;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }

        .city-tooltip {
            background: white;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }
        .city-tooltip .city-name { font-weight: 600; margin-bottom: 4px; }
        .city-tooltip .city-value { color: #667eea; }

        /* Hover tooltip */
        #hover-tooltip {
            position: fixed;
            background: white;
            padding: 8px 12px;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            font-size: 12px;
            pointer-events: none;
            z-index: 2000;
            display: none;
        }
        #hover-tooltip.visible { display: block; }

        @media (max-width: 900px) {
            .container { flex-direction: column; }
            .sidebar { width: 100%; max-height: 45vh; }
            #map { height: 55vh; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>Sundex</h1>
                <p>How many days have weather good enough to get outside and support mental health?</p>
            </div>

            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-value" id="stat-days">--</div>
                    <div class="stat-label">Days in Period</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-min">--</div>
                    <div class="stat-label">Min</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-max">--</div>
                    <div class="stat-label">Max</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-avg">--</div>
                    <div class="stat-label">Average</div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">View Layer</div>
                <div class="layer-toggles">
                    <button class="layer-btn active" data-layer="goodDays">Good Days</button>
                    <button class="layer-btn" data-layer="solar">Solar</button>
                    <button class="layer-btn" data-layer="precip">Precipitation</button>
                    <button class="layer-btn" data-layer="temp">Temperature</button>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Time Period</div>
                <div class="month-presets">
                    <button class="preset-btn" data-months="11,12,1,2">Darkest (Nov-Feb)</button>
                    <button class="preset-btn active" data-months="10,11,12,1,2,3,4">Full Winter</button>
                    <button class="preset-btn" data-months="1,2,3,4,5,6,7,8,9,10,11,12">Year Round</button>
                </div>
                <div class="month-grid">
                    <button class="month-btn" data-month="10">Oct</button>
                    <button class="month-btn" data-month="11">Nov</button>
                    <button class="month-btn" data-month="12">Dec</button>
                    <button class="month-btn" data-month="1">Jan</button>
                    <button class="month-btn" data-month="2">Feb</button>
                    <button class="month-btn" data-month="3">Mar</button>
                    <button class="month-btn" data-month="4">Apr</button>
                    <button class="month-btn disabled" data-month="5">May</button>
                    <button class="month-btn disabled" data-month="6">Jun</button>
                    <button class="month-btn disabled" data-month="7">Jul</button>
                    <button class="month-btn disabled" data-month="8">Aug</button>
                    <button class="month-btn disabled" data-month="9">Sep</button>
                </div>
            </div>

            <div class="section" id="good-days-controls">
                <div class="section-title">Good Day Thresholds</div>
                <div class="control-group">
                    <div class="control-label">
                        <span>Min Sunshine</span>
                        <span class="control-value" id="solar-value">50%</span>
                    </div>
                    <input type="range" id="solar-slider" min="35" max="65" value="50" step="5">
                    <div class="control-hint">Solar transmittance through clouds</div>
                </div>
                <div class="control-group">
                    <div class="control-label">
                        <span>Max Precipitation</span>
                        <span class="control-value" id="precip-value">1.0 mm</span>
                    </div>
                    <input type="range" id="precip-slider" min="0" max="5" value="1" step="0.5">
                    <div class="control-hint">Daily rain threshold for "dry day"</div>
                </div>
            </div>

            <div class="section">
                <div class="section-title" id="legend-title">Good Weather Days</div>
                <div class="legend-bar" id="legend-bar" style="background: linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850);"></div>
                <div class="legend-labels">
                    <span id="legend-min">0</span>
                    <span id="legend-unit">days</span>
                    <span id="legend-max">100</span>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Compare Locations</div>
                <div id="compare-placeholder" class="compare-instructions">
                    Click on the map to select Location A<br>
                    <span style="font-size: 11px; margin-top: 4px; display: inline-block;">Or click a city marker</span>
                </div>
                <div id="compare-content" style="display: none;">
                    <div class="compare-container">
                        <div class="compare-location loc-a" id="loc-a">
                            <button class="compare-clear" onclick="clearLocation('a')">&times;</button>
                            <div class="location-name" id="loc-a-name">Location A</div>
                            <div class="location-stats">
                                <div class="location-stat">
                                    <div class="location-stat-label">Good Days</div>
                                    <div class="location-stat-value" id="loc-a-days">--</div>
                                </div>
                                <div class="location-stat">
                                    <div class="location-stat-label">Solar</div>
                                    <div class="location-stat-value" id="loc-a-solar">--</div>
                                </div>
                                <div class="location-stat">
                                    <div class="location-stat-label">Precip</div>
                                    <div class="location-stat-value" id="loc-a-precip">--</div>
                                </div>
                                <div class="location-stat">
                                    <div class="location-stat-label">Temp</div>
                                    <div class="location-stat-value" id="loc-a-temp">--</div>
                                </div>
                            </div>
                        </div>
                        <div class="compare-location loc-b" id="loc-b" style="opacity: 0.4;">
                            <button class="compare-clear" onclick="clearLocation('b')">&times;</button>
                            <div class="location-name" id="loc-b-name">Click to add B</div>
                            <div class="location-stats">
                                <div class="location-stat">
                                    <div class="location-stat-label">Good Days</div>
                                    <div class="location-stat-value" id="loc-b-days">--</div>
                                </div>
                                <div class="location-stat">
                                    <div class="location-stat-label">Solar</div>
                                    <div class="location-stat-value" id="loc-b-solar">--</div>
                                </div>
                                <div class="location-stat">
                                    <div class="location-stat-label">Precip</div>
                                    <div class="location-stat-value" id="loc-b-precip">--</div>
                                </div>
                                <div class="location-stat">
                                    <div class="location-stat-label">Temp</div>
                                    <div class="location-stat-value" id="loc-b-temp">--</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="compare-diff" id="compare-diff" style="display: none;">
                        <div class="compare-diff-title">Location B has</div>
                        <div class="compare-diff-value" id="diff-value">--</div>
                        <div style="font-size: 11px; color: #6c757d;">more good weather days</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <button class="share-btn" id="share-btn">Copy Link to This View</button>
            </div>

            <div class="sidebar-footer">
                Based on <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8892387/" target="_blank">research</a> showing outdoor daylight reduces depression risk.<br>
                Data: PRISM 30-year normals (1991-2020)
            </div>
        </div>

        <div id="map"></div>
    </div>

    <div id="hover-tooltip"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Climate data
        const climateData = ''' + data_json + ''';

        // State
        let state = {
            layer: 'goodDays',
            months: [10, 11, 12, 1, 2, 3, 4],
            solarMin: 0.50,
            precipMax: 1.0
        };

        // Map setup
        const map = L.map('map').setView([47.4, -120.5], 7);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            subdomains: 'abcd'
        }).addTo(map);

        let imageOverlay = null;
        let cityMarkers = [];

        // Layer configs
        const layerConfigs = {
            goodDays: {
                name: 'Good Weather Days',
                unit: 'days',
                gradient: 'linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850)',
                colors: [[215,48,39],[252,141,89],[254,224,139],[217,239,139],[145,207,96],[26,152,80]]
            },
            solar: {
                name: 'Solar Transmittance',
                unit: '%',
                gradient: 'linear-gradient(to right, #4a1486, #807dba, #fec44f, #fe9929, #ec7014)',
                colors: [[74,20,134],[128,125,186],[254,196,79],[254,153,41],[236,112,20]]
            },
            precip: {
                name: 'Monthly Precipitation',
                unit: 'mm',
                gradient: 'linear-gradient(to right, #f7fcb4, #addd8e, #41ab5d, #006837, #004529)',
                colors: [[247,252,180],[173,221,142],[65,171,93],[0,104,55],[0,69,41]],
                reverse: true
            },
            temp: {
                name: 'Average Temperature',
                unit: '°C',
                gradient: 'linear-gradient(to right, #313695, #74add1, #ffffbf, #f46d43, #a50026)',
                colors: [[49,54,149],[116,173,209],[255,255,191],[244,109,67],[165,0,38]]
            }
        };

        // Normal CDF
        function normCdf(x) {
            const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
            const sign = x < 0 ? -1 : 1;
            x = Math.abs(x) / Math.sqrt(2);
            const t = 1 / (1 + p * x);
            const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * Math.exp(-x*x);
            return 0.5 * (1 + sign * y);
        }

        // Get data for selected months
        function getMonthlyAverage(varName) {
            const { rows, cols } = climateData.shape;
            const result = [];
            for (let r = 0; r < rows; r++) {
                const row = [];
                for (let c = 0; c < cols; c++) {
                    let sum = 0, count = 0;
                    for (const m of state.months) {
                        const val = climateData.data[varName]?.[m]?.[r]?.[c];
                        if (val !== null && val !== undefined) {
                            sum += val;
                            count++;
                        }
                    }
                    row.push(count > 0 ? sum / count : null);
                }
                result.push(row);
            }
            return result;
        }

        // Calculate good days
        function calculateGoodDays() {
            const { rows, cols } = climateData.shape;
            const result = [];
            const solarStd = 0.10;

            let totalDays = 0;
            for (const m of state.months) {
                totalDays += climateData.months[m]?.days || 30;
            }

            for (let r = 0; r < rows; r++) {
                const row = [];
                for (let c = 0; c < cols; c++) {
                    let goodDays = 0;

                    for (const m of state.months) {
                        const sol = climateData.data.soltrans?.[m]?.[r]?.[c];
                        const ppt = climateData.data.ppt?.[m]?.[r]?.[c];
                        const days = climateData.months[m]?.days || 30;

                        if (sol === null || ppt === null) continue;

                        // Solar fraction
                        const zSolar = (state.solarMin - sol) / solarStd;
                        const fracSolar = 1 - normCdf(zSolar);

                        // Precip fraction
                        const dailyPrecip = ppt / days;
                        const fracDry = Math.max(0.05, Math.min(0.95, 0.95 - dailyPrecip / (state.precipMax * 8)));

                        goodDays += fracSolar * fracDry * days;
                    }

                    row.push(goodDays > 0 ? goodDays : null);
                }
                result.push(row);
            }

            return { data: result, totalDays };
        }

        // Get color from value
        function getColor(value, min, max, colors, reverse = false) {
            if (value === null) return 'rgba(0,0,0,0)';
            let t = Math.max(0, Math.min(1, (value - min) / (max - min)));
            if (reverse) t = 1 - t;

            const idx = t * (colors.length - 1);
            const i = Math.floor(idx);
            const f = idx - i;

            if (i >= colors.length - 1) return `rgba(${colors[colors.length-1].join(',')},0.8)`;

            const c1 = colors[i], c2 = colors[i+1];
            const r = Math.round(c1[0] + f * (c2[0] - c1[0]));
            const g = Math.round(c1[1] + f * (c2[1] - c1[1]));
            const b = Math.round(c1[2] + f * (c2[2] - c1[2]));
            return `rgba(${r},${g},${b},0.8)`;
        }

        // Get stats from data
        function getStats(data) {
            let min = Infinity, max = -Infinity, sum = 0, count = 0;
            for (const row of data) {
                for (const val of row) {
                    if (val !== null) {
                        min = Math.min(min, val);
                        max = Math.max(max, val);
                        sum += val;
                        count++;
                    }
                }
            }
            return { min, max, avg: count > 0 ? sum / count : 0 };
        }

        // Render map
        function render() {
            const config = layerConfigs[state.layer];
            let data, stats, totalDays;

            if (state.layer === 'goodDays') {
                const result = calculateGoodDays();
                data = result.data;
                totalDays = result.totalDays;
                stats = getStats(data);
            } else if (state.layer === 'solar') {
                data = getMonthlyAverage('soltrans');
                // Convert to percentage
                data = data.map(row => row.map(v => v !== null ? v * 100 : null));
                stats = getStats(data);
                totalDays = state.months.reduce((s, m) => s + (climateData.months[m]?.days || 0), 0);
            } else if (state.layer === 'precip') {
                data = getMonthlyAverage('ppt');
                stats = getStats(data);
                totalDays = state.months.reduce((s, m) => s + (climateData.months[m]?.days || 0), 0);
            } else if (state.layer === 'temp') {
                data = getMonthlyAverage('tmean');
                stats = getStats(data);
                totalDays = state.months.reduce((s, m) => s + (climateData.months[m]?.days || 0), 0);
            }

            // Update stats display
            document.getElementById('stat-days').textContent = totalDays;
            document.getElementById('stat-min').textContent = stats.min.toFixed(state.layer === 'goodDays' ? 0 : 1);
            document.getElementById('stat-max').textContent = stats.max.toFixed(state.layer === 'goodDays' ? 0 : 1);
            document.getElementById('stat-avg').textContent = stats.avg.toFixed(state.layer === 'goodDays' ? 0 : 1);

            // Update legend
            document.getElementById('legend-title').textContent = config.name;
            document.getElementById('legend-bar').style.background = config.gradient;
            document.getElementById('legend-min').textContent = stats.min.toFixed(state.layer === 'goodDays' ? 0 : 1);
            document.getElementById('legend-max').textContent = stats.max.toFixed(state.layer === 'goodDays' ? 0 : 1);
            document.getElementById('legend-unit').textContent = config.unit;

            // Show/hide controls
            document.getElementById('good-days-controls').style.display = state.layer === 'goodDays' ? 'block' : 'none';

            // Render canvas
            const { rows, cols } = climateData.shape;
            const canvas = document.createElement('canvas');
            canvas.width = cols;
            canvas.height = rows;
            const ctx = canvas.getContext('2d');

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    ctx.fillStyle = getColor(data[r][c], stats.min, stats.max, config.colors, config.reverse);
                    ctx.fillRect(c, r, 1, 1);
                }
            }

            // Update overlay
            if (imageOverlay) map.removeLayer(imageOverlay);
            const { north, south, west, east } = climateData.bounds;
            imageOverlay = L.imageOverlay(canvas.toDataURL(), [[south, west], [north, east]], { opacity: 0.75 }).addTo(map);

            // Update city markers
            updateCityMarkers(data, stats);

            // Store current data for hover/click
            window.currentData = data;
            window.currentStats = stats;
        }

        // Update city markers
        function updateCityMarkers(data, stats) {
            cityMarkers.forEach(m => map.removeLayer(m));
            cityMarkers = [];

            const { rows, cols } = climateData.shape;
            const { north, south, west, east } = climateData.bounds;

            for (const city of climateData.cities) {
                // Find grid cell
                const col = Math.floor((city.lon - west) / climateData.cellSize);
                const row = Math.floor((north - city.lat) / climateData.cellSize);

                if (row >= 0 && row < rows && col >= 0 && col < cols) {
                    const val = data[row]?.[col];
                    if (val !== null) {
                        const config = layerConfigs[state.layer];
                        const displayVal = state.layer === 'goodDays' ? Math.round(val) : val.toFixed(1);

                        const icon = L.divIcon({
                            className: 'city-marker',
                            html: displayVal,
                            iconSize: [32, 32],
                            iconAnchor: [16, 16]
                        });

                        const marker = L.marker([city.lat, city.lon], { icon })
                            .bindTooltip(`<div class="city-tooltip"><div class="city-name">${city.name}</div><div class="city-value">${displayVal} ${config.unit}</div></div>`, { permanent: false, direction: 'top', offset: [0, -15] })
                            .on('click', () => {
                                // Fill comparison slot with city
                                if (!compareLocations.a) {
                                    setLocation('a', city.lat, city.lon, city.name);
                                } else if (!compareLocations.b) {
                                    setLocation('b', city.lat, city.lon, city.name);
                                } else {
                                    setLocation('b', city.lat, city.lon, city.name);
                                }
                            })
                            .addTo(map);

                        cityMarkers.push(marker);
                    }
                }
            }
        }

        // Get value at lat/lon
        function getValueAt(lat, lon) {
            const { rows, cols } = climateData.shape;
            const { north, south, west, east } = climateData.bounds;

            const col = Math.floor((lon - west) / climateData.cellSize);
            const row = Math.floor((north - lat) / climateData.cellSize);

            if (row >= 0 && row < rows && col >= 0 && col < cols && window.currentData) {
                return window.currentData[row]?.[col];
            }
            return null;
        }

        // Get all values at location
        function getAllValuesAt(lat, lon) {
            const { rows, cols } = climateData.shape;
            const { north, west } = climateData.bounds;

            const col = Math.floor((lon - west) / climateData.cellSize);
            const row = Math.floor((north - lat) / climateData.cellSize);

            if (row < 0 || row >= rows || col < 0 || col >= cols) return null;

            // Calculate good days for this cell
            const result = calculateGoodDays();
            const goodDays = result.data[row]?.[col];

            // Get averages
            let solarSum = 0, precipSum = 0, tempSum = 0, count = 0;
            for (const m of state.months) {
                const sol = climateData.data.soltrans?.[m]?.[row]?.[col];
                const ppt = climateData.data.ppt?.[m]?.[row]?.[col];
                const tmp = climateData.data.tmean?.[m]?.[row]?.[col];
                if (sol !== null && ppt !== null && tmp !== null) {
                    solarSum += sol;
                    precipSum += ppt;
                    tempSum += tmp;
                    count++;
                }
            }

            if (count === 0) return null;

            return {
                goodDays: goodDays !== null ? Math.round(goodDays) : null,
                solar: Math.round(solarSum / count * 100),
                precip: Math.round(precipSum / count),
                temp: (tempSum / count).toFixed(1)
            };
        }

        // Update URL
        function updateUrl() {
            const params = new URLSearchParams();
            params.set('layer', state.layer);
            params.set('months', state.months.join(','));
            params.set('solar', Math.round(state.solarMin * 100));
            params.set('precip', state.precipMax);
            history.replaceState(null, '', '?' + params.toString());
        }

        // Load from URL
        function loadFromUrl() {
            const params = new URLSearchParams(window.location.search);
            if (params.has('layer')) state.layer = params.get('layer');
            if (params.has('months')) state.months = params.get('months').split(',').map(Number);
            if (params.has('solar')) state.solarMin = parseInt(params.get('solar')) / 100;
            if (params.has('precip')) state.precipMax = parseFloat(params.get('precip'));
        }

        // Update UI from state
        function updateUI() {
            // Layer buttons
            document.querySelectorAll('.layer-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.layer === state.layer);
            });

            // Month buttons
            document.querySelectorAll('.month-btn').forEach(btn => {
                const m = parseInt(btn.dataset.month);
                btn.classList.toggle('active', state.months.includes(m));
            });

            // Preset buttons
            document.querySelectorAll('.preset-btn').forEach(btn => {
                const presetMonths = btn.dataset.months.split(',').map(Number);
                const match = presetMonths.length === state.months.length &&
                    presetMonths.every(m => state.months.includes(m));
                btn.classList.toggle('active', match);
            });

            // Sliders
            document.getElementById('solar-slider').value = state.solarMin * 100;
            document.getElementById('solar-value').textContent = Math.round(state.solarMin * 100) + '%';
            document.getElementById('precip-slider').value = state.precipMax;
            document.getElementById('precip-value').textContent = state.precipMax.toFixed(1) + ' mm';
        }

        // Event listeners
        document.querySelectorAll('.layer-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                state.layer = btn.dataset.layer;
                updateUI();
                render();
                updateUrl();
            });
        });

        document.querySelectorAll('.month-btn:not(.disabled)').forEach(btn => {
            btn.addEventListener('click', () => {
                const m = parseInt(btn.dataset.month);
                if (state.months.includes(m)) {
                    if (state.months.length > 1) {
                        state.months = state.months.filter(x => x !== m);
                    }
                } else {
                    state.months.push(m);
                    state.months.sort((a, b) => a - b);
                }
                updateUI();
                render();
                updateUrl();
            });
        });

        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                state.months = btn.dataset.months.split(',').map(Number);
                updateUI();
                render();
                updateUrl();
            });
        });

        document.getElementById('solar-slider').addEventListener('input', (e) => {
            state.solarMin = parseInt(e.target.value) / 100;
            document.getElementById('solar-value').textContent = e.target.value + '%';
            render();
            updateUrl();
        });

        document.getElementById('precip-slider').addEventListener('input', (e) => {
            state.precipMax = parseFloat(e.target.value);
            document.getElementById('precip-value').textContent = state.precipMax.toFixed(1) + ' mm';
            render();
            updateUrl();
        });

        // Comparison state
        let compareLocations = { a: null, b: null };
        let compareMarkers = { a: null, b: null };

        function setLocation(which, lat, lon, name) {
            const vals = getAllValuesAt(lat, lon);
            if (!vals) return;

            compareLocations[which] = { lat, lon, name, vals };

            // Update UI
            document.getElementById('compare-placeholder').style.display = 'none';
            document.getElementById('compare-content').style.display = 'block';

            const prefix = 'loc-' + which;
            document.getElementById(prefix + '-name').textContent = name || `${lat.toFixed(2)}°N, ${Math.abs(lon).toFixed(2)}°W`;
            document.getElementById(prefix + '-days').textContent = vals.goodDays !== null ? vals.goodDays : '--';
            document.getElementById(prefix + '-solar').textContent = vals.solar + '%';
            document.getElementById(prefix + '-precip').textContent = vals.precip + ' mm';
            document.getElementById(prefix + '-temp').textContent = vals.temp + '°C';
            document.getElementById('loc-' + which).style.opacity = '1';

            // Add/update marker
            if (compareMarkers[which]) {
                map.removeLayer(compareMarkers[which]);
            }
            const markerClass = which === 'a' ? 'marker-a' : 'marker-b';
            const icon = L.divIcon({
                className: 'map-marker ' + markerClass,
                html: which.toUpperCase(),
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });
            compareMarkers[which] = L.marker([lat, lon], { icon }).addTo(map);

            // Update difference if both locations set
            updateDiff();
        }

        function clearLocation(which) {
            compareLocations[which] = null;
            if (compareMarkers[which]) {
                map.removeLayer(compareMarkers[which]);
                compareMarkers[which] = null;
            }

            const prefix = 'loc-' + which;
            document.getElementById(prefix + '-name').textContent = which === 'a' ? 'Location A' : 'Click to add B';
            document.getElementById(prefix + '-days').textContent = '--';
            document.getElementById(prefix + '-solar').textContent = '--';
            document.getElementById(prefix + '-precip').textContent = '--';
            document.getElementById(prefix + '-temp').textContent = '--';
            document.getElementById('loc-' + which).style.opacity = '0.4';

            if (!compareLocations.a && !compareLocations.b) {
                document.getElementById('compare-placeholder').style.display = 'block';
                document.getElementById('compare-content').style.display = 'none';
            }

            updateDiff();
        }

        function updateDiff() {
            const diffEl = document.getElementById('compare-diff');
            if (compareLocations.a && compareLocations.b) {
                const diff = compareLocations.b.vals.goodDays - compareLocations.a.vals.goodDays;
                const diffEl = document.getElementById('compare-diff');
                diffEl.style.display = 'block';

                const diffValue = document.getElementById('diff-value');
                if (diff > 0) {
                    diffValue.textContent = '+' + Math.round(diff);
                    diffValue.className = 'compare-diff-value positive';
                } else if (diff < 0) {
                    diffValue.textContent = Math.round(diff);
                    diffValue.className = 'compare-diff-value negative';
                } else {
                    diffValue.textContent = '0';
                    diffValue.className = 'compare-diff-value';
                }
            } else {
                diffEl.style.display = 'none';
            }
        }

        // Map click
        map.on('click', (e) => {
            const vals = getAllValuesAt(e.latlng.lat, e.latlng.lng);
            if (vals) {
                // Determine which slot to fill
                if (!compareLocations.a) {
                    setLocation('a', e.latlng.lat, e.latlng.lng);
                } else if (!compareLocations.b) {
                    setLocation('b', e.latlng.lat, e.latlng.lng);
                } else {
                    // Both filled, replace B
                    setLocation('b', e.latlng.lat, e.latlng.lng);
                }
            }
        });

        // Hover tooltip
        const tooltip = document.getElementById('hover-tooltip');
        map.on('mousemove', (e) => {
            const val = getValueAt(e.latlng.lat, e.latlng.lng);
            if (val !== null) {
                const config = layerConfigs[state.layer];
                const displayVal = state.layer === 'goodDays' ? Math.round(val) : val.toFixed(1);
                tooltip.textContent = `${displayVal} ${config.unit}`;
                tooltip.style.left = (e.originalEvent.clientX + 15) + 'px';
                tooltip.style.top = (e.originalEvent.clientY - 10) + 'px';
                tooltip.classList.add('visible');
            } else {
                tooltip.classList.remove('visible');
            }
        });

        map.on('mouseout', () => {
            tooltip.classList.remove('visible');
        });

        // Share button
        document.getElementById('share-btn').addEventListener('click', () => {
            navigator.clipboard.writeText(window.location.href).then(() => {
                const btn = document.getElementById('share-btn');
                btn.textContent = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy Link to This View';
                    btn.classList.remove('copied');
                }, 2000);
            });
        });

        // Initialize
        loadFromUrl();
        updateUI();
        render();
    </script>
</body>
</html>'''


if __name__ == '__main__':
    main()
