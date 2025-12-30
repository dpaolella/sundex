"""
Build a standalone Sundex web application.
Exports data as JSON and generates a professional single-page HTML app.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import json

# Constants
DATA_DIR = "Data/prism_normals"
WA_BOUNDS = {"west": -124.85, "east": -116.90, "south": 45.50, "north": 49.05}
WINTER_MONTHS = {10: 31, 11: 30, 12: 31, 1: 31, 2: 28, 3: 31, 4: 30}

def find_raster_file(pattern_dir):
    if not os.path.isdir(pattern_dir):
        return None
    for f in os.listdir(pattern_dir):
        if f.endswith('.bil') or f.endswith('.tif'):
            return os.path.join(pattern_dir, f)
    return None

def get_file_paths():
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

def load_and_average_data():
    """Load data and compute winter averages."""
    file_paths = get_file_paths()
    averages = {}
    transform = None

    for var, month_files in file_paths.items():
        monthly_arrays = []
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

                    nodata = src.nodata if src.nodata else -9999
                    data = np.where(data == nodata, np.nan, data)
                    monthly_arrays.append(data)

        if monthly_arrays:
            # Ensure all arrays have the same shape
            target_shape = monthly_arrays[0].shape
            aligned = []
            for arr in monthly_arrays:
                if arr.shape != target_shape:
                    new_arr = np.full(target_shape, np.nan)
                    min_r = min(arr.shape[0], target_shape[0])
                    min_c = min(arr.shape[1], target_shape[1])
                    new_arr[:min_r, :min_c] = arr[:min_r, :min_c]
                    aligned.append(new_arr)
                else:
                    aligned.append(arr)
            averages[var] = np.nanmean(np.stack(aligned), axis=0)

    return averages, transform

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

def main():
    print("Loading and processing PRISM data...")
    averages, transform = load_and_average_data()

    # Downsample for web (reduces data size by 4x)
    factor = 2
    downsampled = {var: downsample(arr, factor) for var, arr in averages.items()}

    # Adjust transform for downsampled data
    new_transform = rasterio.Affine(
        transform.a * factor,
        transform.b,
        transform.c,
        transform.d,
        transform.e * factor,
        transform.f
    )

    shape = downsampled['ppt'].shape
    rows, cols = shape

    # Calculate bounds
    west, north = new_transform * (0, 0)
    east, south = new_transform * (cols, rows)

    print(f"Data shape: {shape}")
    print(f"Bounds: N={north:.2f}, S={south:.2f}, W={west:.2f}, E={east:.2f}")

    # Convert to lists for JSON (replace NaN with null)
    def to_json_list(arr):
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

    data_export = {
        'bounds': {'north': north, 'south': south, 'west': west, 'east': east},
        'shape': {'rows': rows, 'cols': cols},
        'cellSize': abs(new_transform.a),
        'winterMonths': WINTER_MONTHS,
        'totalDays': sum(WINTER_MONTHS.values()),
        'data': {
            'soltrans': to_json_list(downsampled['soltrans']),
            'ppt': to_json_list(downsampled['ppt']),
            'tmean': to_json_list(downsampled['tmean']),
        }
    }

    # Calculate data stats
    for var in ['soltrans', 'ppt', 'tmean']:
        arr = downsampled[var]
        valid = arr[~np.isnan(arr)]
        print(f"{var}: min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}")

    json_str = json.dumps(data_export)
    print(f"\nJSON data size: {len(json_str) / 1024:.0f} KB")

    # Generate the HTML app
    html_content = generate_html(json_str)

    with open('index.html', 'w') as f:
        f.write(html_content)

    print(f"Created index.html")

def generate_html(data_json):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sundex - Winter Weather Experience Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            color: #333;
        }}

        .container {{
            display: flex;
            height: 100vh;
        }}

        .sidebar {{
            width: 380px;
            background: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            overflow-y: auto;
            z-index: 1000;
        }}

        .sidebar-header {{
            padding: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .sidebar-header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .sidebar-header p {{
            font-size: 14px;
            opacity: 0.9;
            line-height: 1.5;
        }}

        .stats-bar {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}

        .stat {{
            flex: 1;
            padding: 16px;
            text-align: center;
            border-right: 1px solid #e9ecef;
        }}

        .stat:last-child {{
            border-right: none;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }}

        .stat-label {{
            font-size: 11px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}

        .controls {{
            padding: 24px;
        }}

        .controls h2 {{
            font-size: 14px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .control-group {{
            margin-bottom: 24px;
        }}

        .control-label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .control-label span {{
            font-size: 14px;
            color: #495057;
        }}

        .control-value {{
            font-size: 14px;
            font-weight: 600;
            color: #667eea;
            background: #f0f1ff;
            padding: 4px 10px;
            border-radius: 4px;
        }}

        input[type="range"] {{
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #e9ecef;
            outline: none;
            -webkit-appearance: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.4);
        }}

        input[type="range"]::-webkit-slider-thumb:hover {{
            background: #5a6fd6;
        }}

        .control-hint {{
            font-size: 12px;
            color: #868e96;
            margin-top: 6px;
            line-height: 1.4;
        }}

        .legend {{
            padding: 20px 24px;
            border-top: 1px solid #e9ecef;
        }}

        .legend h3 {{
            font-size: 12px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .legend-bar {{
            height: 12px;
            border-radius: 6px;
            background: linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850);
            margin-bottom: 6px;
        }}

        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #6c757d;
        }}

        .info-section {{
            padding: 20px 24px;
            border-top: 1px solid #e9ecef;
        }}

        .info-section h3 {{
            font-size: 12px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .info-section p {{
            font-size: 13px;
            color: #6c757d;
            line-height: 1.6;
        }}

        .info-section a {{
            color: #667eea;
            text-decoration: none;
        }}

        .info-section a:hover {{
            text-decoration: underline;
        }}

        #map {{
            flex: 1;
            height: 100vh;
        }}

        .leaflet-container {{
            background: #e8edf2;
        }}

        @media (max-width: 768px) {{
            .container {{
                flex-direction: column;
            }}
            .sidebar {{
                width: 100%;
                height: auto;
                max-height: 50vh;
            }}
            #map {{
                height: 50vh;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>Sundex</h1>
                <p>How many days during the dark months have weather good enough to get outside and support mental health?</p>
            </div>

            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-value" id="stat-min">--</div>
                    <div class="stat-label">Min Days</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-max">--</div>
                    <div class="stat-label">Max Days</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-avg">--</div>
                    <div class="stat-label">WA Average</div>
                </div>
            </div>

            <div class="controls">
                <h2>Define a "Good Day"</h2>

                <div class="control-group">
                    <div class="control-label">
                        <span>&#9728;&#65039; Minimum Sunshine</span>
                        <span class="control-value" id="solar-value">50%</span>
                    </div>
                    <input type="range" id="solar-slider" min="30" max="70" value="50" step="5">
                    <div class="control-hint">Solar transmittance through clouds. Seattle averages ~45%, Eastern WA ~55%.</div>
                </div>

                <div class="control-group">
                    <div class="control-label">
                        <span>&#127783;&#65039; Maximum Precipitation</span>
                        <span class="control-value" id="precip-value">1.0 mm</span>
                    </div>
                    <input type="range" id="precip-slider" min="0" max="5" value="1" step="0.5">
                    <div class="control-hint">Daily rain/snow threshold. Lower = stricter "dry day" definition.</div>
                </div>

                <div class="control-group">
                    <div class="control-label">
                        <span>&#127777;&#65039; Minimum Temperature</span>
                        <span class="control-value" id="temp-value">0&deg;C</span>
                    </div>
                    <input type="range" id="temp-slider" min="-10" max="10" value="0" step="1">
                    <div class="control-hint">Cold enough to want to stay inside? Set your threshold.</div>
                </div>
            </div>

            <div class="legend">
                <h3>Good Weather Days (Oct-Apr)</h3>
                <div class="legend-bar"></div>
                <div class="legend-labels">
                    <span id="legend-min">0</span>
                    <span>out of 212 days</span>
                    <span id="legend-max">212</span>
                </div>
            </div>

            <div class="info-section">
                <h3>About</h3>
                <p>
                    Based on <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8892387/" target="_blank">research</a> showing
                    that each hour of outdoor daylight exposure reduces depression risk. This map estimates how many
                    winter days meet your criteria for "good enough weather to go outside."
                </p>
                <p style="margin-top: 12px; font-size: 11px; color: #adb5bd;">
                    Data: PRISM 30-year climate normals &bull; Region: Washington State
                </p>
            </div>
        </div>

        <div id="map"></div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Embedded climate data
        const climateData = {data_json};

        // Initialize map
        const map = L.map('map').setView([47.5, -120.5], 7);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);

        let imageOverlay = null;

        // Normal CDF approximation
        function normCdf(x) {{
            const a1 =  0.254829592;
            const a2 = -0.284496736;
            const a3 =  1.421413741;
            const a4 = -1.453152027;
            const a5 =  1.061405429;
            const p  =  0.3275911;

            const sign = x < 0 ? -1 : 1;
            x = Math.abs(x) / Math.sqrt(2);

            const t = 1.0 / (1.0 + p * x);
            const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

            return 0.5 * (1.0 + sign * y);
        }}

        function calculateGoodDays(solarMin, precipMax, tempMin) {{
            const {{ rows, cols }} = climateData.shape;
            const {{ soltrans, ppt, tmean }} = climateData.data;
            const totalDays = climateData.totalDays;

            const result = [];
            let min = Infinity, max = -Infinity, sum = 0, count = 0;

            const solarStd = 0.12;
            const tempStd = 5;

            for (let r = 0; r < rows; r++) {{
                const row = [];
                for (let c = 0; c < cols; c++) {{
                    const sol = soltrans[r][c];
                    const precip = ppt[r][c];
                    const temp = tmean[r][c];

                    if (sol === null || precip === null || temp === null) {{
                        row.push(null);
                        continue;
                    }}

                    // Solar fraction
                    const zSolar = (solarMin - sol) / solarStd;
                    const fracSolar = 1 - normCdf(zSolar);

                    // Precip fraction (simplified model)
                    const dailyPrecip = precip / 30; // avg daily
                    const fracDry = Math.max(0.1, Math.min(0.95, 0.95 - dailyPrecip / (precipMax * 6)));

                    // Temp fraction
                    const zTemp = (tempMin - temp) / tempStd;
                    const fracWarm = Math.max(0.1, Math.min(1.0, 1 - normCdf(zTemp)));

                    // Combined
                    const goodDays = fracSolar * fracDry * fracWarm * totalDays;
                    row.push(goodDays);

                    if (goodDays < min) min = goodDays;
                    if (goodDays > max) max = goodDays;
                    sum += goodDays;
                    count++;
                }}
                result.push(row);
            }}

            return {{ data: result, min, max, avg: sum / count }};
        }}

        function getColor(value, min, max) {{
            if (value === null) return 'rgba(0,0,0,0)';

            const t = Math.max(0, Math.min(1, (value - min) / (max - min)));

            // RdYlGn colormap
            const colors = [
                [215, 48, 39],
                [252, 141, 89],
                [254, 224, 139],
                [217, 239, 139],
                [145, 207, 96],
                [26, 152, 80]
            ];

            const idx = t * (colors.length - 1);
            const i = Math.floor(idx);
            const f = idx - i;

            if (i >= colors.length - 1) {{
                return `rgba(${{colors[colors.length-1].join(',')}}, 0.8)`;
            }}

            const c1 = colors[i];
            const c2 = colors[i + 1];
            const r = Math.round(c1[0] + f * (c2[0] - c1[0]));
            const g = Math.round(c1[1] + f * (c2[1] - c1[1]));
            const b = Math.round(c1[2] + f * (c2[2] - c1[2]));

            return `rgba(${{r}},${{g}},${{b}},0.8)`;
        }}

        function renderMap(result) {{
            const {{ data, min, max, avg }} = result;
            const {{ rows, cols }} = climateData.shape;
            const {{ north, south, west, east }} = climateData.bounds;

            // Update stats
            document.getElementById('stat-min').textContent = Math.round(min);
            document.getElementById('stat-max').textContent = Math.round(max);
            document.getElementById('stat-avg').textContent = Math.round(avg);
            document.getElementById('legend-min').textContent = Math.round(min);
            document.getElementById('legend-max').textContent = Math.round(max);

            // Create canvas
            const canvas = document.createElement('canvas');
            canvas.width = cols;
            canvas.height = rows;
            const ctx = canvas.getContext('2d');

            for (let r = 0; r < rows; r++) {{
                for (let c = 0; c < cols; c++) {{
                    ctx.fillStyle = getColor(data[r][c], min, max);
                    ctx.fillRect(c, r, 1, 1);
                }}
            }}

            // Remove old overlay
            if (imageOverlay) {{
                map.removeLayer(imageOverlay);
            }}

            // Add new overlay
            const bounds = [[south, west], [north, east]];
            imageOverlay = L.imageOverlay(canvas.toDataURL(), bounds, {{
                opacity: 0.75,
                interactive: false
            }}).addTo(map);
        }}

        function update() {{
            const solarMin = parseFloat(document.getElementById('solar-slider').value) / 100;
            const precipMax = parseFloat(document.getElementById('precip-slider').value);
            const tempMin = parseFloat(document.getElementById('temp-slider').value);

            // Update display values
            document.getElementById('solar-value').textContent = Math.round(solarMin * 100) + '%';
            document.getElementById('precip-value').textContent = precipMax.toFixed(1) + ' mm';
            document.getElementById('temp-value').textContent = tempMin + '°C';

            const result = calculateGoodDays(solarMin, precipMax, tempMin);
            renderMap(result);
        }}

        // Event listeners
        document.getElementById('solar-slider').addEventListener('input', update);
        document.getElementById('precip-slider').addEventListener('input', update);
        document.getElementById('temp-slider').addEventListener('input', update);

        // Initial render
        update();
    </script>
</body>
</html>'''

if __name__ == '__main__':
    main()
