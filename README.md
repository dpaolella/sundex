# Sundex - Winter Weather Experience Map

Interactive map showing how many "good weather days" different locations in Washington State experience during the dark months (October - April).

**[View Live Map](https://dpaolella.github.io/sundex/)**

## What is a "Good Day"?

A day with weather good enough to comfortably spend time outdoors, supporting mental health during winter. Based on [research](https://pmc.ncbi.nlm.nih.gov/articles/PMC8892387/) showing that outdoor daylight exposure reduces depression risk.

The default thresholds are:
- **Solar transmittance ≥ 50%** - enough sunlight reaching the ground through clouds
- **Precipitation < 1 mm/day** - dry enough to be outside comfortably

## Features

- **Multiple view layers**: Good Days, Solar Transmittance, Precipitation, Temperature
- **Customizable time periods**: Darkest months (Nov-Feb), Full Winter (Oct-Apr), or Year Round
- **Adjustable thresholds**: Define what "good weather" means for you
- **Location comparison**: Click two spots to compare them side-by-side
- **City markers**: Quick reference for major Washington cities
- **URL sharing**: Share your exact view settings with others

## Data Source

[PRISM Climate Group](https://prism.oregonstate.edu/) 30-year normals (1991-2020) at 4km resolution.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild index.html from PRISM data
python build_webapp_v2.py

# Serve locally
python -m http.server 8080
```

## License

MIT License - see [LICENSE](LICENSE) file.
