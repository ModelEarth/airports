# Airports / Runways

Merges [wiki state airport table](https://en.wikipedia.org/wiki/List_of_airports_in_Georgia_(U.S._state)) with latitude/longitude from both the faa.gov KML and data.gov runways.csv files, including up to 3 runways per airport with detailed specs.

## Setup and Usage

### First-time Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install requests beautifulsoup4 pandas lxml
```

### Run the Script

```bash
python3 pull-airports.py "Georgia"
```

This will:
- Fetch airport data from Wikipedia
- Merge with KML coordinates from `us/ga/ga.kml`
- Merge with runway data from `runways.csv`
- Output to `us/ga/ga.csv`
- Generate a report at `pull-report.md`

### Runways File

The cvs "Runways" file was download Jan 3, 2026 from:

https://ngda-transportation-geoplatform.hub.arcgis.com/api/download/v1/items/110af7b8a9424a59a3fb1d8fc69a2172/csv?layers=0

The link above was provided by: [catalog-beta.data.gov](https://catalog-beta.data.gov/dataset/runways-6de7a?from_hint=eyJxIjoiRkFBIGFpcnBvcnRzIiwic29ydCI6InJlbGV2YW5jZSJ9)


### KML File

ga.kml file was manually pulled Jan 31, 2026 using the "Display on map" button at 
https://adip.faa.gov/agis/public/#/airportSearch/advanced