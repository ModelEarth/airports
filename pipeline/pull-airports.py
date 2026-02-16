#!/usr/bin/env python3
"""
Fetch airport data from Wikipedia and merge with runway and KML data.
Usage: python pull-airports.py --config config.yaml
"""

import argparse
import csv
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
import yaml
from bs4 import BeautifulSoup
import pandas as pd


# State name to abbreviation mapping
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Reverse mapping for state abbreviations
ABBREV_STATE = {abbr: name for name, abbr in STATE_ABBREV.items()}

# Role code to full name mapping
ROLE_NAMES = {
    'P-L': 'Large Hub',
    'P-M': 'Medium Hub',
    'P-S': 'Small Hub',
    'P-N': 'Nonhub',
    'CS': 'Commercial Service Nonprimary',
    'R': 'Reliever',
    'GA': 'General Aviation'
}


def expand_role_name(role_code):
    """Convert role code to full name."""
    if not role_code:
        return ''
    return ROLE_NAMES.get(role_code, role_code)


def find_repo_root(start_path):
    """Find repository root by walking up to .git."""
    start = Path(start_path).resolve()
    for parent in [start] + list(start.parents):
        if (parent / '.git').exists():
            return parent
    return start


def resolve_path(path_value, repo_root, script_dir, prefer_script_dir=False):
    """Resolve relative paths against repo root or script dir."""
    if not path_value:
        return None
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    if prefer_script_dir:
        return script_dir / path_obj
    return repo_root / path_obj


def load_config(config_path):
    """Load YAML config."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    return config


def parse_state_list(state_value):
    """Parse STATE config which may be a comma-separated string or list."""
    if isinstance(state_value, list):
        states = [str(s).strip() for s in state_value if str(s).strip()]
    elif isinstance(state_value, str):
        states = [s.strip() for s in state_value.split(',') if s.strip()]
    else:
        states = []
    return [s.upper() for s in states]


def template_context(country, state_abbrev, state_name):
    return {
        'country': country.lower(),
        'country_upper': country.upper(),
        'state': state_abbrev.lower(),
        'state_upper': state_abbrev.upper(),
        'statename': state_name,
    }


def resolve_state_value(value, state_abbrev, state_name):
    """Resolve config values that can be a mapping or a string."""
    if isinstance(value, dict):
        return (
            value.get(state_abbrev)
            or value.get(state_abbrev.lower())
            or value.get(state_name)
        )
    return value


def build_wikipedia_url(config_value, context, state_abbrev, state_name):
    """Build Wikipedia URL from template or per-state mapping."""
    resolved = resolve_state_value(config_value, state_abbrev, state_name)
    if not resolved:
        raise ValueError("WIKIPEDIA_AIRPORT_STATE is missing from config")
    if isinstance(resolved, str):
        try:
            return resolved.format(**context)
        except KeyError as exc:
            raise ValueError(f"Unknown template key in WIKIPEDIA_AIRPORT_STATE: {exc}") from exc
    raise ValueError("WIKIPEDIA_AIRPORT_STATE must be a string or mapping")


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def airport_code(airport):
    for key in ['IATA', 'FAA']:
        code = str(airport.get(key, '')).strip().upper()
        if len(code) == 3 and code.replace('-', '').isalnum():
            return code
    return ''


def build_airport_candidates(merged_data):
    candidates = []
    for airport in merged_data:
        code = airport_code(airport)
        lat = parse_float(airport.get('Latitude'))
        lon = parse_float(airport.get('Longitude'))
        if code and lat is not None and lon is not None:
            candidates.append((code, lat, lon))
    return candidates


def nearest_airports(lat, lon, airport_candidates, count=3):
    distances = []
    for code, apt_lat, apt_lon in airport_candidates:
        dist = haversine_km(lat, lon, apt_lat, apt_lon)
        distances.append((dist, code))
    distances.sort(key=lambda x: x[0])
    return [code for _, code in distances[:count]]


def update_city_rows(city_rows_path, airport_candidates):
    """Update city rows CSV with AirportsNearby column."""
    if not city_rows_path or not Path(city_rows_path).exists():
        print(f"Warning: City rows CSV not found at {city_rows_path}")
        return {'updated': 0, 'missing_coords': 0}

    print(f"Updating city rows: {city_rows_path}")
    updated_rows = 0
    missing_coords = 0

    with open(city_rows_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'AirportsNearby' not in fieldnames:
            fieldnames.append('AirportsNearby')

        rows = []
        for row in reader:
            lat = parse_float(row.get('Latitude'))
            lon = parse_float(row.get('Longitude'))
            if lat is None or lon is None:
                row['AirportsNearby'] = ''
                missing_coords += 1
            else:
                nearby = nearest_airports(lat, lon, airport_candidates, 3)
                row['AirportsNearby'] = ','.join(nearby)
            rows.append(row)
            updated_rows += 1

    with open(city_rows_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated_rows} city rows")
    return {'updated': updated_rows, 'missing_coords': missing_coords}


def fetch_wikipedia_airports(state_name, url):
    """Fetch airport data from Wikipedia table."""

    print(f"Fetching data from {url}...")
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the main airport table (usually the first wikitable)
    table = soup.find('table', {'class': 'wikitable'})
    if not table:
        raise ValueError("Could not find airport table on Wikipedia page")

    airports = []
    current_type = "Unknown"

    rows = table.find_all('tr')
    for row in rows:
        # Skip header rows
        cells = row.find_all(['td', 'th'])
        if not cells or len(cells) < 3:
            continue

        # Check if this is the actual header row
        if cells[0].name == 'th':
            continue

        # Extract data from cells
        try:
            cell_data = [cell.get_text(strip=True) for cell in cells]

            # Skip if not enough data
            if len(cell_data) < 4:
                continue

            # Check if this is a separator row (City, FAA, IATA, ICAO are empty, but Airport name has text)
            city = cell_data[0] if len(cell_data) > 0 else ''
            faa = cell_data[1] if len(cell_data) > 1 else ''
            iata = cell_data[2] if len(cell_data) > 2 else ''
            icao = cell_data[3] if len(cell_data) > 3 else ''
            airport_name = cell_data[4] if len(cell_data) > 4 else ''

            # If first 4 columns are empty but airport name has content, it's a separator row
            if not city and not faa and not iata and not icao and airport_name:
                current_type = airport_name
                continue

            # Typical structure: City, FAA, IATA, ICAO, Airport name, Role, Enplanements
            airport_data = {
                'City': city,
                'FAA': faa,
                'IATA': iata,
                'ICAO': icao,
                'Airport': airport_name,
                'Role': cell_data[5] if len(cell_data) > 5 else '',
                'Enplanements': cell_data[6] if len(cell_data) > 6 else '',
                'Type': current_type
            }

            # Clean up enplanements (remove commas, N/A, etc.)
            enpl = airport_data['Enplanements']
            enpl = re.sub(r'[,\s]', '', enpl)
            if enpl and enpl.lower() not in ['n/a', 'none', '-', '']:
                try:
                    airport_data['Enplanements'] = int(enpl)
                except:
                    airport_data['Enplanements'] = ''
            else:
                airport_data['Enplanements'] = ''

            # Extract airport name from full string (remove links, refs, etc.)
            airport_name = airport_data['Airport']
            # Remove citation brackets
            airport_name = re.sub(r'\[.*?\]', '', airport_name)

            # Extract "(was ...)" text to AlsoCalled column
            also_called = ''
            was_match = re.search(r'\(was\s+([^)]+)\)', airport_name, re.IGNORECASE)
            if was_match:
                also_called = was_match.group(1).strip()
                # Remove the "(was ...)" part from airport name
                airport_name = re.sub(r'\(was\s+[^)]+\)', '', airport_name, flags=re.IGNORECASE)

            # Add space before any remaining opening parentheses
            airport_name = re.sub(r'(\S)\(', r'\1 (', airport_name)

            airport_data['Airport'] = airport_name.strip()
            airport_data['AlsoCalled'] = also_called

            # Only add if we have at least a FAA code
            if airport_data['FAA'] and airport_data['FAA'] != 'N/A':
                airports.append(airport_data)

        except Exception as e:
            print(f"Warning: Error parsing row: {e}")
            continue

    return airports


def parse_kml_file(kml_path):
    """Parse KML file and extract coordinates by FAA code."""
    kml_data = {}

    if not Path(kml_path).exists():
        print(f"Warning: KML file not found at {kml_path}")
        return kml_data

    print(f"Parsing KML file: {kml_path}")

    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Handle XML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    for placemark in root.findall('.//kml:Placemark', ns):
        loc_id_elem = placemark.find('.//kml:Data[@name="locId"]/kml:value', ns)
        coords_elem = placemark.find('.//kml:coordinates', ns)
        name_elem = placemark.find('.//kml:name', ns)

        if loc_id_elem is not None and coords_elem is not None:
            faa_code = loc_id_elem.text
            coords = coords_elem.text.strip()
            name = name_elem.text if name_elem is not None else ''

            # Coordinates are in format: longitude,latitude[,altitude]
            parts = coords.split(',')
            if len(parts) >= 2:
                kml_data[faa_code] = {
                    'Longitude': parts[0],
                    'Latitude': parts[1],
                    'KML_Name': name
                }

    print(f"Loaded {len(kml_data)} airports from KML")
    return kml_data


def load_runways_data(runways_csv_path):
    """Load runway data and organize by FAA code."""
    if not Path(runways_csv_path).exists():
        print(f"Warning: Runways CSV not found at {runways_csv_path}")
        return {}

    print(f"Loading runways data: {runways_csv_path}")

    df = pd.read_csv(runways_csv_path, encoding='utf-8-sig')

    # Group by ARPT_ID (FAA code)
    runways_by_airport = {}

    for arpt_id, group in df.groupby('ARPT_ID'):
        runways = []
        for idx, row in group.iterrows():
            runway = {
                'RWY_ID': row.get('RWY_ID', ''),
                'RWY_LEN': row.get('RWY_LEN', ''),
                'RWY_WIDTH': row.get('RWY_WIDTH', ''),
                'SURFACE_TYPE_CODE': row.get('SURFACE_TYPE_CODE', ''),
                'LAT1_DECIMAL': row.get('LAT1_DECIMAL', ''),
                'LONG1_DECIMAL': row.get('LONG1_DECIMAL', ''),
            }
            runways.append(runway)

        runways_by_airport[arpt_id] = {
            'runways': runways[:3],  # Keep up to 3 runways
            'LAT1_DECIMAL': group.iloc[0].get('LAT1_DECIMAL', ''),
            'LONG1_DECIMAL': group.iloc[0].get('LONG1_DECIMAL', ''),
            'ARPT_NAME': group.iloc[0].get('ARPT_NAME', ''),
            'STATE_CODE': group.iloc[0].get('STATE_CODE', ''),
        }

    print(f"Loaded runway data for {len(runways_by_airport)} airports")
    return runways_by_airport


def merge_data(wiki_data, kml_data, runways_data):
    """Merge all data sources."""
    merged = []
    unmatched_airports = []
    used_kml_codes = set()

    for airport in wiki_data:
        faa = airport['FAA']

        # Start with wiki data
        merged_airport = airport.copy()

        # Expand role code to full name
        if 'Role' in merged_airport:
            merged_airport['Role'] = expand_role_name(merged_airport['Role'])

        # Merge KML data (coordinates)
        has_coords = False
        if faa in kml_data:
            kml_info = kml_data[faa]
            merged_airport['Latitude'] = kml_info.get('Latitude', '')
            merged_airport['Longitude'] = kml_info.get('Longitude', '')
            used_kml_codes.add(faa)
            has_coords = True
        else:
            # Try from runways data
            if faa in runways_data:
                merged_airport['Latitude'] = runways_data[faa].get('LAT1_DECIMAL', '')
                merged_airport['Longitude'] = runways_data[faa].get('LONG1_DECIMAL', '')
                has_coords = True
            else:
                merged_airport['Latitude'] = ''
                merged_airport['Longitude'] = ''

        # Track airports without coordinates
        if not has_coords:
            unmatched_airports.append({
                'FAA': faa,
                'Airport': airport.get('Airport', ''),
                'City': airport.get('City', '')
            })

        # Merge runway data
        if faa in runways_data:
            rwy_info = runways_data[faa]

            # Add up to 3 runways
            for i, runway in enumerate(rwy_info.get('runways', [])[:3], 1):
                merged_airport[f'Runway{i}_ID'] = runway.get('RWY_ID', '')

                # Add length with units
                length = runway.get('RWY_LEN', '')
                merged_airport[f'Runway{i}_Length'] = f"{length} ft" if length else ''

                # Add width with units
                width = runway.get('RWY_WIDTH', '')
                merged_airport[f'Runway{i}_Width'] = f"{width} ft" if width else ''

                merged_airport[f'Runway{i}_Surface'] = runway.get('SURFACE_TYPE_CODE', '')

        # Ensure all runway columns exist
        for i in range(1, 4):
            if f'Runway{i}_ID' not in merged_airport:
                merged_airport[f'Runway{i}_ID'] = ''
                merged_airport[f'Runway{i}_Length'] = ''
                merged_airport[f'Runway{i}_Width'] = ''
                merged_airport[f'Runway{i}_Surface'] = ''

        merged.append(merged_airport)

    # Find unused KML records
    unused_kml = []
    for faa_code, kml_info in kml_data.items():
        if faa_code not in used_kml_codes:
            unused_kml.append({
                'FAA': faa_code,
                'Name': kml_info.get('KML_Name', ''),
                'Latitude': kml_info.get('Latitude', ''),
                'Longitude': kml_info.get('Longitude', '')
            })

    return merged, unmatched_airports, unused_kml


def save_csv(data, output_path):
    """Save merged data to CSV."""
    if not data:
        print("No data to save!")
        return

    # Define column order
    columns = [
        'City', 'FAA', 'Airport', 'AlsoCalled', 'Role', 'Enplanements', 'Type',
        'Latitude', 'Longitude',
        'Runway1_ID', 'Runway1_Length', 'Runway1_Width', 'Runway1_Surface',
        'Runway2_ID', 'Runway2_Length', 'Runway2_Width', 'Runway2_Surface',
        'Runway3_ID', 'Runway3_Length', 'Runway3_Width', 'Runway3_Surface',
    ]

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} airports to {output_path}")


def generate_report(state_name, state_abbrev, wiki_url, wiki_count, kml_count, runways_count, output_count, output_path, report_path, kml_path, runways_path, city_rows_path, city_update_stats, runtimes, merged_data, unmatched_airports, unused_kml):
    """Generate pull report."""

    # Calculate airport type breakdown
    from collections import Counter
    type_counts = Counter(airport['Type'] for airport in merged_data)
    type_breakdown = '\n'.join(f"  - {type_name}: {count}" for type_name, count in sorted(type_counts.items()))

    # Format unmatched airports list
    if unmatched_airports:
        unmatched_list = '\n'.join(f"  - {apt['FAA']}: {apt['Airport']} ({apt['City']})" for apt in unmatched_airports)
        unmatched_section = f"""
## Airports Without Coordinates ({len(unmatched_airports)})

The following airports from Wikipedia could not be matched to coordinates in either the KML or Runways CSV:

{unmatched_list}
"""
    else:
        unmatched_section = """
## Airports Without Coordinates (0)

All airports successfully matched to coordinate data.
"""

    # Format unused KML records list
    if unused_kml:
        unused_list = '\n'.join(f"  - {rec['FAA']}: {rec['Name']} (Lat: {rec['Latitude']}, Lon: {rec['Longitude']})" for rec in unused_kml[:500])  # Limit to first 500
        if len(unused_kml) > 500:
            unused_list += f"\n  - ... and {len(unused_kml) - 500} more"
        unused_section = f"""
## Unused KML Records ({len(unused_kml)})

The following records from the KML file were not matched to any Wikipedia airport:

{unused_list}
"""
    else:
        unused_section = """
## Unused KML Records (0)

All KML records were matched to Wikipedia airports.
"""

    report = f"""# Airport Data Pull Report

## Run Information
- **Run Date**: {runtimes['start_time']}
- **State**: {state_name} ({state_abbrev})
- **Source**: {wiki_url}
- **Total Runtime**: {runtimes['total']:.2f} seconds

## Data Sources Row Counts
- **Wikipedia Table**: {wiki_count} airports
- **KML File ({kml_path.as_posix()})**: {kml_count} airports
- **Runways CSV ({runways_path.as_posix()})**: {runways_count} airports

## Processing Times
- **Fetch Wikipedia**: {runtimes['wiki']:.2f}s
- **Parse KML**: {runtimes['kml']:.2f}s
- **Load Runways CSV**: {runtimes['runways']:.2f}s
- **Merge Data**: {runtimes['merge']:.2f}s
- **Save CSV**: {runtimes['save']:.2f}s
- **Update City Rows**: {runtimes['cities']:.2f}s

## Output
- **Output File**: {output_path.as_posix()}
- **Total Rows**: {output_count} airports
## City Rows Update
- **City Rows File**: {city_rows_path if city_rows_path else 'Not configured'}
- **Rows Updated**: {city_update_stats.get('updated', 0)}
- **Rows Missing Coordinates**: {city_update_stats.get('missing_coords', 0)}

## Notes
- Enplanements data from Wikipedia is based on 2019 data
- Airport types extracted from separator rows in Wikipedia table:
{type_breakdown}
- Latitude and Longitude merged from both {kml_path.name} and {runways_path.name} files, with FAA code used as join key
- Up to 3 runways per airport included with ID, Length, Width, and Surface type
- Separator rows omitted from final CSV output
{unmatched_section}{unused_section}
## Command
```bash
python pull-airports.py --config config.yaml
```

## State Abbreviation Mapping
State names are mapped to their 2-character abbreviations:
- Georgia → GA
- (Add other states as needed)
"""

    print(f"Generating report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch airport data and update city rows.")
    script_dir = Path(__file__).parent
    parser.add_argument(
        '--config',
        default=str(script_dir / 'config.yaml'),
        help='Path to config.yaml'
    )
    args = parser.parse_args()

    repo_root = find_repo_root(script_dir)
    config = load_config(args.config)

    country = str(config.get('COUNTRY', 'us')).strip().lower()
    state_list = parse_state_list(config.get('STATE'))
    if not state_list:
        print("Error: STATE is missing or empty in config")
        sys.exit(1)

    runways_value = resolve_state_value(config.get('RUNWAY_DATA'), '', '')
    runways_template = str(runways_value) if runways_value else 'runways.csv'
    runways_path = resolve_path(runways_template, repo_root, script_dir, prefer_script_dir=True)

    t0 = time.time()
    runways_data = load_runways_data(runways_path)
    runways_load_time = time.time() - t0

    if len(state_list) > 1 and isinstance(config.get('CITY_ROWS'), str):
        print("Warning: Multiple states configured with a single CITY_ROWS file.")

    for state_abbrev in state_list:
        state_name = ABBREV_STATE.get(state_abbrev)
        if not state_name:
            print(f"Error: Unknown state abbreviation '{state_abbrev}'")
            continue

        context = template_context(country, state_abbrev, state_name)
        wiki_url = build_wikipedia_url(config.get('WIKIPEDIA_AIRPORT_STATE'), context, state_abbrev, state_name)

        kml_template = resolve_state_value(config.get('KML_LATITUDE_LONGITUDE'), state_abbrev, state_name) or "{country}/{state}/{state}.kml"
        output_template = resolve_state_value(config.get('OUTPUT'), state_abbrev, state_name) or "{country}/{state}/{state}.csv"
        report_template = resolve_state_value(config.get('REPORT_OUTPUT'), state_abbrev, state_name) or "pull-report.md"
        city_rows_template = resolve_state_value(config.get('CITY_ROWS'), state_abbrev, state_name)

        kml_path = resolve_path(kml_template.format(**context), repo_root, script_dir, prefer_script_dir=True)
        output_path = resolve_path(output_template.format(**context), repo_root, script_dir, prefer_script_dir=True)
        report_path = resolve_path(report_template.format(**context), repo_root, script_dir, prefer_script_dir=True)
        city_rows_path = resolve_path(city_rows_template, repo_root, script_dir) if city_rows_template else None

        # Track runtimes
        start_time = time.time()
        start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        runtimes = {'start_time': start_datetime, 'runways': runways_load_time}

        try:
            # Fetch Wikipedia data
            t0 = time.time()
            wiki_data = fetch_wikipedia_airports(state_name, wiki_url)
            runtimes['wiki'] = time.time() - t0
            print(f"Fetched {len(wiki_data)} airports from Wikipedia ({runtimes['wiki']:.2f}s)")

            # Parse KML file
            t0 = time.time()
            kml_data = parse_kml_file(kml_path)
            runtimes['kml'] = time.time() - t0

            # Merge all data
            t0 = time.time()
            merged_data, unmatched_airports, unused_kml = merge_data(wiki_data, kml_data, runways_data)
            runtimes['merge'] = time.time() - t0

            # Save to CSV
            t0 = time.time()
            save_csv(merged_data, output_path)
            runtimes['save'] = time.time() - t0

            # Update city rows
            t0 = time.time()
            airport_candidates = build_airport_candidates(merged_data)
            city_update_stats = update_city_rows(city_rows_path, airport_candidates)
            runtimes['cities'] = time.time() - t0

            # Calculate total runtime
            runtimes['total'] = time.time() - start_time

            # Generate report
            generate_report(
                state_name=state_name,
                state_abbrev=state_abbrev,
                wiki_url=wiki_url,
                wiki_count=len(wiki_data),
                kml_count=len(kml_data),
                runways_count=len(runways_data),
                output_count=len(merged_data),
                output_path=output_path,
                report_path=report_path,
                kml_path=kml_path,
                runways_path=runways_path,
                city_rows_path=city_rows_path.as_posix() if city_rows_path else None,
                city_update_stats=city_update_stats,
                runtimes=runtimes,
                merged_data=merged_data,
                unmatched_airports=unmatched_airports,
                unused_kml=unused_kml
            )

            print(f"\n✓ Complete for {state_name}! (Total runtime: {runtimes['total']:.2f}s)")

        except Exception as e:
            print(f"Error processing {state_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
