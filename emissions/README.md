# Aviation emissions processing

Config-driven batch processing adapted from [VisharadR/aviation-emissions](https://github.com/VisharadR/aviation-emissions).
This adds flight and airport CO2 estimates alongside the existing airport directory.
It runs on a laptop: no Render service, cloud backend, or browser credentials are needed.

## Run

From the airports repository root (Python 3.10+):

```sh
python -m pip install -r emissions/requirements.txt
python emissions/process.py --config emissions/config.yaml --refresh-airports
```

Before running, set `FLIGHTS` in `config.yaml` to your existing OpenSky flight CSVs,
or put them in `emissions/data/flights_YYYYMMDD.csv`. The processor accepts the original
aviation project's `icao24,callsign,firstSeen,lastSeen,dep,arr` columns and raw OpenSky
`estDepartureAirport,estArrivalAirport` aliases. Timestamps are Unix seconds. Callsign
is optional. Missing endpoints are counted as coverage loss, not assigned zero CO2.

`--refresh-airports` downloads the OurAirports catalog; omit it to reuse a local
catalog. You can point `AIRPORT_CATALOG` to the original project's
`backend/data/ourairports_airports.csv`. All paths and globs are relative to the
selected config file, regardless of the working directory. Absolute input paths
are supported. Raw flight exports are not included in this repository.

This processor **imports previously obtained flight observations**; it does not
schedule or authenticate new OpenSky flight pulls. Existing ingestion tools remain
in the source aviation project. Follow OpenSky's applicable data-access terms when
obtaining or redistributing observations. The download switch only refreshes airport
metadata. Credentials do not belong in this config or in Git.

## Configuration

The uppercase keys follow the style of `pipeline/config.yaml`:

| Key | Purpose |
| --- | --- |
| `COUNTRY`, `STATE` | Directory selection; STATE accepts `GA,NY` or a YAML list. |
| `FLIGHTS` | CSV input path/glob; imports all matching files and deduplicates flight identity. |
| `AIRPORT_CATALOG` | OurAirports CSV with ICAO identifiers, FAA/local codes and coordinates. |
| `AIRPORT_CATALOG_URL` | HTTPS metadata download for explicit `--refresh-airports`. |
| `AIRPORT_DIRECTORY` | Existing directory CSV path with `{country}`/`{state}` placeholders. |
| `START_DATE`, `END_DATE` | Optional inclusive UTC departure dates, `YYYY-MM-DD`. |
| `FUEL_KG_PER_KM` | Generic distance-dependent fuel model, default 3. |
| `FIXED_FUEL_KG` | Per-flight fixed fuel allowance, default 500. |
| `CO2_KG_PER_KG_FUEL` | CO2 conversion factor, default 3.16. |
| `OUTPUT` | Separate output directory, default `output`. |

The fork currently includes only `us/ga/ga.csv`, so Georgia is the working directory
default. Process additional states with the existing airport pipeline first, then
list them in `STATE`. Flights may have endpoints anywhere worldwide: state selection
limits only the map-directory export, not route calculations or the global report.

## Outputs and map integration

- `output/flights.csv`: one estimate per accepted flight, with UTC departure date,
  identifiers, distance, fuel and CO2 in kg.
- `output/routes.csv`: daily departure/arrival route totals in kg.
- `output/airports.csv`: daily departure-attributed totals for selected directory
  airports. Includes `FAA`, `ICAO`, `Airport`, `Latitude`, `Longitude`, `Departures`,
  `EstimatedCO2_kg`, `Country`, `State`, and `date`.
- `output/report.json`: input SHA-256 hashes, generation time, model settings,
  exclusion counts, directory matching counts and global computed CO2.

Join `airports.csv` to the directory using `(Country, State, FAA)` and select a
single `date` before mapping. Use `Latitude`/`Longitude` for markers and
`EstimatedCO2_kg` for marker size. Coordinates come from the hashed OurAirports
catalog. Existing directory and runway CSVs are never overwritten. The map site's
configuration and data-pipeline admin node registration live in separate repositories
and are not changed by this addition.

ICAO-to-FAA matching uses OurAirports' explicit `local_code`, country and region.
It does not invent ICAO codes by adding a `K` prefix. Ambiguous identifiers are
excluded. Airports with no computed departures have no output row; absence does
not mean zero actual emissions. Global totals can exceed selected-state totals.

## Method and limitations

```text
distance_km = great-circle distance between airport coordinates
fuel_kg = FIXED_FUEL_KG + FUEL_KG_PER_KM * distance_km
co2_kg = fuel_kg * CO2_KG_PER_KG_FUEL
```

This preserves the historical aviation project's generic model. It is not an
aircraft/engine-specific inventory, a measured fuel burn, or a live emission rate.
It excludes non-CO2 effects and does not reconstruct actual flown tracks. OpenSky
airport assignments are inferred and receiver coverage is incomplete. Same-airport
flights, invalid coordinates, invalid time intervals, and unmatched endpoints are
excluded and reported. Overlapping files use the first occurrence of
`(icao24, firstSeen, lastSeen)` in sorted file order.

Each flight is attributed once to its departure airport; arrivals do not receive
a second copy. Only the departure date is used for grouping, including overnight flights.

## Validation

```sh
python -m unittest discover -s emissions -p "test_*.py" -v
```

Tests use explicitly synthetic records in temporary directories, including known
distances, duplicates, unmatched airports, invalid coordinates, UTC date filtering,
raw OpenSky column aliases and FAA joins. Test records are not production data.

Code derived from the aviation project is distributed under GPL-3.0; see LICENSE.
This code license does not relicense the source datasets.
