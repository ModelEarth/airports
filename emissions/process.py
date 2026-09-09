"""Config-driven batch emissions estimates; no web server or secrets required."""
import argparse
import csv
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import yaml


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def coordinates(row):
    lat, lon = float(row["latitude_deg"]), float(row["longitude_deg"])
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid coordinates")
    return lat, lon


def distance(a, b):
    lat1, lon1 = map(math.radians, coordinates(a))
    lat2, lon2 = map(math.radians, coordinates(b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742 * math.asin(math.sqrt(min(1, max(0, h))))


def unique_index(records, key):
    groups = defaultdict(list)
    for row in records:
        value = key(row)
        if value:
            groups[value].append(row)
    # Ambiguous identifiers must not silently pick an airport.
    return {key: values[0] for key, values in groups.items() if len(values) == 1}


def run(config_path, refresh=False):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    resolve = lambda value: (base / value).resolve()
    factors = [float(cfg[key]) for key in ("FUEL_KG_PER_KM", "FIXED_FUEL_KG", "CO2_KG_PER_KG_FUEL")]
    if not all(math.isfinite(v) and v >= 0 for v in factors) or factors[2] == 0:
        raise ValueError("Model factors must be finite and nonnegative; CO2 factor must be positive")
    start = date.fromisoformat(str(cfg["START_DATE"])) if cfg.get("START_DATE") else None
    end = date.fromisoformat(str(cfg["END_DATE"])) if cfg.get("END_DATE") else None
    if start and end and start > end:
        raise ValueError("START_DATE must not be after END_DATE")
    sources = sorted(Path(p) for p in glob.glob(str(resolve(cfg["FLIGHTS"]))))
    if not sources:
        raise ValueError("No flight CSVs found. Set FLIGHTS in config.yaml to existing flights_*.csv exports.")
    catalog_path = resolve(cfg["AIRPORT_CATALOG"])
    if refresh:
        url = cfg["AIRPORT_CATALOG_URL"]
        if not url.startswith("https://"):
            raise ValueError("Airport catalog download requires HTTPS")
        with urlopen(url, timeout=60) as response:
            content = response.read()
        header = next(csv.reader(content.decode("utf-8-sig").splitlines()))
        if not {"ident", "latitude_deg", "longitude_deg", "local_code"} <= set(header):
            raise ValueError("Downloaded airport catalog has an unexpected schema")
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_bytes(content)
    catalog = rows(catalog_path)
    # OpenSky uses ICAO identifiers. Use explicit ICAO, then legacy ident.
    airport_index = unique_index(catalog, lambda a: (a.get("icao_code") or a.get("ident") or "").strip().upper())
    counts, seen, computed = Counter(), set(), []
    for source in sources:
        records = rows(source)
        if records and not {"icao24", "firstSeen", "lastSeen"} <= records[0].keys():
            raise ValueError(f"{source.name}: missing flight identity/timestamp columns")
        if records and not (({"dep", "arr"} <= records[0].keys()) or
                            ({"estDepartureAirport", "estArrivalAirport"} <= records[0].keys())):
            raise ValueError(f"{source.name}: missing departure/arrival columns")
        for row in records:
            counts["input_rows"] += 1
            try:
                first, last = int(row["firstSeen"]), int(row["lastSeen"])
                day = datetime.fromtimestamp(first, timezone.utc).date()
                ident = row["icao24"].strip().lower()
                if not ident or last <= first:
                    raise ValueError("Invalid flight identity or interval")
            except (ValueError, KeyError, OverflowError, OSError):
                counts["invalid_identity_or_time"] += 1
                continue
            if (start and day < start) or (end and day > end):
                counts["outside_date_range"] += 1
                continue
            dep = (row.get("dep") or row.get("estDepartureAirport") or "").strip().upper()
            arr = (row.get("arr") or row.get("estArrivalAirport") or "").strip().upper()
            key = (ident, first, last)
            if key in seen:
                counts["duplicate_rows"] += 1
                continue
            seen.add(key)
            a, b = airport_index.get(dep), airport_index.get(arr)
            if not a or not b:
                counts["unmatched_endpoints"] += 1
                continue
            if dep == arr:
                counts["same_airport_flights"] += 1
                continue
            try:
                km = distance(a, b)
            except (ValueError, KeyError, TypeError):
                counts["invalid_coordinates"] += 1
                continue
            fuel = factors[1] + factors[0]*km
            computed.append(dict(date=day.isoformat(), icao24=ident, callsign=row.get("callsign", "").strip(), firstSeen=first,
                                 lastSeen=last, dep=dep, arr=arr, distance_km=km, fuel_kg=fuel, co2_kg=fuel*factors[2]))
    counts["computed_flights"] = len(computed)
    airport_totals, routes = {}, {}
    for flight in computed:
        key = (flight["date"], flight["dep"], flight["arr"])
        route = routes.setdefault(key, dict(date=key[0], dep=key[1], arr=key[2], flights=0, co2_kg=0.0))
        route["flights"] += 1
        route["co2_kg"] += flight["co2_kg"]
        key = (flight["date"], flight["dep"])
        total = airport_totals.setdefault(key, dict(departures=0, co2_kg=0.0))
        total["departures"] += 1
        total["co2_kg"] += flight["co2_kg"]
    country = cfg["COUNTRY"].upper()
    states = cfg["STATE"] if isinstance(cfg["STATE"], list) else cfg["STATE"].split(",")
    directory_outputs, directory_sources = [], []
    matched = unmatched = 0
    for state in states:
        state = state.strip().upper()
        directory_path = resolve(cfg["AIRPORT_DIRECTORY"].format(country=country.lower(), state=state.lower()))
        directory_sources.append(directory_path)
        local_index = unique_index([a for a in catalog if a.get("iso_country") == country and a.get("iso_region") == f"{country}-{state}"],
                                   lambda a: a.get("local_code", "").strip().upper())
        for directory_row in rows(directory_path):
            a = local_index.get(directory_row.get("FAA", "").strip().upper())
            icao = (a.get("icao_code") or a.get("ident")) if a else None
            if not a or airport_index.get(icao) is not a:
                unmatched += 1
                continue
            matched += 1
            for (day, dep), total in sorted(airport_totals.items()):
                if dep != icao:
                    continue
                directory_outputs.append(dict(date=day, Country=country, State=state, FAA=directory_row["FAA"], ICAO=icao,
                                              Airport=directory_row.get("Airport", a["name"]),
                                              Latitude=coordinates(a)[0], Longitude=coordinates(a)[1],
                                              Departures=total["departures"], EstimatedCO2_kg=total["co2_kg"]))
    output = resolve(cfg["OUTPUT"])
    if any(output == p.parent for p in sources + [catalog_path] + directory_sources):
        raise ValueError("OUTPUT must be separate from input directories")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "flights.csv", ["date", "icao24", "callsign", "firstSeen", "lastSeen", "dep", "arr", "distance_km", "fuel_kg", "co2_kg"], computed)
    write_csv(output / "routes.csv", ["date", "dep", "arr", "flights", "co2_kg"], [routes[k] for k in sorted(routes)])
    write_csv(output / "airports.csv", ["date", "Country", "State", "FAA", "ICAO", "Airport", "Latitude", "Longitude", "Departures", "EstimatedCO2_kg"], directory_outputs)
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
                  directory_airports_matched=matched, directory_airports_unmatched=unmatched,
                  total_co2_kg=sum(f["co2_kg"] for f in computed),
                  model={k: cfg[k] for k in ("FUEL_KG_PER_KM", "FIXED_FUEL_KG", "CO2_KG_PER_KG_FUEL")},
                  attribution="Each flight's estimated CO2 is attributed once, to its departure airport.",
                  sources=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [config_path, catalog_path]+sources+directory_sources])
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--refresh-airports", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.config, args.refresh_airports)
    except (ValueError, KeyError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f"Emissions processing failed: {exc}\n")
    print(json.dumps(report["counts"], indent=2))
