"""Synthetic fixtures test calculations and coverage; never published as observations."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import yaml
import process
from process import run, distance, unique_index


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = dict(COUNTRY="us", STATE="GA", FLIGHTS="flights.csv", AIRPORT_CATALOG="catalog.csv",
                        AIRPORT_DIRECTORY="directory.csv", OUTPUT="output", FUEL_KG_PER_KM=3,
                        FIXED_FUEL_KG=500, CO2_KG_PER_KG_FUEL=3.16, START_DATE=None, END_DATE=None)
        self.catalog = [dict(ident="KAAA", icao_code="KAAA", local_code="AAA", name="Fixture A", iso_country="US", iso_region="US-GA", latitude_deg=0, longitude_deg=0),
                        dict(ident="KBBB", icao_code="KBBB", local_code="BBB", name="Fixture B", iso_country="US", iso_region="US-NY", latitude_deg=0, longitude_deg=1)]
        self.flight = dict(icao24="abc123", callsign="TEST", firstSeen=1766620800, lastSeen=1766624400, dep="KAAA", arr="KBBB")
        self.write("catalog.csv", self.catalog)
        self.write("directory.csv", [dict(FAA="AAA", Airport="Directory A"), dict(FAA="ZZZ", Airport="Unmatched")])

    def write(self, name, records):
        with (self.root/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def process(self, records):
        self.write("flights.csv", records)
        config = self.root/"config.yaml"
        config.write_text(yaml.safe_dump(self.cfg), encoding="utf-8")
        return run(config)

    def test_model_mapping_and_deduplication(self):
        report = self.process([self.flight, self.flight, dict(self.flight, icao24="bad", arr="UNKNOWN")])
        self.assertEqual(report["counts"], dict(input_rows=3, duplicate_rows=1, unmatched_endpoints=1, computed_flights=1))
        self.assertAlmostEqual(report["total_co2_kg"], (500+3*111.19492664455873)*3.16, places=6)
        with (self.root/"output/airports.csv").open() as stream:
            airports = list(csv.DictReader(stream))
        self.assertEqual(len(airports), 1)
        self.assertEqual(airports[0]["ICAO"], "KAAA")
        self.assertEqual(airports[0]["Airport"], "Directory A")
        self.assertEqual(report["directory_airports_unmatched"], 1)

    def test_date_filter_is_utc_and_inclusive(self):
        self.cfg.update(START_DATE="2025-12-25", END_DATE="2025-12-25")
        report = self.process([self.flight, dict(self.flight, firstSeen=1766707200, lastSeen=1766710800)])
        self.assertEqual(report["counts"]["outside_date_range"], 1)
        self.assertEqual(report["counts"]["computed_flights"], 1)

    def test_invalid_coordinates_and_interval_are_reported(self):
        self.catalog[0]["longitude_deg"] = "nan"
        self.write("catalog.csv", self.catalog)
        report = self.process([self.flight, dict(self.flight, lastSeen=0)])
        self.assertEqual(report["counts"]["invalid_coordinates"], 1)
        self.assertEqual(report["counts"]["invalid_identity_or_time"], 1)
        self.assertEqual(report["total_co2_kg"], 0)

    def test_raw_opensky_columns(self):
        flight = dict(self.flight)
        flight["estDepartureAirport"] = flight.pop("dep")
        flight["estArrivalAirport"] = flight.pop("arr")
        self.assertEqual(self.process([flight])["counts"]["computed_flights"], 1)

    def test_ambiguous_codes_are_not_guessed(self):
        self.assertEqual(unique_index([{"id": "X"}, {"id": "X"}], lambda r: r["id"]), {})

    def test_missing_endpoint_schema_fails(self):
        record = dict(self.flight)
        del record["dep"]
        with self.assertRaisesRegex(ValueError, "departure/arrival columns"):
            self.process([record])

    def test_date_order_and_model_validation(self):
        self.cfg.update(START_DATE="2026-01-01", END_DATE="2025-01-01")
        with self.assertRaisesRegex(ValueError, "START_DATE"):
            self.process([self.flight])
        self.cfg.update(START_DATE=None, END_DATE=None, FUEL_KG_PER_KM=-1)
        with self.assertRaisesRegex(ValueError, "Model factors"):
            self.process([self.flight])

    def test_same_airport_is_not_a_zero_distance_trip_estimate(self):
        self.assertEqual(self.process([dict(self.flight, arr="KAAA")])["counts"]["same_airport_flights"], 1)

    def test_antipodal_distance(self):
        self.assertAlmostEqual(distance(dict(latitude_deg=0, longitude_deg=0), dict(latitude_deg=0, longitude_deg=180)), 20015.086796, places=5)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = dict(COUNTRY="us", STATE="GA", FLIGHTS="flights.csv", AIRPORT_CATALOG="catalog.csv",
                        AIRPORT_DIRECTORY="directory.csv", OUTPUT="output", PUBLISH_DIR="pub",
                        FUEL_KG_PER_KM=3, FIXED_FUEL_KG=500, CO2_KG_PER_KG_FUEL=3.16,
                        START_DATE=None, END_DATE=None)
        catalog = [dict(ident="KAAA", icao_code="KAAA", local_code="AAA", name="Fixture A", iso_country="US", iso_region="US-GA", latitude_deg=0, longitude_deg=0),
                   dict(ident="KBBB", icao_code="KBBB", local_code="BBB", name="Fixture B", iso_country="US", iso_region="US-NY", latitude_deg=0, longitude_deg=1)]
        self._write("catalog.csv", catalog)
        self._write("directory.csv", [dict(FAA="AAA", Airport="Directory A")])
        self.day1 = 1766620800              # 2025-12-25 UTC
        self.day2 = self.day1 + 86400
        self.day3 = self.day1 + 2 * 86400

    def _write(self, name, records):
        with (self.root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def _flight(self, first, icao):
        return dict(icao24=icao, callsign="T", firstSeen=first, lastSeen=first + 3600, dep="KAAA", arr="KBBB")

    def _run(self, flights):
        self._write("flights.csv", flights)
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(self.cfg), encoding="utf-8")
        return run(config, publish=True)

    @staticmethod
    def _date(ts):
        return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()

    def _manifest(self):
        return json.loads((self.root / "pub/emissions/manifest.json").read_text(encoding="utf-8"))

    def test_partitions_by_utc_day(self):
        self._run([self._flight(self.day1, "a1"), self._flight(self.day2, "a2")])
        daily = self.root / "pub/emissions/daily"
        for ts in (self.day1, self.day2):
            day_dir = daily / self._date(ts)
            self.assertTrue(day_dir.is_dir())
            for name in ("flights.csv", "routes.csv", "airports.csv", "report.json"):
                self.assertTrue((day_dir / name).exists(), f"{self._date(ts)}/{name}")
        manifest = self._manifest()
        self.assertEqual(sorted(manifest["dates"]), [self._date(self.day1), self._date(self.day2)])
        self.assertEqual(manifest["latest"], self._date(self.day2))
        entry = manifest["dates"][self._date(self.day1)]
        self.assertEqual(entry["flights"], 1)
        self.assertEqual(len(entry["files"]["flights.csv"]), 64)   # sha256 hex digest
        self.assertEqual(manifest["model"], {"FUEL_KG_PER_KM": 3, "FIXED_FUEL_KG": 500, "CO2_KG_PER_KG_FUEL": 3.16})

    def test_manifest_merges_across_runs(self):
        self._run([self._flight(self.day1, "a1")])
        self._run([self._flight(self.day3, "a3")])     # separate run, different date
        manifest = self._manifest()
        self.assertEqual(sorted(manifest["dates"]), [self._date(self.day1), self._date(self.day3)])
        self.assertEqual(manifest["latest"], self._date(self.day3))
        self.assertTrue((self.root / "pub/emissions/daily" / self._date(self.day1)).is_dir())  # earlier date kept

    def test_rerun_is_idempotent(self):
        self._run([self._flight(self.day1, "a1")])
        self._run([self._flight(self.day1, "a1")])     # same date again
        manifest = self._manifest()
        self.assertEqual(list(manifest["dates"]), [self._date(self.day1)])
        self.assertEqual(manifest["dates"][self._date(self.day1)]["flights"], 1)
        with (self.root / "pub/emissions/daily" / self._date(self.day1) / "flights.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)   # replaced, not appended

    def test_size_guard_fails_over_limit(self):
        with mock.patch.object(process, "PUBLISH_FAIL_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "publish limit"):
                self._run([self._flight(self.day1, "a1")])

    def test_size_guard_warns_over_threshold(self):
        buf = io.StringIO()
        with mock.patch.object(process, "PUBLISH_WARN_BYTES", 10), \
             mock.patch.object(process, "PUBLISH_FAIL_BYTES", 10 ** 9):
            with contextlib.redirect_stdout(buf):
                self._run([self._flight(self.day1, "a1")])
        self.assertIn("Warning", buf.getvalue())

    def test_publish_requires_publish_dir(self):
        self.cfg.pop("PUBLISH_DIR")
        with self.assertRaisesRegex(ValueError, "PUBLISH_DIR"):
            self._run([self._flight(self.day1, "a1")])


if __name__ == "__main__":
    unittest.main()
