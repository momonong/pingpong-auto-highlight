"""Exact interval metadata regressions using synthetic identities and durations."""

import copy
import math
import unittest

import pytest

from pingpong_highlight import research_window_coverage as SUBJECT


class MetadataCoverageTests(unittest.TestCase):
    def test_exact_cap_equality_contains_the_half_open_core(self):
        result = SUBJECT.classify_intervals([1000, 8001000], [1000, 8001000], 9000000)
        self.assertIs(result["full_core_contained"], True)
        self.assertIs(result["full_core_feasible_under_8s_and_source_bounds"], True)
        self.assertEqual(result["coverage_fraction"], "1/1")
        self.assertEqual(result["uncovered_us"], 0)
        self.assertEqual(result["endpoint_event_visibility"], "not_evaluated_metadata_only")

    def test_one_us_over_cap_is_impossible_without_float_rounding(self):
        result = SUBJECT.classify_intervals([0, 8000001], [0, 8000000], 9000000)
        self.assertIs(result["full_core_contained"], False)
        self.assertIs(result["full_core_feasible_under_8s_and_source_bounds"], False)
        self.assertEqual(result["missing_end_us"], 1)
        self.assertEqual(result["coverage_fraction"], "8000000/8000001")

    def test_odd_ms_midpoint_keeps_the_actual_legacy_window(self):
        result = SUBJECT.coverage_for_window({"target_ms": [1000, 9001]}, {"duration_us": 20000000})
        self.assertEqual(result["actual_window_us"], [1000000, 9000000])
        self.assertEqual(result["missing_end_us"], 1000)
        self.assertEqual(result["minimum_required_arbitrary_span_us"], 8001000)

    def test_source_clipping_does_not_imply_core_clipping(self):
        for core, duration, actual in (
            ([0, 1000], 10000000, [0, 4500000]),
            ([9000, 10000], 10000000, [5500000, 10000000]),
        ):
            with self.subTest(core=core):
                result = SUBJECT.coverage_for_window({"target_ms": core}, {"duration_us": duration})
                self.assertEqual(result["actual_window_us"], actual)
                self.assertIs(result["full_core_contained"], True)
                self.assertTrue(result["source_clipping_reasons"])

    def test_recorded_canonical_overhang_is_not_a_tolerance(self):
        result = SUBJECT.classify_intervals(
            [22000000, 23394000], [18697000, 23393800], 23393800, canonical_quantization=True
        )
        self.assertEqual(result["core_source_end_overhang_us"], 200)
        self.assertEqual(result["missing_end_us"], 200)
        self.assertIs(result["full_core_contained"], False)
        self.assertIs(result["full_core_feasible_under_8s_and_source_bounds"], False)
        self.assertIn(
            "canonical_core_end_exceeds_true_source_duration", result["feasibility_reasons"]
        )
        with self.assertRaises(SUBJECT.MetadataError):
            SUBJECT.classify_intervals([22000000, 23394000], [18697000, 23393800], 23393800)
        with self.assertRaises(SUBJECT.MetadataError):
            SUBJECT.classify_intervals(
                [222000000, 23394001], [18697000, 23393800], 23393800, canonical_quantization=True
            )

    def test_disjoint_intervals_never_report_more_missing_than_core(self):
        result = SUBJECT.classify_intervals([9000000, 10000000], [0, 1000000], 11000000)
        self.assertEqual(result["intersection_us"], None)
        self.assertEqual(result["covered_us"], 0)
        self.assertEqual(result["uncovered_us"], 1000000)
        self.assertEqual(result["missing_end_us"], 1000000)

    def test_rejects_invalid_types_empty_ranges_and_source_contradictions(self):
        for value in (True, False, 1.0, 1.5, math.nan, math.inf, "1", None):
            with self.subTest(value=repr(value)):
                with self.assertRaises(SUBJECT.MetadataError):
                    SUBJECT.classify_intervals([value, 2000], [0, 3000], 4000)
                with self.assertRaises(SUBJECT.MetadataError):
                    SUBJECT.classify_intervals([0, 2000], [0, 3000], value)
                with self.assertRaises(SUBJECT.MetadataError):
                    SUBJECT.coverage_for_window({"target_ms": [0, value]}, {"duration_us": 4000})
        for core, observation, duration in (
            ([1, 1], [0, 3], 4),
            ([2, 1], [0, 3], 4),
            ([-1, 1], [0, 3], 4),
            ([0, 5], [0, 3], 4),
            ([0, 2], [1, 1], 4),
            ([0, 2], [-1, 3], 4),
            ([0, 2], [0, 5], 4),
            ([0, 2], [0, 8000001], 9000000),
        ):
            with (
                self.subTest(core=core, observation=observation),
                self.assertRaises(SUBJECT.MetadataError),
            ):
                SUBJECT.classify_intervals(core, observation, duration)

    def test_refuses_row_source_identity_and_recipe_mismatches(self):
        source = {"upload_id": "source-a", "source_sha256": "a" * 64, "duration_us": 10000000}
        row = {"target_ms": [1000, 2000], "source_id": "source-b", "source_sha256": "a" * 64}
        with self.assertRaises(SUBJECT.MetadataError):
            SUBJECT.coverage_for_window(row, source)
        row["source_id"] = "source-a"
        row["source_sha256"] = "b" * 64
        with self.assertRaises(SUBJECT.MetadataError):
            SUBJECT.coverage_for_window(row, source)
        row["source_sha256"] = "a" * 64
        row["actual_window_us"] = [1, 5500000]
        with self.assertRaises(SUBJECT.MetadataError):
            SUBJECT.coverage_for_window(row, source)

    def test_gate_refuses_oversize_observation_and_canonical_overhang(self):
        source = {"upload_id": "synthetic", "duration_us": 23393800}
        window = {
            "window_id": "synthetic-window",
            "source_id": "synthetic",
            "target_ms": [1000, 2000],
            "actual_window_us": [0, 9000000],
        }
        with self.assertRaises(SUBJECT.MetadataError):
            SUBJECT.require_full_core([window], {"synthetic": source})
        window = {
            "window_id": "canonical-end",
            "source_id": "synthetic",
            "target_ms": [22000, 23394],
        }
        with self.assertRaises(SUBJECT.StrictCoverageError) as caught:
            SUBJECT.require_full_core([window], {"synthetic": source}, canonical_quantization=True)
        self.assertEqual(caught.exception.failures[0]["missing_end_us"], 200)

    def test_provided_legacy_time_fields_are_validated_instead_of_ignored(self):
        for field, value in (
            ("center_ms", True),
            ("center_ms", 999),
            ("requested_window_ms", [0, 9000000]),
            ("target_seconds_original", ["NaN", "2"]),
            ("target_seconds_original", ["1.001", "2"]),
        ):
            with self.subTest(field=field, value=value):
                row = {"target_ms": [1000, 2000], field: value}
                with self.assertRaises(SUBJECT.MetadataError):
                    SUBJECT.coverage_for_window(row, {"duration_us": 10000000})

    def test_strict_gate_validates_provided_seconds_and_timecodes(self):
        source = {"upload_id": "synthetic", "duration_us": 10000000}
        window = {
            "window_id": "synthetic-window",
            "source_id": "synthetic",
            "target_ms": [1000, 2000],
        }
        for field, value in (
            ("actual_window_seconds", ["NaN", "Infinity"]),
            ("actual_window_timecode", ["nonsense", "nonsense"]),
            ("core_timecode", ["nonsense", "nonsense"]),
        ):
            with self.subTest(field=field), self.assertRaises(SUBJECT.MetadataError):
                SUBJECT.require_full_core(
                    windows=[{**window, field: value}], sources={"synthetic": source}
                )
        matching = {
            **window,
            "actual_window_seconds": ["0.000000", "5.500000"],
            "actual_window_timecode": ["00:00:00.000000", "00:00:05.500000"],
            "core_timecode": ["00:00:01.000000", "00:00:02.000000"],
        }
        rows = SUBJECT.require_full_core(windows=[matching], sources={"synthetic": source})
        self.assertIs(rows[0]["full_core_contained"], True)


def test_batch_rejection_retains_every_requested_row_and_order():
    source = {"upload_id": "synthetic-source", "duration_us": 30000000}
    windows = [
        {
            "window_id": "synthetic-short",
            "source_id": "synthetic-source",
            "target_ms": [1000, 2000],
        },
        {
            "window_id": "synthetic-long",
            "source_id": "synthetic-source",
            "target_ms": [10000, 19000],
        },
        {
            "window_id": "synthetic-last",
            "source_id": "synthetic-source",
            "target_ms": [20000, 21000],
        },
    ]
    before = copy.deepcopy(windows)
    with pytest.raises(SUBJECT.StrictCoverageError) as caught:
        SUBJECT.require_full_core(windows, {"synthetic-source": source})
    assert [row["window_id"] for row in caught.value.coverage_rows] == [
        "synthetic-short",
        "synthetic-long",
        "synthetic-last",
    ]
    assert [row["window_id"] for row in caught.value.failures] == ["synthetic-long"]
    assert caught.value.failures[0]["coverage_fraction"] == "8/9"
    assert windows == before
    assert (
        len(SUBJECT.require_full_core([windows[0], windows[2]], {"synthetic-source": source})) == 2
    )


@pytest.mark.parametrize(
    "windows,sources",
    [
        ([], {}),
        (
            [{"window_id": "duplicate", "source_id": "s", "target_ms": [0, 1000]}] * 2,
            {"s": {"upload_id": "s", "duration_us": 2000000}},
        ),
        ([{"window_id": "unknown", "source_id": "absent", "target_ms": [0, 1000]}], {}),
    ],
)
def test_batch_rejects_empty_duplicate_and_unknown_sources(windows, sources):
    with pytest.raises(SUBJECT.MetadataError):
        SUBJECT.require_full_core(windows, sources)


@pytest.mark.parametrize("source_id", [[], {}])
def test_batch_rejects_unhashable_source_ids_as_metadata_errors(source_id):
    windows = [
        {"window_id": "synthetic-invalid-source", "source_id": source_id, "target_ms": [0, 1000]}
    ]
    sources = {"synthetic": {"upload_id": "synthetic", "duration_us": 2000000}}
    with pytest.raises(SUBJECT.MetadataError):
        SUBJECT.require_full_core(windows, sources)
