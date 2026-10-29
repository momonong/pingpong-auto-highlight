"""Experimental exact candidate-core metadata coverage; never reads media.

Half-open integer-us containment is distinct from source clipping, endpoint-event
visibility, rally completeness and preference. Decimal values are display-only.
"""

from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext
from math import gcd

MAX_SPAN_US = 8000000


class MetadataError(ValueError):
    """Malformed, contradictory or untrusted metadata; stop before media work."""


class StrictCoverageError(MetadataError):
    def __init__(self, coverage_rows):
        self.coverage_rows = coverage_rows
        self.failures = [row for row in coverage_rows if not row["full_core_contained"]]
        super().__init__(
            "full-core batch rejected: "
            + "; ".join(
                row["window_id"]
                + ": "
                + ", ".join(row["noncontainment_reasons"] + row["feasibility_reasons"])
                for row in self.failures
            )
        )


def require(condition, message):
    if not condition:
        raise MetadataError(message)


def integer(value, label):
    require(type(value) is int, f"{label} must be an integer (bool/float/string prohibited)")
    return value


def interval(values, label):
    require(isinstance(values, (list, tuple)) and len(values) == 2, f"{label} needs two endpoints")
    start, end = (integer(value, label) for value in values)
    require(0 <= start < end, f"{label} must be nonnegative, nonempty and ordered")
    return start, end


def classify_intervals(
    core_us, observation_us, source_duration_us, *, canonical_quantization=False
):
    """Classify half-open integer-us intervals, with no floating containment decision.

    canonical_quantization records an explicitly known millisecond-rounded source
    endpoint. It never grants containment or source feasibility to that overhang.
    """
    start, end = interval(core_us, "core_us")
    actual_start, actual_end = interval(observation_us, "observation_us")
    duration = integer(source_duration_us, "source_duration_us")
    require(duration > 0, "source_duration_us must be positive")
    require(type(canonical_quantization) is bool, "canonical_quantization must be bool")
    require(actual_end <= duration, "observation exceeds true source bounds")
    require(actual_end - actual_start <= MAX_SPAN_US, "observation exceeds fixed 8-second cap")
    require(start < duration, "core starts outside true source bounds")
    overhang = max(0, end - duration)
    if overhang:
        quantized_end = ((duration + 500) // 1000) * 1000
        require(
            canonical_quantization and end == quantized_end and overhang <= 500,
            "core exceeds true source bounds without a recorded canonical quantization endpoint",
        )

    core_length = end - start
    intersection_start, intersection_end = max(start, actual_start), min(end, actual_end)
    covered = max(0, intersection_end - intersection_start)
    missing_start = min(core_length, max(0, actual_start - start))
    missing_end = min(core_length, max(0, end - actual_end))
    contained = actual_start <= start and end <= actual_end
    noncontainment = []
    if missing_start:
        noncontainment.append("observation_starts_after_core_start")
    if missing_end:
        noncontainment.append("observation_ends_before_core_end")
    feasibility = []
    if core_length > MAX_SPAN_US:
        feasibility.append("core_span_exceeds_fixed_8_second_cap")
    if overhang:
        feasibility.append("canonical_core_end_exceeds_true_source_duration")
    divisor = gcd(covered, core_length)
    with localcontext() as context:
        context.prec = 32
        coverage_decimal = format(Decimal(covered) / Decimal(core_length), ".12f")
    return {
        "core_us": [start, end],
        "actual_window_us": [actual_start, actual_end],
        "source_duration_us": duration,
        "core_duration_us": core_length,
        "observation_duration_us": actual_end - actual_start,
        "intersection_us": [intersection_start, intersection_end] if covered else None,
        "covered_us": covered,
        "uncovered_us": core_length - covered,
        "coverage_fraction": f"{covered // divisor}/{core_length // divisor}",
        "coverage_decimal_display_only": coverage_decimal,
        "missing_start_us": missing_start,
        "missing_end_us": missing_end,
        "full_core_contained": contained,
        "full_core_feasible_under_8s_and_source_bounds": not feasibility,
        "fixed_observation_cap_us": MAX_SPAN_US,
        "minimum_required_arbitrary_span_us": core_length,
        "minimum_span_basis": (
            "arbitrary interval lower bound; not a centered integral-ms window proposal"
        ),
        "core_source_end_overhang_us": overhang,
        "noncontainment_reasons": noncontainment,
        "feasibility_reasons": feasibility,
        "endpoint_event_visibility": "not_evaluated_metadata_only",
        "rally_completeness": "not_evaluated_metadata_only",
    }


def _coverage_for_window(row, source, canonical_quantization):
    require(
        isinstance(row, dict) and isinstance(source, dict), "window/source must be metadata objects"
    )
    start_ms, end_ms = interval(row.get("target_ms"), "target_ms")
    duration = integer(source.get("duration_us"), "source duration_us")
    require(duration > 0, "source duration_us must be positive")
    for row_key, source_key in (
        ("source_id", "upload_id"),
        ("source_sha256", "source_sha256"),
        ("session_id", "session_id"),
    ):
        if row_key in row:
            require(
                source_key in source and row[row_key] == source[source_key],
                f"window/source identity mismatch: {row_key}",
            )
    if "source_duration_us" in row:
        require(
            integer(row["source_duration_us"], "source_duration_us") == duration,
            "window/source duration mismatch",
        )
    try:
        if "target_seconds_original" in row:
            original = row["target_seconds_original"]
            require(
                isinstance(original, (list, tuple)) and len(original) == 2,
                "target_seconds_original needs two decimal endpoints",
            )
            require(
                [_milliseconds(value) for value in original] == [start_ms, end_ms],
                "original decimal/target_ms conversion mismatch",
            )
        observed = _observation(row, source)
    except DecimalException as error:
        raise MetadataError(str(error)) from error
    if "center_ms" in row:
        require(
            integer(row["center_ms"], "center_ms") == observed["center_ms"],
            "legacy center mismatch",
        )
    if "requested_window_ms" in row:
        requested = row["requested_window_ms"]
        require(
            isinstance(requested, (list, tuple)) and len(requested) == 2,
            "requested_window_ms needs two endpoints",
        )
        require(
            [integer(value, "requested_window_ms") for value in requested]
            == observed["requested_window_ms"],
            "legacy requested window mismatch",
        )
    if "actual_window_us" in row:
        interval(row["actual_window_us"], "actual_window_us")
        require(
            row["actual_window_us"] == observed["actual_window_us"],
            "legacy observation recipe mismatch",
        )
    if "actual_duration_us" in row:
        require(
            integer(row["actual_duration_us"], "actual_duration_us")
            == observed["actual_duration_us"],
            "legacy observation duration mismatch",
        )
    reconstructed_times = {
        "actual_window_seconds": observed["actual_window_seconds"],
        "actual_window_timecode": observed["actual_window_timecode"],
        "core_timecode": [_timecode(start_ms * 1000), _timecode(end_ms * 1000)],
    }
    for key, expected in reconstructed_times.items():
        if key in row:
            require(row[key] == expected, f"legacy {key} mismatch")
    result = classify_intervals(
        [start_ms * 1000, end_ms * 1000],
        observed["actual_window_us"],
        duration,
        canonical_quantization=canonical_quantization,
    )
    for key in ("window_id", "row_id", "source_id", "source_sha256", "candidate_id", "session_id"):
        if key in row:
            require(isinstance(row[key], str) and row[key], f"{key} must be a nonempty string")
            result[key] = row[key]
    result["source_clipping_reasons"] = observed["short_window_reason"] or []
    result["legacy_center_ms"] = observed["center_ms"]
    result["legacy_requested_window_ms"] = observed["requested_window_ms"]
    return result


def coverage_for_window(row, source, *, canonical_quantization=False):
    """Use the legacy integral-ms midpoint ±4s recipe, then classify coverage."""
    return _coverage_for_window(row, source, canonical_quantization)


def _classify_batch(windows, sources, canonical_quantization):
    require(isinstance(windows, list) and windows, "a nonempty explicit batch is required")
    require(isinstance(sources, dict), "sources must map source IDs to metadata")
    ids, rows = set(), []
    for window in windows:
        require(isinstance(window, dict), "window must be a metadata object")
        identifier = window.get("window_id")
        require(
            isinstance(identifier, str) and identifier and identifier not in ids,
            "window IDs must be nonempty and unique",
        )
        ids.add(identifier)
        source_id = window.get("source_id")
        require(isinstance(source_id, str) and source_id, "source_id must be a nonempty string")
        require(source_id in sources, f"unknown source: {identifier}")
        rows.append(_coverage_for_window(window, sources[source_id], canonical_quantization))
    return rows


def require_full_core(windows, sources, *, canonical_quantization=False):
    """All-or-nothing metadata condition for future workflows, before media work.

    Returns every row in the original order or raises with every computed row and
    all noncontained IDs/reasons. It never filters, reselects or mutates inputs.
    Passing is only this metadata condition, not resource/job/media admission.
    """
    rows = _classify_batch(windows, sources, canonical_quantization)
    if any(not row["full_core_contained"] for row in rows):
        raise StrictCoverageError(rows)
    return rows


def _milliseconds(value):
    require(
        isinstance(value, (str, int, Decimal)) and not isinstance(value, bool), "non-decimal time"
    )
    number = Decimal(str(value))
    require(number.is_finite(), "non-finite time")
    return int((number * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def _timecode(us):
    whole, fractional = divmod(us, 1000000)
    hours, rest = divmod(whole, 3600)
    minutes, sec = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{fractional:06d}"


def _observation(row, source):
    # Preserve the original floor midpoint in integral milliseconds.
    center_ms = sum(row["target_ms"]) // 2
    requested = [center_ms - 4000, center_ms + 4000]
    actual = [max(0, requested[0] * 1000), min(source["duration_us"], requested[1] * 1000)]
    require(actual[0] < actual[1], "empty observation interval")
    clipped = []
    if actual[0] != requested[0] * 1000:
        clipped.append("clipped_at_source_start")
    if actual[1] != requested[1] * 1000:
        clipped.append("clipped_at_source_end")
    return {
        "center_ms": center_ms,
        "requested_window_ms": requested,
        "actual_window_us": actual,
        "actual_window_seconds": [format(Decimal(value) / 1000000, ".6f") for value in actual],
        "actual_window_timecode": [_timecode(value) for value in actual],
        "actual_duration_us": actual[1] - actual[0],
        "short_window_reason": clipped or None,
    }
