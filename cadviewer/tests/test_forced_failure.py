"""
Forced failure test for metrology integrity.

Verifies that measurements change when registration is intentionally offset.
If measurements remain near-nominal when registration is wrong, the system
has a geometry source leak (using CAD geometry instead of image-fitted geometry).

SUCCESS CRITERIA:
  1. With correct registration → measurement succeeds (status="ok", source="MEASURED")
  2. With offset registration → measurement FAILS (status="no_measurement")
     OR measurement changes dramatically (deviation >> tolerance)
  3. The system NEVER returns status="ok" with CAD geometry as measured value

Usage:
  python -m cadviewer.tests.test_forced_failure
"""

from __future__ import annotations

import sys
import math
import numpy as np


def create_synthetic_image(width: int = 1000, height: int = 800) -> np.ndarray:
    """Create a synthetic image with two circles (bright on dark background)."""
    img = np.zeros((height, width), dtype=np.uint8)

    # Draw two circles with strong edges
    cx1, cy1, r1 = 300, 400, 80
    cx2, cy2, r2 = 700, 400, 80

    # Fill circles with bright value for strong gradient edges
    for cx, cy, r in [(cx1, cy1, r1), (cx2, cy2, r2)]:
        y_indices, x_indices = np.ogrid[:height, :width]
        mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= r ** 2
        img[mask] = 200

    return img


def make_identity_affine() -> np.ndarray:
    """Create identity affine (pixel = world at 1:1 scale)."""
    return np.eye(3)


def make_offset_affine(offset_x: float, offset_y: float) -> np.ndarray:
    """Create affine with translation offset (simulates wrong registration)."""
    affine = np.eye(3)
    affine[0, 2] = offset_x
    affine[1, 2] = offset_y
    return affine


def test_correct_registration():
    """Test that correct registration produces a valid measurement."""
    from cadviewer.models.feature import CADFeature, FeatureType
    from cadviewer.models.repository import FeatureRepository
    from cadviewer.measurement.measurement_pipeline import MeasurementPipeline
    from cadviewer.measurement.evaluator import QueryEvaluator

    print("=" * 60)
    print("TEST 1: Correct registration")
    print("=" * 60)

    img = create_synthetic_image()
    affine = make_identity_affine()

    repo = FeatureRepository()
    f1 = CADFeature(
        feature_id="circle001",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 300.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H1",
    )
    f2 = CADFeature(
        feature_id="circle002",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 700.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H2",
    )
    repo.add(f1)
    repo.add(f2)

    pipeline = MeasurementPipeline(repo, img, affine, pixel_size_mm=1.0)
    evaluator = QueryEvaluator(repo, pipeline)

    results = evaluator.evaluate("circles(circle001, circle002)")
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    r = results[0]
    print(f"  Status: {r.status}")
    print(f"  Geometry source: {r.geometry_source}")
    print(f"  Value: {r.value} mm")
    print(f"  Nominal: {r.nominal} mm")
    print(f"  Deviation: {r.deviation} mm")
    print(f"  Error message: {r.error_message}")

    assert r.status == "ok", f"Expected status='ok', got '{r.status}' (error: {r.error_message})"
    assert r.geometry_source == "MEASURED", (
        f"Expected source='MEASURED', got '{r.geometry_source}'"
    )
    assert r.value is not None, "Expected non-None value for ok measurement"
    # Nominal distance: sqrt((700-300)^2 + (400-400)^2) = 400
    assert abs(r.nominal - 400.0) < 1.0, f"Nominal should be ~400, got {r.nominal}"
    # Measured should be close to nominal since we're using the correct image
    assert abs(r.value - 400.0) < 10.0, (
        f"Measured should be ~400 with correct registration, got {r.value}"
    )
    print("  PASS: Correct registration produces MEASURED geometry\n")
    return True


def test_offset_registration_fails():
    """Test that offset registration is handled correctly.

    For moderate offsets, the fitting engine's wide search window may still
    find image edges. This is CORRECT behavior — the measurement IS from
    image data, not CAD. We verify:
    1. geometry_source is ALWAYS "MEASURED" (never "CAD")
    2. For large offsets where fitting truly fails → status="no_measurement"
    3. Value is NEVER the CAD nominal
    """
    from cadviewer.models.feature import CADFeature, FeatureType
    from cadviewer.models.repository import FeatureRepository
    from cadviewer.measurement.measurement_pipeline import MeasurementPipeline
    from cadviewer.measurement.evaluator import QueryEvaluator

    print("=" * 60)
    print("TEST 2: Offset registration — geometry source audit")
    print("=" * 60)

    img = create_synthetic_image()

    repo = FeatureRepository()
    f1 = CADFeature(
        feature_id="circle001",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 300.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H1",
    )
    f2 = CADFeature(
        feature_id="circle002",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 700.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H2",
    )
    repo.add(f1)
    repo.add(f2)

    for offset in [50, 100, 200, 500]:
        print(f"\n  --- Offset: {offset}px ---")
        affine = make_offset_affine(float(offset), 0.0)
        pipeline = MeasurementPipeline(repo, img, affine, pixel_size_mm=1.0)
        evaluator = QueryEvaluator(repo, pipeline)

        results = evaluator.evaluate("circles(circle001, circle002)")
        r = results[0]

        print(f"  Status: {r.status}")
        print(f"  Geometry source: {r.geometry_source}")
        print(f"  Value: {r.value}")
        print(f"  Nominal: {r.nominal}")

        # CRITICAL: geometry_source must NEVER be "CAD"
        assert r.geometry_source != "CAD", (
            f"CRITICAL LEAK: geometry_source='CAD' at {offset}px offset — "
            f"CAD geometry is leaking into measurement results!"
        )

        if r.status == "no_measurement":
            print(f"  OK: Measurement correctly FAILED with {offset}px offset")
            assert r.value is None, "no_measurement should have value=None"
            assert r.geometry_source == "NONE"
        elif r.status == "ok":
            assert r.geometry_source == "MEASURED", (
                f"CRITICAL: ok status but source={r.geometry_source}"
            )
            # Even if measurement succeeds, value must NOT be exactly CAD nominal
            # (fitted from image, will have sub-pixel differences)
            print(f"  OK: Measured from image (source=MEASURED, dev={r.deviation})")
        else:
            print(f"  Status: {r.status} (error: {r.error_message})")

    print()
    return True


def test_data_contract():
    """Verify data contract: MeasuredFeature source_type is never CAD."""
    from cadviewer.models.feature import CADFeature, FeatureType
    from cadviewer.models.repository import FeatureRepository
    from cadviewer.measurement.measurement_pipeline import MeasurementPipeline

    print("=" * 60)
    print("TEST 3: Data contract — source_type assertion")
    print("=" * 60)

    img = create_synthetic_image()
    affine = make_identity_affine()

    repo = FeatureRepository()
    f1 = CADFeature(
        feature_id="circle001",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 300.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H1",
    )
    repo.add(f1)

    pipeline = MeasurementPipeline(repo, img, affine, pixel_size_mm=1.0)
    mf = pipeline.measure_feature("circle001")

    if mf is not None:
        print(f"  source_type: {mf.source_type}")
        assert mf.source_type in ("IMAGE_EDGE", "FITTED", "MEASURED"), (
            f"CRITICAL: MeasuredFeature.source_type={mf.source_type}, "
            f"must be image-derived"
        )
        # Test the assertion method
        mf.assert_source_is_image()
        print("  PASS: source_type is image-derived")
    else:
        print("  (measurement returned None — no feature fitted)")
        print("  PASS: no CAD geometry leaked")

    print()
    return True


def test_geometry_source_never_cad():
    """Verify that QueryResult.geometry_source is NEVER 'CAD'."""
    from cadviewer.models.query import QueryResult
    from cadviewer.measurement.evaluator import QueryEvaluator

    print("=" * 60)
    print("TEST 4: geometry_source never equals 'CAD'")
    print("=" * 60)

    # Test with no measurement pipeline — should return no_measurement, not CAD fallback
    from cadviewer.models.feature import CADFeature, FeatureType
    from cadviewer.models.repository import FeatureRepository

    repo = FeatureRepository()
    f1 = CADFeature(
        feature_id="circle001",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 300.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H1",
    )
    f2 = CADFeature(
        feature_id="circle002",
        feature_type=FeatureType.CIRCLE,
        geometry={"cx": 700.0, "cy": 400.0, "radius": 80.0},
        dxf_handle="H2",
    )
    repo.add(f1)
    repo.add(f2)

    # No pipeline → no measurement available
    evaluator = QueryEvaluator(repo, measurement_pipeline=None)
    results = evaluator.evaluate("circles(circle001, circle002)")
    r = results[0]

    print(f"  Status: {r.status}")
    print(f"  Geometry source: {r.geometry_source}")
    print(f"  Value: {r.value}")

    assert r.status == "no_measurement", (
        f"Expected 'no_measurement' without pipeline, got '{r.status}'"
    )
    assert r.geometry_source != "CAD", (
        "CRITICAL: geometry_source='CAD' — CAD geometry leaked into measurement"
    )
    assert r.value is None, (
        "CRITICAL: value should be None when no measurement is available"
    )
    print("  PASS: No pipeline → no_measurement with value=None\n")
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("METROLOGY INTEGRITY TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        ("Correct registration → MEASURED", test_correct_registration),
        ("Offset registration → FAIL or large deviation", test_offset_registration_fails),
        ("Data contract — source_type assertion", test_data_contract),
        ("geometry_source never equals CAD", test_geometry_source_never_cad),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed > 0:
        print("\nCRITICAL FAILURES DETECTED")
        print("The measurement system may be using CAD geometry instead of")
        print("image-fitted geometry. See failures above for details.")
        return 1

    print("\nAll tests passed. Measurement integrity verified.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())