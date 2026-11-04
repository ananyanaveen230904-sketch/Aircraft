"""
Quick example/verification script for the Decision Fusion Module
(src/fusion.py). Constructs sample RUL output and a sample defect
list (hardcoded, mimicking real output shapes from src/rul/predict.py
and src/detector.py) and prints both the structured dict and the
prompt text, so the format can be checked before wiring it into the
real dashboard.

Run:
    python src/test_fusion.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion import fuse_inspection_results, to_prompt_text


# Mimics src.detector.DefectDetection's shape without importing it
# (keeps this script runnable standalone, and shows fusion.py works
# against either a real DefectDetection object or an equivalent one).
@dataclass
class FakeDetection:
    defect_type: str
    confidence: float
    bbox: tuple
    relative_area: float
    length_cm: float = None


def main():
    print("=" * 70)
    print("EXAMPLE 1: Full record (RUL + defects both present)")
    print("=" * 70)

    # Sample Stage 1 output, mimicking src.rul.predict's return values
    rul_cycles = 18
    health_score = 63
    failure_risk = "MEDIUM"  # as returned by failure_risk_band()

    # Sample Stage 2 output, mimicking src.detector.detect()'s return
    # value — an 800x600 (h, w) image with a crack near the top-left
    # and corrosion near the bottom-right.
    image_shape = (600, 800)
    detections = [
        FakeDetection(defect_type="crack", confidence=0.94, bbox=(50, 40, 80, 60), relative_area=0.01),
        FakeDetection(defect_type="corrosion", confidence=0.87, bbox=(600, 480, 100, 70), relative_area=0.015),
    ]

    record = fuse_inspection_results(
        rul_cycles=rul_cycles,
        health_score=health_score,
        failure_risk=failure_risk,
        detections=detections,
        image_shape=image_shape,
    )
    print("\nStructured dict:")
    print(record)
    print("\nPrompt text:")
    print(to_prompt_text(record))

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Partial record — defects only, no RUL/health data")
    print("=" * 70)

    record2 = fuse_inspection_results(
        detections=[FakeDetection(defect_type="dent", confidence=0.71, bbox=(300, 250, 50, 50), relative_area=0.005)],
        image_shape=image_shape,
    )
    print("\nStructured dict:")
    print(record2)
    print("\nPrompt text:")
    print(to_prompt_text(record2))

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Partial record — RUL only, no defects detected")
    print("=" * 70)

    record3 = fuse_inspection_results(
        rul_cycles=112,
        health_score=90,
        failure_risk="LOW",
        detections=[],
    )
    print("\nStructured dict:")
    print(record3)
    print("\nPrompt text:")
    print(to_prompt_text(record3))

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Edge case — detection with no image_shape given")
    print("=" * 70)

    record4 = fuse_inspection_results(
        detections=[FakeDetection(defect_type="rivet_damage", confidence=0.55, bbox=(10, 10, 20, 20), relative_area=0.002)],
        # image_shape omitted on purpose — location should be dropped, not guessed
    )
    print("\nStructured dict:")
    print(record4)
    print("\nPrompt text:")
    print(to_prompt_text(record4))


if __name__ == "__main__":
    main()
