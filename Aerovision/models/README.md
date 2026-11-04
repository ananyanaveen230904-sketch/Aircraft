# models/

No trained weights are included in this repo.

- The app runs perfectly well without anything in this folder — it
  uses the classical OpenCV heuristic detector in `src/detector.py`.
- To use a real trained YOLOv8 model instead, train one with
  `train.py` (see instructions there) and place the resulting weights
  file here as:

      models/best.pt

- `src/detector.py` automatically detects `models/best.pt` and
  switches from the classical CV backend to YOLOv8 inference — no
  other code changes needed.
