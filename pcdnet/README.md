# PCDNet paper comparison

The parent `visualize_two_weights_paper.py` includes the PCDNet checkpoint trained
by `ddc/PZ_8/train_pcdnet.py`. Model definitions and required inference helpers are
bundled here from `compare/AAAI24-PCDNet`; the original license is retained in
`LICENSE`. Relative imports keep the legacy `models` and `utils` packages isolated
from Ultralytics. Checkpoint loading maps their original names into this package.
No separate PZ_8 or AAAI24-PCDNet checkout is needed on the server.

Run from the comparison repository with your existing comparison arguments and:

```bash
python visualize_two_weights_paper.py \
  --weights /path/to/nodepth/best.pt /path/to/depth/best.pt \
  --official-weight /path/to/yolo26n/best.pt \
  --yolo12-weight /path/to/yolo12n/best.pt \
  --pcdnet-weight /media/ddc/新加卷/hys/qmy/PZ_8/runs/train/pcdnet_s0_aolp_dolp/weights/best.pt \
  --data ultralytics/cfg/datasets/pz7-rgb-color-dolp-depth.yaml \
  --output runs/paper_compare --device 0
```

The supplied PCDNet path is also the default, so existing commands can omit
`--pcdnet-weight` when that server path exists.

Panel order, left to right:

1. Ground truth (`ground_truth/`).
2. YOLO12n (`yolo12n/`).
3. YOLO26n (`rgb_baseline/`).
4. PCDNet (`pcdnet/`).
5. RGB + color DoLP without depth (`rgb_dolp/`).
6. RGB + color DoLP + depth (`rgb_dolp_depth/`).

All panels use Ultralytics `Results.plot`, consistent class colors, class labels,
and prediction confidence values. No extra model titles are drawn on the images.
The complete comparison images, individual panels, and `predictions.json` are saved.

## PCDNet inputs

For a split entry `/dataset/images/scene.png`, the training-compatible inputs are:

- `/dataset/images/scene_S0_rgb.png`: S0 RGB, three channels.
- `/dataset/images/scene_aolp_rgb.png`: scalar grayscale AoLP, repeated three times.
- `/dataset/images/scene_dolp_rgb.png`: scalar grayscale DoLP, repeated three times.

Images are resized as float32 to the requested long side, padded to a square, and
divided by 255, matching the PZ_8 validation loader. Color DoLP used by the other
two fusion models is not substituted for PCDNet's scalar DoLP. No NIR or depth
images are required for PCDNet itself.

Use `--aolp-root` and `--pcdnet-dolp-root` if the scalar maps are stored elsewhere;
these may also be set as `aolp_root` and `pcdnet_dolp_root` in the data YAML. They
are independent of the existing color `--dolp-root`. All trained class names and
IDs must match across the five checkpoints.

At the end, all five models are evaluated on the entire selected split, even when
`--max-images` limits visualization. PCDNet uses confidence 0.001 and NMS IoU 0.65,
matching its previous evaluator; the visualization uses `--conf` and `--iou`.
PCDNet AP uses the same Ultralytics matching/metrics as the other comparisons.
Metrics appear in the terminal and in `validation_metrics.json`.
