# PCDNet paper comparison

The parent `visualize_two_weights_paper.py` includes the PCDNet checkpoint trained
by `ddc/PZ_8/train_pcdnet.py`. Model definitions and required inference helpers are
bundled here from `compare/AAAI24-PCDNet`; the original license is retained in
`LICENSE`. Relative imports keep the legacy `models` and `utils` packages isolated
from Ultralytics. Checkpoint loading maps their original names into this package.
No separate PZ_8 or AAAI24-PCDNet checkout is needed on the server.

## Select named models

Repeat `--net NAME /absolute/path/to/best.pt` to choose any number of models.
Only the supplied models are loaded and evaluated; the legacy baseline defaults
are not required in this mode. YOLO and RT-DETR are detected from the checkpoint.
Use the `pcdnet:` prefix on a name to select the migrated PCDNet loader:

```bash
python visualize_two_weights_paper.py \
  --net YOLO11n /path/to/yolo11n/best.pt \
  --net RTDETR_L /media/ddc/新加卷/hys/qmy/pz_fusion/runs/train/rtdetr_l_pz_rgb_scratch/weights/best.pt \
  --net pcdnet:PCDNet /media/ddc/新加卷/hys/qmy/PZ_8/runs/train/pcdnet_s0_aolp_dolp/weights/best.pt \
  --net RGB_DoLP /path/to/nodepth/best.pt \
  --net RGB_DoLP_Depth /path/to/depth/best.pt \
  --data ultralytics/cfg/datasets/pz7-rgb-color-dolp-depth.yaml \
  --output runs/paper_compare --device 0
```

The comparison order is ground truth first, followed by the order of `--net`
arguments. Individual panels are saved to `ground_truth/`, `YOLO11n/`,
`RTDETR_L/`, `PCDNet/`, `RGB_DoLP/`, and `RGB_DoLP_Depth/` in this example.
Names must be unique directory names; quote names or paths containing spaces.
Names appear in the terminal and JSON keys, without adding titles to images.

Complete comparisons are enabled by the script variable `SAVE_COMPARISON = True`.
Append `--no-save-comparison` to save only the ground-truth and individual model
panels; append `--save-comparison` to explicitly enable complete comparisons.
The switch applies to newly generated images and does not remove files from
earlier runs. Every selected model still runs full-split validation, and results
are printed to the terminal and written to `validation_metrics.json`.

## Legacy command

Existing `--weights` commands remain supported with their original defaults and
directory names. Use either `--net` or `--weights`, not both:

```bash
python visualize_two_weights_paper.py \
  --weights /path/to/nodepth/best.pt /path/to/depth/best.pt \
  --official-weight /path/to/yolo26n/best.pt \
  --yolo12-weight /path/to/yolo12n/best.pt \
  --rtdetr-weight /media/ddc/新加卷/hys/qmy/pz_fusion/runs/train/rtdetr_l_pz_rgb_scratch/weights/best.pt \
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
4. RGB-only RT-DETR-L (`rtdetr_l/`).
5. PCDNet (`pcdnet/`).
6. RGB + DoLP without depth (`rgb_dolp/`).
7. RGB + DoLP + depth (`rgb_dolp_depth/`).

The supplied RT-DETR path is the default for `--rtdetr-weight`. It uses the native
RT-DETR predictor's square stretch and decoder postprocessing on the S0 RGB image.
Its final validation uses the native RT-DETR validator with a three-channel,
`rgb_s0: true` dataset configuration.

The two fusion checkpoints independently select their input format: four/five
channels use scalar `*_dolp_rgb.png`, while six/seven channels use color
`*_dolp_color_rgb.png`. A five-channel depth checkpoint can be compared with a
six-channel color no-depth checkpoint using the same data YAML. When its
`dolp_channels` differs from a checkpoint's format, the YAML's DoLP directory is
ignored for that checkpoint and its maps are read beside the base RGB image.
An explicit `--dolp-root` overrides this location for both fusion checkpoints.
Visualization and final validation use the same per-checkpoint configuration.

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
IDs must match across the six checkpoints.

At the end, all six models are evaluated on the entire selected split, even when
`--max-images` limits visualization. PCDNet uses confidence 0.001 and NMS IoU 0.65,
matching its previous evaluator; the visualization uses `--conf` and `--iou`.
PCDNet AP uses the same Ultralytics matching/metrics as the other comparisons.
Metrics appear in the terminal and in `validation_metrics.json`.
