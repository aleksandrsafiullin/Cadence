# Decisions

## Lockfile

Date: 2026-10-09.

`uv.lock` is the output of `uv lock` for this repository, then applied with `uv sync`. It is not written by hand. `torch` and `torchvision` stay unpinned in `pyproject.toml` so the resolver can match `rfdetr[train,augment]==1.11.2`. `.gitignore` does not ignore `uv.lock`.

## RF-DETR 1.11.2

Date: 2026-10-09. darwin aarch64, CPython 3.11.16, uv 0.12.13.

`uv lock` resolved `rfdetr` 1.11.2 with extras `train` and `augment` (142 packages). `uv sync` installed that lock. `importlib.metadata.version("rfdetr")` reports `1.11.2`. No replacement.

Unpinned `torch` resolved to 2.14.1 and `torchvision` to 0.29.1. Those versions live in `uv.lock` only.

The `train` extra depends on `roboflow` 1.7.0 on this platform. The Roboflow `inference` package is not in `uv.lock`. This project does not call a hosted endpoint.

## eval.max_detections

Date: 2026-10-09.

Spec §18 requires a fixed max-detections cap so the precision-recall curve is not clipped by the 0.10 operating floor, and it does not give a number. `configs/pilot.yaml` sets `eval.max_detections` to 100. The evaluation score floor stays 0.001. This cap is a starting choice, not a measured optimum.

## RF-DETR 1.11.2 RGB and box coordinates

Date: 2026-10-09.

Read from the installed package (`import rfdetr` resolves to `.venv/lib/python3.11/site-packages/rfdetr`, version 1.11.2). `RFDETR.predict` in `rfdetr/detr.py` says the image is RGB channel order. The NumPy path does not call `cv2.cvtColor` and does not swap channels. A PIL image whose mode is not RGB is converted to RGB once inside `predict`. A negative-stride view such as `frame[:, :, ::-1]` is copied so `torch.from_numpy` can read it; that copy does not swap channels again. A search of `rfdetr/*.py` finds no `cvtColor` and no `COLOR_BGR`. The only BGR mention is that negative-stride comment.

The adapter copies contiguous RGB from the reader (`rgb24`) and does not convert it again. `bgr_to_rgb` is the one swap for a buffer that is actually BGR. A channel-reversed view passed to `predict` raises `ColorOrderError` instead of being treated as a second swap.

Resize in `predict` is a stretch, not a letterbox: `torchvision.transforms.functional.resize` to `(resolution, resolution)` with `antialias=False`, then ImageNet normalize (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225). `PostProcess._gather_and_scale_boxes` multiplies normalized xyxy by the original width and height (`target_sizes`). Those boxes are already in original-image pixels.

`ModelMetadata.boxes_are_original` is true for this path. `inverse_boxes` returns those boxes unchanged. The same function still undoes an explicit letterbox (`scale`, `pad_x`, `pad_y`) when `boxes_are_original` is false. Official weights are constructed only in `BaselineCoco` via `RFDETRNano`, and only when the caller did not inject a module. Default tests inject a module or use `FakeDetector`. They do not construct `RFDETRNano`.

## Doctor probe class is RFDETRNano

Date: 2026-10-09.

Read from the installed package (`.venv/lib/python3.11/site-packages/rfdetr`, version 1.11.2). `rfdetr/variants.py` defines `class RFDETRNano(RFDETR)` with `size = "rfdetr-nano"` and `_model_config_class = RFDETRNanoConfig`. `rfdetr/__init__.py` exports that name. `RFDETRNanoConfig.pretrain_weights` defaults to `rf-detr-nano.pth`. The doctor probe constructs this class, not Small and not a stand-in detector.

`RFDETR.train` in `rfdetr/detr.py` starts a full PyTorch Lightning fit. `RFDETRModelModule.training_step` is a Lightning hook that needs a trainer and a dataset batch. There is no public one-step method on `RFDETR`. `select_training_device()` therefore calls `tensor_optimizer_step`: one forward, backward, and SGD update on a single parameter on that device. `training_step.method` is `tensor_optimizer_step`. The report records `passed` only after that step returns `completed: true`, which the function sets only after the parameter changes.

`excavation doctor` without `--load-weights` does not construct `RFDETRNano` and does not download. `excavation doctor --load-weights` passes `pretrain_weights` as an absolute path `workspace/cache/weights/rf-detr-nano.pth`. A bare filename would use `~/.roboflow/models` (`RF_HOME`). `download_pretrain_weights` skips the network when that file is already present and the MD5 matches. MPS does not expose a peak-memory counter (`torch.mps` has current and driver totals only); that field stays null. CUDA records `max_memory_allocated` when that backend ran the probe. CPU has no peak counter, so peak stays null there too. A missing measurement is null, not 0.

## RF-DETR 1.11.2 category ids

Date: 2026-10-10.

Read from the installed package (`.venv/lib/python3.11/site-packages/rfdetr`, version 1.11.2), `rfdetr/datasets/coco.py`.

`build_roboflow_from_coco` is the loader for a custom detection dataset. Its `PATHS` map the split key `"val"` to the directory `valid/` (not `val/`), and `"train"` / `"test"` to `train/` and `test/`. Each split reads `_annotations.coco.json` in that directory. The builder passes `remap_category_ids=True`.

With remapping on, `CocoDetection` builds `cat2label` as `{category_id: index}` by enumerating `filter_parent_categories(...)`. That helper sorts categories by id. The model label is the index in that sorted list, not the raw COCO `category_id`, unless the ids are already `0 .. N-1` in that order. Validation and test reuse the train split's map (`_train_split_cat2label`) so a category missing from one split does not shift later labels. `filter_parent_categories` drops an unannotated grouping node whose name is another category's `supercategory`. A flat list whose `supercategory` is `"none"` is kept, including categories that have no boxes.

The official COCO builder does not remap for detection (`remap_category_ids=include_keypoints`, false for boxes). There the raw `category_id` is the label, which matches pretrained heads where slot `k` is COCO category `k`.

`ConvertCoco` drops annotations whose `iscrowd` is not 0. That is not how this project records occlusion. Import rejects `iscrowd` other than 0 with `UnsupportedCrowdError`.

`class_embed.bias` in a checkpoint has length `num_classes + 1` (`rfdetr/utilities/state_dict.py`). The extra logit is the detector background class. It is not COCO category id 0, and it is not `excavator_body`.

`class_mapping.json` is the authority for this lab. Import resolves a category by `name`. Snapshot COCO writes `category_id` equal to the internal id (`excavator_body` 0, `bucket` 1, `haul_truck` 2, `truck_bed` 3) and `supercategory` `"none"`. Those ids are already contiguous and sorted, so RF-DETR's remap leaves them on the same indices. A source id such as 99 named `bucket` is stored as internal id 1. A source id 0 named `dump` is rejected. It is not rewritten as background or as `excavator_body`.

## RF-DETR 1.11.2 train profile mapping

Date: 2026-10-10.

Read from the installed package (`.venv/lib/python3.11/site-packages/rfdetr`, version 1.11.2).

`RFDETR.train` in `rfdetr/detr.py` is `train(self, **kwargs: Any) -> None`. `RFDETR.__init__` is `__init__(self, *, trust_checkpoint: bool = False, **kwargs: Any) -> None`. Kwargs to `__init__` go to the variant's `ModelConfig` (`RFDETRNanoConfig` or `RFDETRSmallConfig`). `_prepare_run_config` pops `device` and `resolution`, then `get_train_config` builds `TrainConfig(**kwargs)`. `TrainConfig` is `extra="forbid"`. On this install the only keys popped before that constructor are `device` and `resolution`. `resolution` is a `ModelConfig` field, not a `TrainConfig` field. `gradient_checkpointing` is a `ModelConfig` field and is absent from `TrainConfig`.

`rfdetr.variants.RFDETRNano` has `size = "rfdetr-nano"` and `_model_config_class = RFDETRNanoConfig` (default resolution 384, patch 16, `num_windows` 2). `RFDETRSmall` has `size = "rfdetr-small"` and `RFDETRSmallConfig` (default resolution 512, same patch and windows). Both use `TrainConfig`, not `KeypointTrainConfig`. 384 and 512 are divisible by `patch_size * num_windows` (32).

`models/train_config.py` maps `configs/pilot.yaml` profile keys as follows. Epoch counts stay 2 and 50.

| Project key | Payload | Installed name |
|---|---|---|
| `variant` | constructor class selector | `RFDETRNano` / `RFDETRSmall`. There is no config field named `variant`. |
| `resolution` | constructor | `resolution`. Not repeated on the train payload. |
| `gradient_checkpointing` | constructor only | `gradient_checkpointing`. Omitted from YAML means false, which is the `ModelConfig` default. Never a train kwarg. |
| `epochs` | train | `epochs` |
| `batch_size` | train | `batch_size`. Integer. `"auto"` is rejected. The `TrainConfig` default is `4`. |
| `gradient_accumulation` | train | `grad_accum_steps` |
| `learning_rate` | train | `lr` |
| `encoder_learning_rate` | train | `lr_encoder`. The installed default is `1.5e-4`. Profiles send `1e-5`. |
| `early_stopping` | train | `early_stopping` |
| `patience` | train when early stopping is on | `early_stopping_patience` |
| `min_delta` | train when early stopping is on | `early_stopping_min_delta` |
| `best_skip` | train | `skip_best_epochs` |
| `workers` | train | `num_workers`. The installed default is `2`. Profiles send `0`. |
| `amp_dtype` | train | `amp_dtype=None` (FP32). The installed default is `"auto"`, not `None`. |
| `seed` | train | `seed` |
| `run_test` | train | `run_test=False`. The document has no `dataset_dir` and no test-directory path. |
| checkpoint metric | train | `best_model_metric="map"`. For detection the trainer monitors `val/mAP_50_95` (`_BEST_MODEL_MONITOR_KEYS` in `rfdetr/training/trainer.py`). |

Smoke leaves `patience` and `min_delta` null. Those values are `not_applied`. `early_stopping_patience` and `early_stopping_min_delta` are ints (installed defaults 10 and 0.001). Sending null is invalid, and the defaults are not written into `train_kwargs` while `early_stopping` is false. Pilot-small sends 12 and 0.001.

`_resolve_amp_dtype` returns `None` when `train_config.amp_dtype` is `None`, because that is not the default `"auto"`. `ModelConfig.amp` still defaults to `True` and is not set on the constructor. The explicit `amp_dtype` wins.

`compile` and `cuda_graphs` default to false on `ModelConfig` and are omitted. `amp_dtype="fp8"` is not sent. `export` is a different method and is not called, so ONNX and CoreML are not in the payload.

`augmentation_train` is the mild preset body. `augmentation_eval` is deterministic preprocess plus the library off switch. The names are project keys, so `library_train_kwargs()` drops them. The library fields `aug_config`, `scale_jitter`, and `multi_scale` are set on the train payload and on both eval payloads. See the next section. An unset `aug_config` is not recorded as "no preset": in rfdetr 1.11.2 that default is torchvision `RandomHorizontalFlip(p=0.5)`.

Per-class bucket AP has no train argument. `not_applied.bucket_metric` records that. Checkpoint selection stays `best_model_metric="map"`.

Unrecognized keys on the profile mapping, including real `TrainConfig` names such as `wandb`, go to `ignored_keys` and are not forwarded. A typo `epoch` does not replace `epochs`.

This module does not call `RFDETR.train`. `dataset_dir` is still required by `TrainConfig` at the call, and that call is a later step. The resolved JSON does not contain it.

## Mild train augmentation

Date: 2026-10-10.

Preset name: `mild`, implemented in `models/augment.py`. It is the v0.1 preset, not a tuned optimum. The resolved `augmentation_train` body is:

| Key | Value |
|---|---|
| `brightness_limit` | 0.2 (±20% brightness gain) |
| `contrast_limit` | 0.2 (±20% contrast gain) |
| `scale_limit` | 0.1 (at most 10% of each axis removed by the crop) |
| `horizontal_flip_p` | 0.5 |

Contrast is centered on the image mean, then brightness scales the pixels: `((pixel - mean) * (1 + contrast) + mean) * (1 + brightness)`, rounded and clipped to uint8. The crop window shifts boxes and clips them to the window. A box that would have non-positive width or height rejects the whole crop. A box that only crosses the edge stays, clipped to a positive area. Horizontal flip rewrites `x1' = width - x2`, `x2' = width - x1`. There is no vertical-flip code path.

Library key names on the train payload and on `validation_kwargs` / `test_kwargs`:

| `TrainConfig` field | Value we send | Installed default if omitted |
|---|---|---|
| `aug_config` | `{}` | `None`, which `make_coco_transforms` treats as torchvision `RandomHorizontalFlip(p=0.5)` |
| `scale_jitter` | `False` | `True`, the resize → `RandomSizedCrop(min_max_height=[384, 600])` → resize branch |
| `multi_scale` | `False` | `MultiScale.PER_BATCH` (`"per-batch"`). `False` is stored as `MultiScale.OFF` (`"off"`) |

`aug_config={}` is the empty-dict off switch documented in `make_coco_transforms` and in `rfdetr/datasets/aug_configs.py`. `augmentation_eval.mode` is `deterministic_preprocess`. Validation and test in `augment.apply` call `model_rgb_frame` and copy the boxes.

Named presets `AUG_CONFIG`, `AUG_CONSERVATIVE`, `AUG_AGGRESSIVE`, `AUG_AERIAL`, and `AUG_INDUSTRIAL` are not selected. `AUG_AGGRESSIVE` and `AUG_AERIAL` include `VerticalFlip`. `AUG_INDUSTRIAL` adds blur and noise. `AUG_CONSERVATIVE` is a horizontal flip plus ±10% brightness and has no box-safe ±10% crop. The `scale_jitter` crop is wider than ±10% and can drop boxes. None of those hooks reject a crop that would delete a box or clip a boundary box to a still-positive area, so the mild preset stays in `augment.py` and the library switch stays off. `expanded_scales` is omitted. It does not change the resize while `multi_scale` is false.

`save_examples` writes `before.png` and `after.png`. It does not call `train`. Evaluate is a later command and does not apply this preset.

## Bundle reload safe load

Date: 2026-10-10.

Read from the installed package (`.venv/lib/python3.11/site-packages/rfdetr`, version 1.11.2), `rfdetr/utilities/io.py`, and torch 2.14.1.

`_safe_torch_load(path, *, trust=False)` is the library loader. The flag on the call we use is `trust=False`. Inside that helper the torch flag is `weights_only=True`:

1. `torch.load(path, map_location="cpu", weights_only=True)`.
2. The same `weights_only=True` load inside `torch.serialization.safe_globals([argparse.Namespace, types.SimpleNamespace])` when the strict load raises `RuntimeError` or `pickle.UnpicklingError`. The allow-list is only that call.
3. `torch.load(..., weights_only=False)` runs only when `trust=True`.

`load_bundle` calls `_safe_torch_load(path, trust=False)` on `weights.pth`. It does not pass `trust=True`, `trust_checkpoint=True`, or `weights_only=False`. A file that cannot be read on that path raises `IncompatibleCheckpointError`. The resume file `training_checkpoint.ckpt` is not passed to the loader.

`load_pretrain_weights` in `rfdetr/models/weights.py` indexes `checkpoint["model"]["class_embed.bias"]` after it has built the Nano or Small module. A metadata-only state dict has no detection head, so the unit suite does not construct `RFDETRNano` or `RFDETRSmall`. `load_bundle` still returns the adapter: class order, variant, and resolution are checked against `class_mapping.json` and `model_config.json` on the safe load. The first `predict` calls `RFDETRNano` or `RFDETRSmall` with `pretrain_weights` set to the absolute `weights.pth` path, `resolution` and `num_classes` from the bundle, and `trust_checkpoint=False`. No score-threshold argument is passed. `predict` then uses the same raw floor as the baseline adapter (`LIBRARY_SCORE_THRESHOLD`, 0.0), not 0.10 and not the display threshold 0.35.

`download_pretrain_weights` looks up `os.path.basename(pretrain_weights)`. `weights.pth` is not in `ModelWeights`, so that function returns without a download. Official COCO weights are not opened when `weights.pth` is missing (`CheckpointMissingError`). `BaselineCoco` is not constructed.

## ByteTrack 2.6.1 vehicle arguments

Date: 2026-10-10.

Read from the installed package (`importlib.metadata.version("trackers")` is `2.6.1`), `trackers/core/bytetrack/tracker.py`. `from trackers import ByteTrackTracker`.

`ByteTrackTracker.__init__` does not use the project names `high_confidence`, `activation`, or `min_observations`. This module passes the installed names:

| Project key | Installed argument | Starting value |
|---|---|---|
| `bytetrack.high_confidence` | `high_conf_det_threshold` | 0.35 |
| `bytetrack.activation` | `track_activation_threshold` | 0.45 |
| `bytetrack.min_observations` | `minimum_consecutive_frames` | 2 |
| `bytetrack.lost_buffer_s` × `reference_fps` | `lost_track_buffer` | 60 for 2.0 s at 30 fps |
| `bytetrack.reference_fps` | `frame_rate` | 30.0 |

`lost_track_buffer` is a count of reference 30 fps frames. `seconds_to_lost_track_buffer(2.0)` is 60. Analysis at 5 fps does not change that argument to 10. `frame_rate` is the same reference, 30, not the analysis fps. The library would otherwise compute `ceil(frame_rate / 30 * lost_track_buffer)`, which is 10 when `frame_rate` is 5. With these arguments, `maximum_frames_without_update` is 60 and `maximum_time_without_update` is `60 / 30` = 2.0 s.

`ByteTrackTracker.update` parameters are `detections`, `frame`, and `timestamp`. There is no `source_time_s` parameter. `timestamp` is absolute frame time in seconds. The adapter passes the reader's `source_time_s` as `timestamp`. It does not pass `source_frame_index` or `analysis_index`, and it does not pass `frame`.

timestamp_api: timestamp

The emitted vehicle row still stores `source_time_s` itself. `timestamp_api` is not `unsupported`, because a seconds parameter exists under the name `timestamp`. That is what makes the 2.0 s budget (`lost_track_buffer / 30`) the prune clock. `bucket` and `truck_bed` are not submitted to either tracker.

## UI locale

Date: 2026-10-11.

Russian is the default UI locale. English is selectable. The spec's "interface in Russian" remains the default.

## Page server

Date: 2026-10-11.

Streamlit left because the Video page could not be a real layout.

The replacement is Starlette + Jinja2 + CSS + small JS.

FastAPI, React, and a SPA bundler stay forbidden. Services and the worker lock stay.
