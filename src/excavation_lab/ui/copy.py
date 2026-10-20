"""UI string catalogs for Russian and English.

Russian is the default. Compute code may import this module. It does not
import Streamlit, read the filesystem, or follow a browser locale.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# Switcher endonyms are shared. They are not translated catalog sentences.
LANGUAGE_RUSSIAN = "Русский"
LANGUAGE_ENGLISH = "English"
LANGUAGE_OPTIONS: tuple[str, str] = (LANGUAGE_RUSSIAN, LANGUAGE_ENGLISH)

_PAGE_IDS: tuple[str, ...] = (
    "video",
    "data",
    "training",
    "analysis",
    "quality",
    "actions",
    "events",
)
_CLASS_IDS: tuple[str, ...] = ("excavator_body", "bucket", "haul_truck", "truck_bed")
_SAMPLE_PLAN_IDS: tuple[str, ...] = ("diagnostic", "content", "expanded")
_JOB_KEYS: tuple[str, ...] = (
    "queued",
    "running",
    "completed",
    "failed",
    "cancel_requested",
    "cancelled",
)
_ERROR_KIND_KEYS: tuple[str, ...] = ("false_positive", "false_negative")
_METRIC_KEYS: tuple[str, ...] = (
    "ap50",
    "ap50_95",
    "precision",
    "recall",
    "recall_clear",
    "recall_partial",
    "reviewed_frame_count",
    "evaluated_frame_count",
    "gt_object_count",
    "gt_count",
    "excluded_label_fraction",
    "decode_ms",
    "inference_ms",
    "encode_ms",
    "score_floor",
    "operating_threshold",
    "iou",
    "max_detections",
    "idf1",
    "hota",
)
_EVENT_INDICATOR_COUNT = 12
_MESSAGE_NAMES: tuple[str, ...] = (
    "not_train_frame_message",
    "frame_not_found_message",
    "frame_ambiguous_message",
    "eval_not_found_message",
    "sent_to_labeling_message",
    "episode_ambiguous_message",
    "hardware_caption",
    "hardware_unavailable",
    "weights_warning",
    "allowed_directory_extras",
    "duplicate_video_message",
)
_MESSAGE_SAMPLES: dict[str, tuple[tuple[object, ...], ...]] = {
    "not_train_frame_message": (("frame-1", "train"),),
    "frame_not_found_message": (("frame-1",),),
    "frame_ambiguous_message": (("frame-1",),),
    "eval_not_found_message": (("eval-1",), ("",)),
    "sent_to_labeling_message": (("frame-1",),),
    "episode_ambiguous_message": (("episode-1",),),
    "hardware_caption": (
        ("cpu", False, False, None, False),
        ("cuda", True, False, 4096, True),
    ),
    "hardware_unavailable": (("ffprobe missing",),),
    "weights_warning": (("/cache/weights",),),
    "allowed_directory_extras": (("clips",), ("clips", "dumps")),
    "duplicate_video_message": ((),),
}


def _require_keys(
    actual: dict[str, str], expected: tuple[str, ...], label: str
) -> None:
    if tuple(actual) != expected:
        raise ValueError(f"{label} keys must be {expected}")


@dataclass(frozen=True, slots=True)
class Catalog:
    not_calculated: str
    not_evaluated: str
    not_evaluated_caption: str
    page_quality: str
    send_to_labeling: str
    target_caption: str
    errors_unavailable: str
    errors_from_matches: str
    metrics_broken: str
    metrics_object: str
    match_broken: str
    sample_broken: str
    invalid_frame: str
    invalid_batch: str
    class_labels: dict[str, str]
    metric_labels: dict[str, str]
    episode_overlap: str
    product_title: str
    kicker_project: str
    kicker_workflow: str
    kicker_appearance: str
    page_video: str
    page_data: str
    page_training: str
    page_analysis: str
    page_actions: str
    page_events: str
    project_label: str
    no_project: str
    project_help: str
    add_local_video: str
    local_path_label: str
    local_path_placeholder: str
    video_metadata_unknown: str
    video_registered: str
    allowed_directory: str
    allowed_directory_no_project: str
    path_help: str
    add_video_button: str
    load_video_button: str
    add_another: str
    path_required: str
    need_project: str
    video_added: str
    upload_label: str
    upload_button: str
    upload_help: str
    upload_missing: str
    upload_empty: str
    upload_too_big: str
    upload_bad_name: str
    advanced: str
    source_url_label: str
    source_url_help: str
    rotation_label: str
    meta_duration: str
    meta_fps: str
    meta_fps_nominal: str
    meta_fps_average: str
    meta_frame: str
    meta_codec: str
    meta_sha: str
    meta_device: str
    hardware_probes: str
    offset_label: str
    interval_start: str
    interval_end: str
    interval_help: str
    interval_required: str
    baseline_button: str
    baseline_caption: str
    baseline_help: str
    baseline_queued: str
    probe_button: str
    refresh_runs: str
    runs_heading: str
    runs_empty: str
    runs_no_boxes: str
    video_choice: str
    playback_file: str
    playback_preview: str
    playback_failed: str
    choose_file_button: str
    choose_file_prompt: str
    choose_failed: str
    job_status_labels: dict[str, str]
    data_empty: str
    freeze_split: str
    propose_split: str
    freeze_done: str
    freeze_note: str
    propose_done: str
    propose_kept_frozen: str
    split_hash: str
    proposal_missing: str
    proposal_broken: str
    proposal_legacy: str
    proposal_heading: str
    frozen_heading: str
    independence_label: str
    episodes_heading: str
    episodes_empty: str
    episode_id: str
    episode_group: str
    episode_label: str
    episode_concurrent: str
    add_episode: str
    episode_added: str
    default_episode_label: str
    cut_here: str
    cut_hint: str
    concurrent_pass: str
    remove_cut: str
    save_episode_label: str
    episodes_frozen_note: str
    frame_step_label: str
    arrow_frame_hint: str
    nudge_hint: str
    episodes_saved: str
    episodes_save_failed: str
    guards_heading: str
    guards_empty: str
    split_heading: str
    sample_heading: str
    sample_rule: str
    sample_plan: str
    sample_plan_button: str
    sample_split: str
    sample_count: str
    sample_count_button: str
    sample_spacing: str
    plan_counts_caption: str
    contact_heading: str
    contact_empty: str
    cvat_heading: str
    export_cvat: str
    import_coco: str
    import_file: str
    import_path_required: str
    import_bad_name: str
    batch_split: str
    include_predictions: str
    predictions_help: str
    step_label_title: str
    step_snapshot_title: str
    step_episodes_text: str
    step_split_text: str
    step_sample_text: str
    step_label_text: str
    step_snapshot_text: str
    step_now: str
    step_done: str
    step_wait: str
    step_wait_episode: str
    step_wait_frozen: str
    step_wait_frames: str
    step_snapshot_open: str
    step_snapshot_written: str
    episode_group_help: str
    episode_label_help: str
    freeze_once: str
    split_name_train: str
    split_name_validation: str
    split_name_test: str
    equipment_block_help: str
    action_block_help: str
    block_guard_help: str
    block_guard_gap: str
    block_unassigned: str
    block_overlap: str
    add_block: str
    save_blocks: str
    block_start: str
    block_end: str
    block_role: str
    blocks_saved: str
    blocks_save_failed: str
    sample_plan_help: str
    label_class_names: str
    add_class: str
    class_added: str
    add_class_name: str
    add_class_title: str
    add_class_box: str
    class_box_lines: dict[str, str]
    label_review_rule: str
    label_zip_back: str
    export_path_note: str
    download_batch: str
    contact_no_boxes: str
    snapshot_dataset: str
    snapshot_locked: str
    snapshot_ready: str
    snapshot_other: str
    snapshot_no_train: str
    snapshot_no_validation: str
    snapshot_gate_scope: str
    label_counts_this_video: str
    label_counts_project: str
    label_counts_snapshot: str
    review_heading: str
    count_reviewed: str
    count_unreviewed: str
    count_excluded: str
    count_reviewed_empty: str
    guard_prefix: str
    file_prefix: str
    plan_diagnostic_prefix: str
    plan_content_prefix: str
    episode_boundary_error: str
    column_episode: str
    column_start_s: str
    column_end_s: str
    column_label: str
    column_split_group: str
    column_split: str
    sample_plan_labels: dict[str, str]
    training_empty: str
    train_dataset: str
    train_dataset_empty: str
    train_variant: str
    train_profile: str
    train_device: str
    train_start: str
    train_cancel: str
    train_status: str
    train_epoch: str
    train_checkpoint: str
    train_eta: str
    train_jobs: str
    train_jobs_empty: str
    train_queued: str
    train_cancelled: str
    train_resolution: str
    train_epochs: str
    train_batch: str
    train_accumulation: str
    train_lr: str
    train_encoder_lr: str
    train_workers: str
    train_seed: str
    smoke_help: str
    pilot_help: str
    device_help: str
    train_start_help: str
    train_cancel_help: str
    dataset_not_ready: str
    snapshot_not_built: str
    train_log_broken: str
    analysis_empty: str
    analysis_no_video: str
    analysis_interval_unknown: str
    analysis_fps_label: str
    analysis_fps_help: str
    analysis_player_help: str
    analysis_start: str
    analysis_start_help: str
    analysis_export: str
    analysis_export_help: str
    analysis_queued: str
    export_queued: str
    player_waiting: str
    custom_needs_bundle: str
    model_mode: str
    mode_baseline: str
    mode_custom: str
    bundle_label: str
    bundle_synthetic: str
    synthetic_card_use: str
    synthetic_card_role: str
    threshold_new_run: str
    bt_high: str
    bt_activation: str
    bt_lost: str
    bt_min: str
    display_threshold: str
    toggle_boxes: str
    toggle_classes: str
    toggle_scores: str
    toggle_ids: str
    toggle_trail: str
    toggle_zones: str
    toggle_help: str
    source_time: str
    model_version: str
    observed_share: str
    observed_bucket: str
    observed_all: str
    observed_none: str
    observed_of: str
    progress_time: str
    run_choice: str
    diagnostic_time: str
    detections_heading: str
    predicted_heading: str
    detections_none: str
    frame_unread: str
    zone_heading: str
    zone_help: str
    zone_name: str
    zone_x: str
    zone_y: str
    zone_view: str
    zone_save: str
    zone_view_required: str
    zone_size_unknown: str
    zone_saved: str
    zone_export_size: str
    zone_coordinate_file: str
    zone_coordinate_preview: str
    zone_three_vertices: str
    invalid_view_id: str
    thresholds_unset: str
    export_no_run: str
    frame_list_not_built: str
    quality_empty: str
    quality_run: str
    quality_compare: str
    quality_run_a: str
    quality_run_b: str
    quality_compare_same: str
    quality_delta: str
    quality_visibility: str
    quality_errors: str
    quality_classes: str
    error_kind: str
    error_frame: str
    error_class: str
    error_split: str
    error_kind_labels: dict[str, str]
    send_train_only: str
    thresholds_differ: str
    events_unavailable: str
    event_indicator_labels: tuple[str, ...]
    events_v02_caption: str
    theme_label: str
    theme_light: str
    theme_dark: str
    theme_display_only: str
    theme_write_failed: str
    leads: dict[str, str]
    locale_label: str
    locale_caption: str
    locale_write_failed: str
    seconds_suffix: str
    process_lost: str
    partial_result: str
    events_unavailable_caption: str
    create_project_heading: str
    create_project_name: str
    create_project_name_hint: str
    create_project_button: str
    catalog_existing: str
    catalog_new: str
    catalog_id_label: str
    catalog_copy_label: str
    catalog_copy_none: str
    catalog_none: str
    catalog_required: str
    catalog_id_required: str
    class_row_incomplete: str
    project_name_required: str
    add_class_row: str
    class_name_column: str
    class_title_column: str
    class_box_column: str
    catalog_classes_heading: str
    project_write_failed: str
    rename_project_label: str
    rename_project_button: str
    delete_project_button: str
    delete_project_confirm: str
    delete_confirm_required: str
    project_manager_title: str
    project_manager_close: str
    project_manager_list: str
    home_title: str
    home_lead: str
    home_empty: str
    open_project: str
    back_to_projects: str
    home_current: str
    home_created_column: str
    home_name_column: str
    home_id_column: str
    home_catalog_column: str
    home_videos_column: str
    home_episodes_column: str
    home_datasets_column: str
    home_models_column: str
    label_page_title: str
    label_open_app: str
    label_loading: str
    label_back_to_data: str
    label_progress: str
    label_mark_reviewed: str
    label_mark_excluded: str
    label_mark_unreviewed: str
    label_unreviewed_hint: str
    label_visibility_clear: str
    label_visibility_partial: str
    label_visibility_unlabelable: str
    label_unlabelable_excludes: str
    label_truncated: str
    label_delete_box: str
    label_prev: str
    label_next: str
    label_annotator: str
    label_saved: str
    label_save_failed: str
    label_not_frozen: str
    label_frame_not_sampled: str
    label_fit_width_note: str
    label_save: str
    label_note: str
    label_box_note: str
    label_instance: str
    label_class: str
    label_visibility: str
    label_invalid_json: str
    label_invalid_review: str
    label_invalid_visibility: str
    label_invalid_box: str
    label_unknown_class: str
    label_unknown_video: str
    label_unknown_frame: str
    label_frame_unassigned: str
    label_image_missing: str
    label_annotator_invalid: str
    label_unsupported_crowd: str
    label_unsupported_media_type: str
    label_cross_site: str
    label_confirm_empty: str
    label_excluded_hint: str
    label_filter_status: str
    label_filter_all: str
    label_filter_unreviewed: str
    label_filter_reviewed: str
    label_filter_excluded: str
    label_filter_split: str
    label_filter_all_splits: str
    label_shortcuts_title: str
    label_shortcut_class: str
    label_shortcut_visibility: str
    label_shortcut_truncated: str
    label_shortcut_reviewed: str
    label_shortcut_excluded: str
    label_shortcut_unreviewed: str
    label_shortcut_nav: str
    label_shortcut_delete: str
    label_shortcut_help: str
    label_reviewed_at: str
    label_frame_index: str
    label_frames_heading: str
    label_review_group: str
    label_suggest_button: str
    label_suggest_model: str
    label_suggest_accept: str
    label_suggest_accept_all: str
    label_suggest_reject_all: str
    label_not_train_frame: str
    label_job_running: str
    label_bundle_missing: str
    label_suggest_dropped: str
    action_catalog_heading: str
    action_name_column: str
    action_title_column: str
    action_seed: str
    action_name_help: str
    action_add: str
    action_delete: str
    action_save: str
    action_edit_title: str
    action_set_start: str
    action_set_end: str
    action_select: str
    action_overlap: str
    action_crosses_block: str
    action_outside_block: str
    action_invalid_interval: str
    action_stale: str
    action_save_failed: str
    action_blocks_unfrozen: str
    action_coverage_missing: str
    action_coverage_hint: str
    action_table_start: str
    action_table_end: str
    action_table_duration: str
    action_table_action: str
    action_table_role: str
    action_shortcuts: str
    action_snap_note: str
    action_reload: str
    action_draft: str
    action_no_video: str
    action_step_catalog: str
    action_step_labels: str
    action_step_train: str
    action_step_results: str
    action_step_catalog_text: str
    action_step_labels_text: str
    action_step_train_wait: str
    action_step_results_wait: str
    action_step_need_catalog: str
    action_step_train_text: str
    action_step_results_text: str
    action_runs_heading: str
    action_no_runs: str
    action_ready_heading: str
    action_ready_ok: str
    action_start: str
    action_cancel: str
    action_advanced: str
    action_epochs: str
    action_window_s: str
    action_min_confidence: str
    action_min_duration: str
    action_models_heading: str
    action_no_model: str
    action_support_train: str
    action_support_validation: str
    action_classes_scored: str
    action_selection_unstable: str
    action_selection_stable: str
    action_infer_heading: str
    action_other_detector: str
    action_infer_start: str
    action_inferences_heading: str
    action_no_inference: str
    action_select_runs: str
    action_unknown_share: str
    action_export_csv: str
    action_compare_heading: str
    action_test_hidden: str
    action_predicted_heading: str
    action_predicted_empty: str
    action_evaluate_validation: str
    action_evaluate_final: str
    action_final_once: str
    action_eval_empty: str
    action_precision: str
    action_recall: str
    action_f1: str
    action_support_frames: str
    action_support_intervals: str
    action_accuracy: str
    action_macro_f1: str
    action_evaluated_frames: str
    action_unlabelled_frames: str
    action_unknown_rate: str
    action_mean_confidence: str
    action_progress: str
    action_no_chart: str
    action_epoch: str
    action_run_fps: str
    job_already_running: str
    label_suggest_none: str

    def __post_init__(self) -> None:
        _require_keys(self.leads, _PAGE_IDS, "leads")
        _require_keys(self.class_labels, _CLASS_IDS, "class_labels")
        _require_keys(self.job_status_labels, _JOB_KEYS, "job_status_labels")
        _require_keys(self.error_kind_labels, _ERROR_KIND_KEYS, "error_kind_labels")
        _require_keys(self.metric_labels, _METRIC_KEYS, "metric_labels")
        _require_keys(self.sample_plan_labels, _SAMPLE_PLAN_IDS, "sample_plan_labels")
        _require_keys(self.class_box_lines, _CLASS_IDS, "class_box_lines")
        if len(self.event_indicator_labels) != _EVENT_INDICATOR_COUNT:
            raise ValueError("event indicators must have 12 labels")
        for page_id, text in self.leads.items():
            if not text:
                raise ValueError(f"empty lead: {page_id}")

    def lead(self, page_id: str) -> str:
        try:
            return self.leads[page_id]
        except KeyError:
            raise ValueError(f"unknown page id: {page_id}") from None

    @property
    def page_titles(self) -> tuple[str, ...]:
        return (
            self.page_video,
            self.page_data,
            self.page_training,
            self.page_analysis,
            self.page_quality,
            self.page_actions,
            self.page_events,
        )

    @property
    def language_options(self) -> tuple[str, str]:
        return LANGUAGE_OPTIONS

    def not_train_frame_message(self, frame_id: str, split: str) -> str:
        raise NotImplementedError

    def frame_not_found_message(self, frame_id: str) -> str:
        raise NotImplementedError

    def frame_ambiguous_message(self, frame_id: str) -> str:
        raise NotImplementedError

    def eval_not_found_message(self, eval_id: str) -> str:
        raise NotImplementedError

    def sent_to_labeling_message(self, frame_id: str) -> str:
        raise NotImplementedError

    def episode_ambiguous_message(self, episode_id: str) -> str:
        raise NotImplementedError

    def hardware_caption(
        self,
        device: str,
        cuda_available: object,
        mps_available: object,
        ram_bytes: object,
        tools_found: object,
    ) -> str:
        raise NotImplementedError

    def hardware_unavailable(self, detail: str) -> str:
        raise NotImplementedError

    def weights_warning(self, cache_dir: object) -> str:
        raise NotImplementedError

    def allowed_directory_extras(self, *names: str) -> str:
        raise NotImplementedError

    def duplicate_video_message(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _RussianCatalog(Catalog):
    def not_train_frame_message(self, frame_id: str, split: str) -> str:
        """Train is the only split that may return to labeling."""
        return (
            f"Кадр {frame_id} относится к «{split}». "
            "В разметку для дообучения отправляются только кадры train. "
            "Кадры validation и test не становятся очередью обучения."
        )

    def frame_not_found_message(self, frame_id: str) -> str:
        return (
            f"Кадр {frame_id} не найден среди отобранных кадров и в файле сопоставления. "
            "Пакет разметки не создаётся."
        )

    def frame_ambiguous_message(self, frame_id: str) -> str:
        return (
            f"У кадра {frame_id} несколько разных меток времени. "
            "Пакет разметки не создаётся."
        )

    def eval_not_found_message(self, eval_id: str) -> str:
        shown = eval_id or "—"
        return f"Прогон оценки «{shown}» не найден. Разница метрик не выводится."

    def sent_to_labeling_message(self, frame_id: str) -> str:
        """One train frame accepted into the labeling queue."""
        return f"Кадр {frame_id} поставлен в очередь разметки."

    def episode_ambiguous_message(self, episode_id: str) -> str:
        return f"Эпизод «{episode_id}» повторяется в зафиксированном разбиении."

    def hardware_caption(
        self,
        device: str,
        cuda_available: object,
        mps_available: object,
        ram_bytes: object,
        tools_found: object,
    ) -> str:
        if (
            isinstance(ram_bytes, int)
            and not isinstance(ram_bytes, bool)
            and ram_bytes > 0
        ):
            ram_text = f"{ram_bytes} байт"
        else:
            ram_text = self.not_calculated
        cuda = "да" if cuda_available else "нет"
        mps = "да" if mps_available else "нет"
        tools = "найдены" if tools_found else "не найдены"
        return (
            f"{self.meta_device}: {device}. CUDA: {cuda}. MPS: {mps}. "
            f"Память: {ram_text}. ffprobe и ffmpeg: {tools}. "
            f"{self.hardware_probes}"
        )

    def hardware_unavailable(self, detail: str) -> str:
        return f"{self.meta_device}: {self.not_calculated}. {detail}"

    def weights_warning(self, cache_dir: object) -> str:
        return (
            "Официальные веса RF-DETR могут быть скачаны один раз в "
            f"{cache_dir}. Повторный запуск использует этот локальный кэш."
        )

    def allowed_directory_extras(self, *names: str) -> str:
        listed = ", ".join(f"«{name}»" for name in names)
        return f"{self.allowed_directory} Дополнительно: {listed}."

    def duplicate_video_message(self) -> str:
        return "Видео с таким идентификатором или путём уже зарегистрировано."


@dataclass(frozen=True, slots=True)
class _EnglishCatalog(Catalog):
    def not_train_frame_message(self, frame_id: str, split: str) -> str:
        return (
            f'Frame {frame_id} belongs to "{split}". '
            "Only train frames are sent to labeling for further training. "
            "Validation and test frames do not become a training queue."
        )

    def frame_not_found_message(self, frame_id: str) -> str:
        return (
            f"Frame {frame_id} was not found among the sampled frames or in the match file. "
            "The labeling batch is not created."
        )

    def frame_ambiguous_message(self, frame_id: str) -> str:
        return (
            f"Frame {frame_id} has several different timestamps. "
            "The labeling batch is not created."
        )

    def eval_not_found_message(self, eval_id: str) -> str:
        shown = eval_id or "\u2014"
        return f'Evaluation run "{shown}" was not found. The metric difference is not shown.'

    def sent_to_labeling_message(self, frame_id: str) -> str:
        return f"Frame {frame_id} was queued for labeling."

    def episode_ambiguous_message(self, episode_id: str) -> str:
        return f'The frozen split repeats episode "{episode_id}".'

    def hardware_caption(
        self,
        device: str,
        cuda_available: object,
        mps_available: object,
        ram_bytes: object,
        tools_found: object,
    ) -> str:
        if (
            isinstance(ram_bytes, int)
            and not isinstance(ram_bytes, bool)
            and ram_bytes > 0
        ):
            ram_text = f"{ram_bytes} bytes"
        else:
            ram_text = self.not_calculated
        cuda = "yes" if cuda_available else "no"
        mps = "yes" if mps_available else "no"
        tools = "found" if tools_found else "not found"
        return (
            f"{self.meta_device}: {device}. CUDA: {cuda}. MPS: {mps}. "
            f"Memory: {ram_text}. ffprobe and ffmpeg: {tools}. "
            f"{self.hardware_probes}"
        )

    def hardware_unavailable(self, detail: str) -> str:
        return f"{self.meta_device}: {self.not_calculated}. {detail}"

    def weights_warning(self, cache_dir: object) -> str:
        return (
            "Official RF-DETR weights may be downloaded once into "
            f"{cache_dir}. A later run uses this local cache."
        )

    def allowed_directory_extras(self, *names: str) -> str:
        listed = ", ".join(f'"{name}"' for name in names)
        return (
            f"{self.allowed_directory} "
            f"Additional directories are also allowed: {listed}."
        )

    def duplicate_video_message(self) -> str:
        return "A video with this identifier or path is already registered."


RU = _RussianCatalog(
    not_calculated="не рассчитано",
    not_evaluated="not_evaluated",
    not_evaluated_caption="not evaluated",
    page_quality="Качество",
    send_to_labeling="Отправить в разметку",
    target_caption="Ориентир пилота. Это не измеренный результат.",
    errors_unavailable="Список ошибок недоступен: нет файла сопоставления. Рамки не подставляются.",
    errors_from_matches="Ложные срабатывания и пропуски из файла сопоставления оценки.",
    metrics_broken="metrics.json повреждён.",
    metrics_object="metrics.json должен быть объектом.",
    match_broken="Файл сопоставления повреждён. Ошибки не подставляются.",
    sample_broken="Файл отбора кадров повреждён. Пакет разметки не создаётся.",
    invalid_frame="Идентификатор кадра должен иметь вид «video_id:source_frame_index».",
    invalid_batch="Недопустимый идентификатор пакета.",
    class_labels={
        "excavator_body": "корпус экскаватора",
        "bucket": "ковш",
        "haul_truck": "самосвал",
        "truck_bed": "кузов",
    },
    metric_labels={
        "ap50": "AP50",
        "ap50_95": "AP50:95",
        "precision": "точность",
        "recall": "полнота",
        "recall_clear": "полнота clear",
        "recall_partial": "полнота partial",
        "reviewed_frame_count": "проверенные кадры",
        "evaluated_frame_count": "оценённые кадры",
        "gt_object_count": "объекты GT",
        "gt_count": "объекты GT",
        "excluded_label_fraction": "доля исключённых по labelability",
        "decode_ms": "декодирование, мс",
        "inference_ms": "инференс, мс",
        "encode_ms": "кодирование, мс",
        "score_floor": "нижний порог оценки",
        "operating_threshold": "рабочий порог",
        "iou": "IoU",
        "max_detections": "максимум детекций",
        "idf1": "IDF1",
        "hota": "HOTA",
    },
    episode_overlap="В одной точке времени разные split.",
    product_title="Cadence",
    kicker_project="PROJECT",
    kicker_workflow="WORKFLOW",
    kicker_appearance="APPEARANCE",
    page_video="Видео",
    page_data="Данные и разметка",
    page_training="Обучение",
    page_analysis="Анализ",
    page_actions="Действия",
    page_events="События",
    project_label="Проект",
    no_project="нет проекта",
    project_help="Проект в локальном каталоге. Демонстрационный проект не создаётся.",
    add_local_video="Ролика нет. Выберите локальный MP4, MOV или MKV, который уже лежит на диске. Образец не встроен, и приложение не скачивает замену.",
    local_path_label="Локальный путь к видео",
    local_path_placeholder="Путь к файлу на этом компьютере",
    video_metadata_unknown="Длительность, частота кадров и размер файла: не рассчитано.",
    video_registered="Видео зарегистрировано.",
    allowed_directory="Разрешённый каталог — корень workspace проекта. Положите MP4, MOV или MKV туда. Ссылка на ролик не скачивается.",
    allowed_directory_no_project="Разрешённый каталог — корень workspace. Он доступен после создания проекта. Ссылка на ролик не скачивается.",
    path_help="Путь к файлу внутри разрешённого каталога. Большой файл добавляйте только так.",
    add_video_button="Добавить видео",
    load_video_button="Загрузить видео",
    add_another="Добавить другой файл",
    path_required="Укажите локальный путь к видео.",
    need_project="Сначала создайте проект.",
    video_added="Видео добавлено.",
    upload_label="Небольшой файл через браузер",
    upload_button="Добавить загруженный файл",
    upload_help="Файл копируется в каталог проекта. Ролик больше 200 МБ в плеере открывается коротким началом, анализ идёт по всему файлу.",
    upload_missing="Сначала выберите файл.",
    upload_empty="Файл пустой. Выберите целый MP4, MOV или MKV.",
    upload_too_big="Файл не удалось принять через браузер.",
    upload_bad_name="Недопустимое имя файла. Нужен MP4, MOV или MKV.",
    advanced="Дополнительно",
    source_url_label="Исходный URL",
    source_url_help="Сохраняется как происхождение файла. По ссылке ролик не скачивается.",
    rotation_label="Поворот",
    meta_duration="Длительность",
    meta_fps="Частота кадров",
    meta_fps_nominal="номинальная",
    meta_fps_average="средняя",
    meta_frame="Размер кадра",
    meta_codec="Кодек",
    meta_sha="SHA-256",
    meta_device="Устройство",
    hardware_probes="Загрузка модели, проверка вывода и шаг обучения: не запускались.",
    offset_label="Смещение начала",
    interval_start="Начало интервала, с",
    interval_end="Конец интервала, с",
    interval_help="Секунды от начала исходного видео. Интервал полуоткрытый: конец не входит.",
    interval_required="Укажите начало и конец интервала в секундах.",
    baseline_button="Запустить базовую модель",
    baseline_caption="Базовая модель, не обучена на наших классах",
    baseline_help="Ставит анализ базовой модели в очередь. Пока задание выполняется, второй запуск не стартует.",
    baseline_queued="Анализ поставлен в очередь.",
    probe_button="Проверить файл",
    refresh_runs="Обновить статус",
    runs_heading="Запуски",
    runs_empty="Запусков нет. Список рамок не строится.",
    runs_no_boxes="Список рамок не строится. Статус берётся из базы заданий.",
    video_choice="Ролик",
    playback_file="Плеер читает файл на диске. Кадры исходника в интерфейсе не перерисовываются.",
    playback_preview="Короткий preview начала ролика. Кадры исходника в интерфейсе не перерисовываются.",
    playback_failed="Плеер не открылся. Кадры исходника в интерфейсе не рисуются.",
    choose_file_button="Выбрать файл",
    choose_file_prompt="Выберите MP4, MOV или MKV",
    choose_failed="Не удалось открыть выбор файла.",
    job_status_labels={
        "queued": "в очереди",
        "running": "выполняется",
        "completed": "завершен",
        "failed": "ошибка",
        "cancel_requested": "запрошена отмена",
        "cancelled": "отменен",
    },
    data_empty="Импорта разметки нет. Проверенные, непроверенные и исключённые кадры: не рассчитано. Снимок датасета закрыт, пока разбиение не зафиксировано.",
    freeze_split="Зафиксировать разбиение",
    propose_split="Предложить разбиение",
    freeze_done="Разбиение зафиксировано.",
    freeze_note="Повторная фиксация этого видео не выполняется.",
    propose_done="Предложение записано. Разбиение ещё не зафиксировано.",
    propose_kept_frozen="Предложение записано. Зафиксированное разбиение не заменяется.",
    split_hash="Хеш разбиения",
    proposal_missing="Блоков ещё нет.",
    proposal_broken="Файл разбиения повреждён.",
    proposal_legacy="На диске осталось разбиение по заездам. Оно не является блоками и не стёрто. Поставьте блоки обучения, проверки и итоговой оценки.",
    proposal_heading="Блоки",
    frozen_heading="Зафиксированное разбиение",
    independence_label="Независимость",
    episodes_heading="Эпизоды",
    episodes_empty="Эпизодов нет. Поставьте разрез на шкале, затем предложите разбиение.",
    episode_id="Идентификатор эпизода",
    episode_group="Группа заезда",
    episode_label="Метка",
    episode_concurrent="Одновременные заезды",
    add_episode="Добавить эпизод",
    episode_added="Эпизод добавлен.",
    default_episode_label="заезд",
    cut_here="Разрезать",
    cut_hint="Клавиша S ставит разрез в текущей точке.",
    concurrent_pass="Одновременный заезд",
    remove_cut="Убрать разрез",
    save_episode_label="Сохранить метку",
    episodes_frozen_note="Разбиение уже зафиксировано.",
    frame_step_label="Шаг кадра",
    arrow_frame_hint="Стрелки сдвигают выбранную границу на один кадр.",
    nudge_hint="Shift и стрелки сдвигают выбранную границу на 1 с.",
    episodes_saved="Эпизоды записаны.",
    episodes_save_failed="Не удалось записать эпизоды.",
    guards_heading="Защитные интервалы",
    guards_empty="Защитных интервалов нет.",
    split_heading="Разбиение",
    sample_heading="Отбор кадров",
    sample_rule="План и число кадров — разные кнопки. Кадры validation и test не добавляются в train.",
    sample_plan="План отбора",
    sample_plan_button="Отобрать по плану",
    sample_split="Часть",
    sample_count="Число кадров",
    sample_count_button="Отобрать число кадров",
    sample_spacing="Интервал между кадрами, с",
    plan_counts_caption="Числа планов из конфигурации. Это нагрузка на разметку, не оценка точности.",
    contact_heading="Контактный лист",
    contact_empty="Отобранных JPEG нет. Контактный лист не строится. Рамки не подставляются.",
    cvat_heading="Разметка в CVAT",
    export_cvat="Экспорт для CVAT",
    import_coco="Импорт COCO",
    import_file="ZIP с разметкой",
    import_path_required="Укажите путь к ZIP с разметкой.",
    import_bad_name="Недопустимое имя файла. Имя не должно содержать «..».",
    batch_split="Часть пакета",
    include_predictions="Предразметка train",
    predictions_help=(
        "Только для части «обучение». Это подсказки модели, не ground truth "
        "и не готовая разметка. Проверка и итоговая оценка скачиваются без подсказок."
    ),
    step_label_title="Разметка",
    step_snapshot_title="Снимок",
    step_episodes_text=(
        "Эпизод — один заезд самосвала или ожидание между заездами. "
        "Секунды от начала ролика. Конец не входит."
    ),
    step_split_text=(
        "Поставьте блоки обучения, проверки и итоговой оценки. "
        "Секунда вне блоков ни к одной роли не относится. Фиксация один раз."
    ),
    step_sample_text=(
        "Здесь рамки не ставятся. Программа выбирает редкие кадры, "
        "чтобы не размечать сотни почти одинаковых."
    ),
    step_label_text="Рамки рисуются в CVAT, не на этой странице. Сначала скачайте пакет.",
    step_snapshot_text=(
        "Снимок — это датасет для страницы «Обучение». В него входят только "
        "проверенные кадры. Кнопка открывается, когда все ролики зафиксированы "
        "и во всём проекте есть хотя бы один проверенный кадр в обучении и в проверке."
    ),
    step_now="сейчас",
    step_done="готово",
    step_wait="ждёт",
    step_wait_episode="Сначала добавьте эпизод.",
    step_wait_frozen="Сначала зафиксируйте разбиение.",
    step_wait_frames="Сначала отберите кадры.",
    step_snapshot_open=(
        "Кнопка уже открыта. В снимок всё равно входят только проверенные кадры."
    ),
    step_snapshot_written="Снимок уже записан.",
    episode_group_help=(
        "Один и тот же код, если заезды идут одновременно и их нельзя "
        "разнести по разным частям."
    ),
    episode_label_help=(
        "Короткое имя для себя, например «погрузка». На классы детектора не влияет."
    ),
    freeze_once="После фиксации границы этого ролика не меняются.",
    split_name_train="обучение",
    split_name_validation="проверка",
    split_name_test="итоговая оценка",
    equipment_block_help=(
        "В блоке обучения должны быть все единицы и части оборудования, "
        "которые модель должна узнавать."
    ),
    action_block_help="В блоке обучения должны быть все действия.",
    block_guard_help=(
        "Между блоками разных ролей оставьте не менее 10 с без роли. "
        "Этот промежуток не отбирается ни в одну часть. "
        "Операция, пересекающая границу, целиком уходит в один блок."
    ),
    block_guard_gap=(
        "Промежуток между блоками {earlier} ({earlier_end} с) и {later} ({later_start} с) — "
        "{gap} с, меньше {minimum} с. Разбиение зафиксировано как есть."
    ),
    block_unassigned="не назначено",
    block_overlap="Блоки пересекаются. Одна секунда не относится к двум ролям.",
    add_block="Добавить блок",
    save_blocks="Сохранить блоки",
    block_start="Начало",
    block_end="Конец",
    block_role="Роль",
    blocks_saved="Блоки записаны.",
    blocks_save_failed="Не удалось сохранить блоки.",
    sample_plan_help=(
        "Диагностический — первая проверка импорта. Содержательный — основной набор. "
        "Расширенный — больший набор. Числа планов — нагрузка на разметку, не оценка точности."
    ),
    label_class_names=(
        "Классы каталога проекта. Имя — идентификатор. Название и рамка — как записаны."
    ),
    add_class="Добавить класс",
    class_added="Класс записан.",
    add_class_name="Имя класса",
    add_class_title="Название",
    add_class_box="Рамка",
    class_box_lines={
        "excavator_body": (
            "Кабина, поворотная платформа, противовес и видимая ходовая часть, "
            "без стрелы, рукояти и ковша."
        ),
        "bucket": (
            "Видимая металлическая часть ковша, включая зубья, "
            "без рукояти, падающей породы и тени."
        ),
        "haul_truck": "Вся видимая машина: кабина, шасси, колёса и кузов.",
        "truck_bed": (
            "Видимый грузовой кузов как конструкция, не только поверхность груза "
            "и не только открытый проём."
        ),
    },
    label_review_rule=(
        "На каждом кадре атрибут review_status=reviewed. Без него кадр не входит "
        "в обучение. Пустой проверенный кадр — объектов нет. "
        "Непроверенный кадр — разметки ещё нет."
    ),
    label_zip_back=(
        "Обратно загрузите ZIP, в котором есть исходный frames_manifest.csv, "
        "те же JPEG и COCO из CVAT (images, annotations, categories). "
        "Subset CVAT не задаёт часть."
    ),
    export_path_note="После экспорта сообщение на странице показывает путь к zip.",
    download_batch="Скачать пакет",
    contact_no_boxes="Отобранные кадры. Рамок на них нет.",
    snapshot_dataset="Снимок датасета",
    snapshot_locked="Снимок датасета закрыт, пока разбиение не зафиксировано.",
    snapshot_ready="Разбиение зафиксировано. Можно собрать снимок датасета.",
    snapshot_other=(
        "У ролика «{name}» разбиение ещё не зафиксировано. "
        "Снимок собирается по всему проекту."
    ),
    snapshot_no_train=(
        "Нет проверенных кадров в блоке обучения. "
        "Проверенных обучаемых: обучение {train}, проверка {validation}."
    ),
    snapshot_no_validation=(
        "Нет проверенных кадров в блоке проверки. "
        "Проверенных обучаемых: обучение {train}, проверка {validation}."
    ),
    snapshot_gate_scope=(
        "Порог снимка считается по всему проекту: нужны проверенные кадры "
        "в обучении и в проверке, а не в одном ролике."
    ),
    label_counts_this_video="Счётчики этого ролика",
    label_counts_project="Счётчики всего проекта",
    label_counts_snapshot="Проверка последнего снимка",
    review_heading="Проверка разметки",
    count_reviewed="Проверенные",
    count_unreviewed="Непроверенные",
    count_excluded="Исключённые",
    count_reviewed_empty="Проверенные пустые",
    guard_prefix="Защитный интервал:",
    file_prefix="Файл:",
    plan_diagnostic_prefix="Диагностический:",
    plan_content_prefix="Содержательный:",
    episode_boundary_error="У эпизода нет числовой границы.",
    column_episode="эпизод",
    column_start_s="начало, с",
    column_end_s="конец, с",
    column_label="метка",
    column_split_group="Группа заезда",
    column_split="Часть",
    sample_plan_labels={
        "diagnostic": "диагностический",
        "content": "содержательный",
        "expanded": "расширенный",
    },
    training_empty="Кривая loss не строится: в журнале нет строк. Оценка времени: не рассчитано.",
    train_dataset="Версия датасета",
    train_dataset_empty="нет снимка",
    train_variant="Вариант",
    train_profile="Профиль",
    train_device="Устройство",
    train_start="Старт",
    train_cancel="Отмена",
    train_status="Статус",
    train_epoch="Эпоха",
    train_checkpoint="Чекпоинт",
    train_eta="Оценка времени",
    train_jobs="Задания",
    train_jobs_empty="Заданий обучения нет.",
    train_queued="Обучение поставлено в очередь.",
    train_cancelled="Отмена запрошена.",
    train_resolution="Разрешение",
    train_epochs="Эпохи",
    train_batch="Размер батча",
    train_accumulation="Накопление градиента",
    train_lr="Скорость обучения",
    train_encoder_lr="Скорость обучения энкодера",
    train_workers="Процессы загрузки",
    train_seed="Зерно",
    smoke_help="Smoke — проверка, что очередь и worker соединены. Это не результат точности и не приёмка модели.",
    pilot_help="pilot-small — стартовый профиль Small. Открытие страницы его не запускает.",
    device_help="auto выбирает одно устройство: CUDA, затем MPS, затем CPU. Отказ устройства не перезапускает обучение на другом.",
    train_start_help="Ставит обучение в очередь отдельного процесса. Повторный запуск и обновление страницы второе задание не создают.",
    train_cancel_help="Останавливает текущее задание и снимает блокировку.",
    dataset_not_ready=(
        "Снимок не готов к обучению: нет проверенных кадров в блоке обучения "
        "или в блоке проверки. Обучение не запускается. Кривая loss не строится."
    ),
    snapshot_not_built=(
        "Снимок датасета ещё не собран. Обучение не запускается. "
        "Кривая loss не строится."
    ),
    train_log_broken="Журнал метрик не читается. Кривая loss не строится.",
    analysis_empty="Прогона нет. Время источника, версия модели и доля наблюдаемых кадров: не рассчитано. Список рамок не строится.",
    analysis_no_video="Ролика нет. Добавьте файл на странице «Видео». Ссылка на ролик не скачивается.",
    analysis_interval_unknown="Интервал источника: не рассчитано.",
    analysis_fps_label="Частота анализа, кадр/с",
    analysis_fps_help="Как часто детектор берёт кадр. По умолчанию 5. Это не частота исходного ролика и не частота закодированного MP4.",
    analysis_player_help="Длинный просмотр — один закодированный MP4. Точная диагностика — одно время источника, один кадр и таблица из detections.parquet. Плеер не рисует каждый исходный кадр заново.",
    analysis_start="Старт",
    analysis_start_help="Ставит анализ в очередь отдельного процесса. Повторный запуск и обновление страницы второе задание не создают. Веса в этот процесс не загружаются.",
    analysis_export="Собрать MP4",
    analysis_export_help="Кодирует уже посчитанный прогон в один MP4. Детектор для этого не запускается.",
    analysis_queued="Анализ поставлен в очередь.",
    export_queued="Кодирование поставлено в очередь.",
    player_waiting="MP4 ещё нет. Исходные кадры здесь не перерисовываются.",
    custom_needs_bundle="Своя модель требует bundle с weights.pth и классами выбранного каталога. Без bundle запуск не выполняется. Базовая модель не подставляется.",
    model_mode="Модель",
    mode_baseline="Базовая",
    mode_custom="Своя",
    bundle_label="Bundle",
    bundle_synthetic="синтетическая проводка",
    synthetic_card_use=(
        "Синтетический прогон проводки: разметка, снимок и задание доходят до bundle. "
        "Это не модель и не измерение точности."
    ),
    synthetic_card_role=(
        "Роль метрик: synthetic_wiring. "
        "mAP и AP не рассчитаны. Это не карточка модели."
    ),
    threshold_new_run="Изменение порога не переписывает завершённый прогон. Нужен новый запуск.",
    bt_high="Порог высокой уверенности ByteTrack",
    bt_activation="Порог активации трека",
    bt_lost="Буфер потери, с",
    bt_min="Подтверждение, наблюдения",
    display_threshold="Порог показа",
    toggle_boxes="Рамки",
    toggle_classes="Классы",
    toggle_scores="Оценки",
    toggle_ids="Идентификаторы",
    toggle_trail="Траектория",
    toggle_zones="Зоны",
    toggle_help="Переключатели входят в закодированный MP4, а не в таблицу детекций.",
    source_time="Время источника",
    model_version="Версия модели",
    observed_share="Доля наблюдаемых кадров",
    observed_bucket="ковш",
    observed_all="все классы",
    observed_none="нет наблюдаемых кадров из {total}",
    observed_of="{observed} из {total}",
    progress_time="Время источника в прогрессе",
    run_choice="Прогон",
    diagnostic_time="Время источника кадра",
    detections_heading="Детекции",
    predicted_heading="predicted",
    detections_none="На этом кадре в detections.parquet нет рамок.",
    frame_unread="Кадр по этому времени источника не прочитан. Таблица берётся из detections.parquet.",
    zone_heading="Зоны камеры",
    zone_help="Вершины — доли ширины и высоты кадра. В пиксели они переводятся только на превью и в MP4, по размеру этого кадра. Сторона кадра не назначается забоем.",
    zone_name="Зона",
    zone_x="x",
    zone_y="y",
    zone_view="Идентификатор ракурса",
    zone_save="Сохранить зоны",
    zone_view_required="Укажите идентификатор ракурса.",
    zone_size_unknown="Размер кадра не рассчитан. Превью зоны не строится.",
    zone_saved="Зоны записаны.",
    zone_export_size="Размер кадра не рассчитан. Зоны в MP4 не подставляются.",
    zone_coordinate_file="Координата зоны вне доли кадра. Файл не записан.",
    zone_coordinate_preview="Координата зоны вне доли кадра. Превью не строится.",
    zone_three_vertices="Зона не сохранена: нужны три вершины и ненулевая площадь.",
    invalid_view_id="Недопустимый идентификатор ракурса.",
    thresholds_unset="Пороги не заданы. Завершённый прогон не изменён.",
    export_no_run="Прогона нет. MP4 не кодируется.",
    frame_list_not_built="Список рамок не строится.",
    quality_empty="Файла оценки нет. Метрики качества: не рассчитано. Рамки не подписываются как результат модели.",
    quality_run="Прогон оценки",
    quality_compare="Сравнение прогонов",
    quality_run_a="Первый прогон",
    quality_run_b="Второй прогон",
    quality_compare_same="Нужны два разных прогона. Разница метрик не выводится.",
    quality_delta="Изменение по классам",
    quality_visibility="Видимость",
    quality_errors="Ошибки",
    quality_classes="Классы",
    error_kind="Тип",
    error_frame="Кадр",
    error_class="Класс",
    error_split="Разбиение",
    error_kind_labels={
        "false_positive": "ложное срабатывание",
        "false_negative": "пропуск",
    },
    send_train_only="В разметку для дообучения отправляются только кадры train. Кадры validation, test и без роли остаются недоступны.",
    thresholds_differ="Нижний порог оценки и рабочий порог — разные величины. Показаны значения выбранного прогона.",
    events_unavailable="Раздел «События» недоступен. События и показатели: не рассчитано.",
    event_indicator_labels=(
        "Подтверждённые ковши",
        "Оценочное число ковшей",
        "Ковши на заезд",
        "Ковши экскаватора",
        "Время присутствия самосвала",
        "Время обслуживания",
        "Окно разгрузок в самосвал",
        "Полное время загрузки",
        "Интервал между разгрузками",
        "Полный цикл ковша",
        "Среднее время на самосвал",
        "Доля неизвестности",
    ),
    events_v02_caption="Подтвердить, отклонить, исправить время, назначить заезд и добавить пропущенное событие можно будет с v0.2.",
    theme_label="Тема",
    theme_light="Light",
    theme_dark="Dark",
    theme_display_only="Меняет только оформление.",
    theme_write_failed="Не удалось записать тему, оформление этого сеанса сохранено.",
    leads={
        "video": "Ролик и список запусков.",
        "data": "Четыре шага: блоки разбиения, отбор кадров, разметка в CVAT, снимок для обучения.",
        "training": "Профиль обучения, статус и checkpoint. Кривая loss — только из журнала.",
        "analysis": "Закодированный результат и один диагностический кадр.",
        "quality": "Измеренные метрики детектора и ошибки. Нет оценки — числа не подставляются.",
        "actions": "Каталог действий и интервалы на шкале ролика.",
        "events": "Кандидаты событий и показатели. Сейчас раздел выключен.",
    },
    locale_label="Язык",
    locale_caption="Меняет только язык интерфейса, не данные, модель и измерения.",
    locale_write_failed="Не удалось записать язык, язык этого сеанса сохранён.",
    seconds_suffix="с",
    process_lost="процесс пропал",
    partial_result="частичный результат",
    events_unavailable_caption="События и показатели в версии 0.1 не рассчитываются.",
    create_project_heading="Новый проект",
    create_project_name="Имя проекта",
    create_project_name_hint="Латинские буквы, цифры, _ или -, с буквы. Это идентификатор. Отображаемое имя меняется ниже, у уже созданного проекта.",
    create_project_button="Создать проект",
    catalog_existing="Существующий каталог",
    catalog_new="Новый каталог",
    catalog_id_label="Идентификатор каталога",
    catalog_copy_label="Скопировать каталог",
    catalog_copy_none="не копировать",
    catalog_none="каталог не выбран",
    catalog_required="Выберите каталог или создайте новый. Проект без каталога не создаётся.",
    catalog_id_required="Укажите идентификатор нового каталога.",
    class_row_incomplete="В строке класса нужны имя, название и рамка.",
    project_name_required="Укажите имя проекта.",
    add_class_row="Добавить строку",
    class_name_column="имя",
    class_title_column="название",
    class_box_column="рамка",
    catalog_classes_heading="Классы каталога",
    project_write_failed="Не удалось записать проект, выбранный проект этого сеанса сохранён.",
    rename_project_label="Отображаемое имя",
    rename_project_button="Переименовать",
    delete_project_button="Удалить проект",
    delete_project_confirm="Подтверждаю удаление проекта и его файлов",
    delete_confirm_required="Отметьте подтверждение, чтобы удалить проект.",
    project_manager_title="Управление проектами",
    project_manager_close="Закрыть",
    project_manager_list="Проекты",
    home_title="Проекты",
    home_lead="Откройте проект, чтобы перейти к видео. Этот список не считает обучение и качество.",
    home_empty="Проектов пока нет. Создайте проект, чтобы открыть видео.",
    open_project="Открыть",
    back_to_projects="К проектам",
    home_current="Текущий",
    home_created_column="Создан, UTC",
    home_name_column="Имя",
    home_id_column="Идентификатор",
    home_catalog_column="Каталог",
    home_videos_column="Видео",
    home_episodes_column="Эпизоды",
    home_datasets_column="Датасеты",
    home_models_column="Модели",
    label_page_title="Разметка кадра",
    label_open_app="Разметить в приложении",
    label_loading="Разметка кадра ещё не загружена.",
    label_back_to_data="К данным и разметке",
    label_progress=(
        "{split}: проверено {reviewed} из {total}, "
        "не проверено {unreviewed}, исключено {excluded}."
    ),
    label_mark_reviewed="Проверено",
    label_mark_excluded="Исключить",
    label_mark_unreviewed="Не проверено",
    label_unreviewed_hint=(
        "Непроверенный кадр сохраняется и не считается отрицательным примером."
    ),
    label_visibility_clear="полностью видно",
    label_visibility_partial="частично видно",
    label_visibility_unlabelable="нельзя разметить",
    label_unlabelable_excludes=(
        "Видимость «нельзя разметить» переводит кадр в excluded. "
        "В обучение детектора он не входит."
    ),
    label_truncated="обрезано краем кадра",
    label_delete_box="Удалить рамку",
    label_prev="Предыдущий кадр",
    label_next="Следующий кадр",
    label_annotator="Разметчик",
    label_saved="Кадр сохранён.",
    label_save_failed="Не удалось сохранить кадр.",
    label_not_frozen="Сначала зафиксируйте блоки. Кадр не сохраняется.",
    label_frame_not_sampled="Этого кадра нет в отборе. Сначала отберите кадры.",
    label_fit_width_note=(
        "Кадр показан по ширине панели. "
        "Масштаб на экране не меняет координаты в пикселях изображения."
    ),
    label_save="Сохранить",
    label_note="Заметка кадра",
    label_box_note="Заметка рамки",
    label_instance="Метка экземпляра",
    label_class="Класс",
    label_visibility="Видимость",
    label_invalid_json="Нужны review_status и список boxes.",
    label_invalid_review="Статус проверки: unreviewed, reviewed или excluded.",
    label_invalid_visibility="Видимость: clear, partial или unlabelable.",
    label_invalid_box="Рамка вне изображения или с неположительным размером.",
    label_unknown_class=(
        "Имени класса нет в каталоге проекта. Ближайший класс не подставляется."
    ),
    label_unknown_video="Видео не найдено в этом проекте.",
    label_unknown_frame="Кадр не найден в отборе этого видео.",
    label_frame_unassigned="Кадр не попадает ни в один блок.",
    label_image_missing="Изображения этого кадра нет в отборе.",
    label_annotator_invalid="Укажите имя разметчика одной строкой.",
    label_unsupported_crowd=(
        "iscrowd, отличное от 0, не записывается и не считается окклюзией."
    ),
    label_unsupported_media_type="Тело запроса должно быть в формате JSON.",
    label_cross_site="Запрос с другого сайта не принимается.",
    label_confirm_empty="Кадр без объектов станет отрицательным примером",
    label_excluded_hint=(
        "Исключённый кадр сохраняется и не входит в обучение детектора."
    ),
    label_filter_status="Показ",
    label_filter_all="все",
    label_filter_unreviewed="непроверенные",
    label_filter_reviewed="проверенные",
    label_filter_excluded="исключённые",
    label_filter_split="Блок",
    label_filter_all_splits="все блоки",
    label_shortcuts_title="Клавиши",
    label_shortcut_class="класс выбранной рамки или следующей",
    label_shortcut_visibility="видимость по кругу",
    label_shortcut_truncated="обрезано краем кадра",
    label_shortcut_reviewed="сохранить как проверенный и открыть следующий кадр",
    label_shortcut_excluded="сохранить как исключённый и открыть следующий кадр",
    label_shortcut_unreviewed="сохранить как непроверенный, рамки остаются",
    label_shortcut_nav="предыдущий или следующий кадр",
    label_shortcut_delete="удалить выбранную рамку",
    label_shortcut_help="показать список клавиш",
    label_reviewed_at="Проверено в",
    label_frame_index="Кадр",
    label_frames_heading="Кадры отбора",
    label_review_group="Статус проверки",
    label_suggest_button="Предложить рамки (только train)",
    label_suggest_model="Модель",
    label_suggest_accept="Принять",
    label_suggest_accept_all="Принять все",
    label_suggest_reject_all="Отклонить все",
    label_suggest_none="Предложений рамок ещё нет. Принять или отклонить нечего.",
    label_not_train_frame=(
        "Предложения только для кадров train. "
        "Кадры validation и test их не получают."
    ),
    label_job_running=(
        "В этом рабочем каталоге уже выполняется задание. "
        "Предложения рамок не строятся."
    ),
    label_bundle_missing=(
        "Нет своей модели для предложений. "
        "Базовая и синтетическая не используются."
    ),
    label_suggest_dropped="Имён нет в каталоге: {count}. Они не стали рамками.",
    action_catalog_heading="Каталог действий",
    action_name_column="Имя",
    action_title_column="Подпись",
    action_seed="Заполнить примером",
    action_name_help=(
        "name — английский токен. title показывается как введено, без перевода."
    ),
    action_add="Добавить",
    action_delete="Удалить",
    action_save="Сохранить",
    action_edit_title="Сохранить подпись",
    action_set_start="Начало = текущее время",
    action_set_end="Конец = текущее время",
    action_select="Действие",
    action_overlap=(
        "Интервалы действий пересекаются. Одна секунда не относится к двум действиям."
    ),
    action_crosses_block="Интервал пересекает границу блока. Он не обрезается.",
    action_outside_block="Интервал не лежит ни в одном блоке роли. Он не обрезается.",
    action_invalid_interval="Интервал должен быть полуоткрытым: конец больше начала.",
    action_stale="Список интервалов устарел. Замена не записана.",
    action_save_failed="Интервалы не записаны.",
    action_blocks_unfrozen=(
        "Блоки роли ещё не зафиксированы. Интервалы принимаются по черновику "
        "и будут проверены по зафиксированному разбиению перед обучением."
    ),
    action_coverage_missing="нет в обучении или в проверке",
    action_coverage_hint=(
        "Каждое действие из примера нужно хотя бы раз в обучении и раз в проверке. "
        "swing_loaded и swing_empty — разные действия, разметьте оба поворота."
    ),
    action_table_start="Начало, с",
    action_table_end="Конец, с",
    action_table_duration="Длительность, с",
    action_table_action="Действие",
    action_table_role="Роль",
    action_shortcuts=(
        "Клавиши: i — начало, o — конец, Enter — добавить, Delete — удалить, "
        "1–9 — действие, запятая и точка — один кадр."
    ),
    action_snap_note=(
        "Шаг кадра только на экране: показанный край округляется к шагу частоты кадров. "
        "В файл пишутся секунды исходного видео, которые посчитал сервер, "
        "а не номер кадра / fps."
    ),
    action_reload="Перезагрузить",
    action_draft="черновик",
    action_no_video="В проекте нет видео.",
    action_step_catalog="Каталог",
    action_step_labels="Интервалы",
    action_step_train="Обучение",
    action_step_results="Результат",
    action_step_catalog_text="Добавьте действия. Имя — английский токен, подпись хранится как введена.",
    action_step_labels_text=(
        "Отметьте интервалы под плеером. Один момент — одно действие, внутри одного блока роли."
    ),
    action_step_train_wait="Обучение классификатора действий на этом экране ещё не подключено.",
    action_step_results_wait="Результаты и оценка появятся после обучения. Сейчас их нет.",
    action_step_need_catalog="Сначала заполните каталог действий.",
    action_step_train_text=(
        "Отметьте завершённые custom-прогоны одного детектора и одной частоты. "
        "Старт ждёт ответ сервера о готовности."
    ),
    action_step_results_text=(
        "Интервалы модели, сравнение с разметкой обучения и проверки, затем оценка. "
        "Блок итоговой оценки открывается только после записи файла."
    ),
    action_runs_heading="Прогоны анализа",
    action_no_runs="Нет завершённого custom-прогона анализа для этого видео.",
    action_ready_heading="Готовность",
    action_ready_ok="Выбранные прогоны можно отправить на обучение.",
    action_start="Старт",
    action_cancel="Отмена",
    action_advanced="Дополнительно",
    action_epochs="Эпохи",
    action_window_s="Окно, с",
    action_min_confidence="Мин. уверенность",
    action_min_duration="Мин. длительность, с",
    action_models_heading="Модели действий",
    action_no_model="Модели действий ещё нет.",
    action_support_train="Окна обучения",
    action_support_validation="Окна проверки",
    action_classes_scored="Классы в macro-F1",
    action_selection_unstable="Мало окон проверки: выбор чекпоинта ненадёжен.",
    action_selection_stable="Окон проверки достаточно для выбора чекпоинта.",
    action_infer_heading="Распознавание",
    action_other_detector="другой детектор",
    action_infer_start="Запустить",
    action_inferences_heading="Задания распознавания",
    action_no_inference="Распознавания ещё нет.",
    action_select_runs="Отметьте хотя бы один прогон.",
    action_unknown_share="Доля unknown",
    action_export_csv="Скачать CSV",
    action_compare_heading="Размеченные интервалы",
    action_test_hidden=(
        "интервалы блока итоговой оценки скрыты до финальной проверки"
    ),
    action_predicted_heading="Интервалы модели",
    action_predicted_empty="Интервалов модели ещё нет.",
    action_evaluate_validation="Оценить на блоке проверки",
    action_evaluate_final="Итоговая оценка (test)",
    action_final_once="Файл пишется один раз и больше не меняется.",
    action_eval_empty="Оценки на блоке проверки ещё нет.",
    action_precision="Точность",
    action_recall="Полнота",
    action_f1="F1",
    action_support_frames="Кадры",
    action_support_intervals="Интервалы",
    action_accuracy="Точность кадра",
    action_macro_f1="macro-F1",
    action_evaluated_frames="Кадров оценено",
    action_unlabelled_frames="Кадров без метки",
    action_unknown_rate="Доля unknown",
    action_mean_confidence="Средняя уверенность",
    action_progress="Ход обучения",
    action_no_chart="Кривая появится после первой эпохи.",
    action_epoch="Эпоха",
    action_run_fps="кадр/с",
    job_already_running=(
        "В этом рабочем каталоге уже выполняется задание. "
        "Новое обучение или анализ не запускается, "
        "пока текущее не завершится или не будет отменено."
    ),
)

EN = _EnglishCatalog(
    not_calculated="not calculated",
    not_evaluated="not_evaluated",
    not_evaluated_caption="not evaluated",
    page_quality="Quality",
    send_to_labeling="Send to labeling",
    target_caption="Pilot target. This is not a measured result.",
    errors_unavailable="The error list is unavailable: there is no match file. Boxes are not substituted.",
    errors_from_matches="False positives and false negatives from the evaluation match file.",
    metrics_broken="metrics.json is damaged.",
    metrics_object="metrics.json must be an object.",
    match_broken="The match file is damaged. Errors are not substituted.",
    sample_broken="The frame-sampling file is damaged. The labeling batch is not created.",
    invalid_frame='A frame identifier must have the form "video_id:source_frame_index".',
    invalid_batch="Invalid batch identifier.",
    class_labels={
        "excavator_body": "excavator body",
        "bucket": "bucket",
        "haul_truck": "haul truck",
        "truck_bed": "truck bed",
    },
    metric_labels={
        "ap50": "AP50",
        "ap50_95": "AP50:95",
        "precision": "precision",
        "recall": "recall",
        "recall_clear": "recall clear",
        "recall_partial": "recall partial",
        "reviewed_frame_count": "reviewed frames",
        "evaluated_frame_count": "evaluated frames",
        "gt_object_count": "GT objects",
        "gt_count": "GT objects",
        "excluded_label_fraction": "fraction excluded by labelability",
        "decode_ms": "decode, ms",
        "inference_ms": "inference, ms",
        "encode_ms": "encode, ms",
        "score_floor": "score floor",
        "operating_threshold": "operating threshold",
        "iou": "IoU",
        "max_detections": "maximum detections",
        "idf1": "IDF1",
        "hota": "HOTA",
    },
    episode_overlap="Different splits at one point in time.",
    product_title="Cadence",
    kicker_project="PROJECT",
    kicker_workflow="WORKFLOW",
    kicker_appearance="APPEARANCE",
    page_video="Video",
    page_data="Data and labeling",
    page_training="Training",
    page_analysis="Analysis",
    page_actions="Actions",
    page_events="Events",
    project_label="Project",
    no_project="no project",
    project_help="A project in the local directory. A demonstration project is not created.",
    add_local_video="There is no video. Choose a local MP4, MOV, or MKV that is already on disk. A sample is not included, and the application does not download a substitute.",
    local_path_label="Local path to the video",
    local_path_placeholder="Path to a file on this computer",
    video_metadata_unknown="Duration, frame rate, and file size: not calculated.",
    video_registered="Video registered.",
    allowed_directory="The allowed directory is the project workspace root. Put an MP4, MOV, or MKV there. A link to the video is not downloaded.",
    allowed_directory_no_project="The allowed directory is the workspace root. It is available after a project is created. A link to the video is not downloaded.",
    path_help="Path to a file inside the allowed directory. Add a large file only this way.",
    add_video_button="Add video",
    load_video_button="Load a video",
    add_another="Add another file",
    path_required="Enter a local path to the video.",
    need_project="Create a project first.",
    video_added="Video added.",
    upload_label="Small file via the browser",
    upload_button="Add the uploaded file",
    upload_help="The file is copied into the project directory. A clip over 200 MB opens as a short start in the player; analysis uses the whole file.",
    upload_missing="Select a file first.",
    upload_empty="The file is empty. Select a whole MP4, MOV, or MKV.",
    upload_too_big="The browser could not accept the file.",
    upload_bad_name="Invalid file name. An MP4, MOV, or MKV is required.",
    advanced="Advanced",
    source_url_label="Source URL",
    source_url_help="Stored as the file origin. The video is not downloaded from the link.",
    rotation_label="Rotation",
    meta_duration="Duration",
    meta_fps="Frame rate",
    meta_fps_nominal="nominal",
    meta_fps_average="average",
    meta_frame="Frame size",
    meta_codec="Codec",
    meta_sha="SHA-256",
    meta_device="Device",
    hardware_probes="Model load, output check, and training step: not run.",
    offset_label="Start offset",
    interval_start="Interval start, s",
    interval_end="Interval end, s",
    interval_help="Seconds from the start of the source video. The interval is half-open: the end is not included.",
    interval_required="Enter the interval start and end in seconds.",
    baseline_button="Run the baseline model",
    baseline_caption="Baseline model, not trained on our classes",
    baseline_help="Queues analysis with the baseline model. A second run does not start while a job is running.",
    baseline_queued="Analysis queued.",
    probe_button="Check the file",
    refresh_runs="Refresh status",
    runs_heading="Runs",
    runs_empty="There are no runs. The box list is not built.",
    runs_no_boxes="The box list is not built. Status comes from the job database.",
    video_choice="Video",
    playback_file="The player reads the file on disk. Source frames are not redrawn in the interface.",
    playback_preview="A short preview of the start of the video. Source frames are not redrawn in the interface.",
    playback_failed="The player did not open. Source frames are not drawn in the interface.",
    choose_file_button="Choose file",
    choose_file_prompt="Choose an MP4, MOV, or MKV",
    choose_failed="Could not open the file chooser.",
    job_status_labels={
        "queued": "Queued",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "cancel_requested": "Cancel requested",
        "cancelled": "Cancelled",
    },
    data_empty="There is no labeling import. Reviewed, unreviewed, and excluded frames: not calculated. The dataset snapshot stays closed until the split is frozen.",
    freeze_split="Freeze the split",
    propose_split="Propose a split",
    freeze_done="Split frozen.",
    freeze_note="This video is not frozen again.",
    propose_done="Proposal saved. The split is not frozen yet.",
    propose_kept_frozen="Proposal saved. The frozen split is not replaced.",
    split_hash="Split hash",
    proposal_missing="There are no blocks yet.",
    proposal_broken="The split file is damaged.",
    proposal_legacy="A haul-episode split is still on disk. It is not a block split, and it was not erased. Place the training, validation, and final-evaluation blocks.",
    proposal_heading="Blocks",
    frozen_heading="Frozen split",
    independence_label="Independence",
    episodes_heading="Episodes",
    episodes_empty="There are no episodes. Cut the timeline, then propose a split.",
    episode_id="Episode identifier",
    episode_group="Pass group",
    episode_label="Label",
    episode_concurrent="Concurrent passes",
    add_episode="Add episode",
    episode_added="Episode added.",
    default_episode_label="pass",
    cut_here="Cut",
    cut_hint="The S key cuts at the current time.",
    concurrent_pass="Concurrent pass",
    remove_cut="Remove the cut",
    save_episode_label="Save the label",
    episodes_frozen_note="The split is already frozen.",
    frame_step_label="Frame step",
    arrow_frame_hint="Arrow keys move the selected boundary by one frame.",
    nudge_hint="Shift and the arrow keys move the selected boundary by 1 s.",
    episodes_saved="Episodes saved.",
    episodes_save_failed="Could not save the episodes.",
    guards_heading="Guard intervals",
    guards_empty="There are no guard intervals.",
    split_heading="Split",
    sample_heading="Frame sampling",
    sample_rule="The plan and the frame count are different buttons. Validation and test frames are not added to train.",
    sample_plan="Sampling plan",
    sample_plan_button="Sample by plan",
    sample_split="Split",
    sample_count="Frame count",
    sample_count_button="Sample a frame count",
    sample_spacing="Spacing between frames, s",
    plan_counts_caption="Plan counts from the configuration. This is labeling load, not an accuracy estimate.",
    contact_heading="Contact sheet",
    contact_empty="There are no sampled JPEGs. The contact sheet is not built. Boxes are not substituted.",
    cvat_heading="Labeling in CVAT",
    export_cvat="Export for CVAT",
    import_coco="Import COCO",
    import_file="Labeled ZIP",
    import_path_required="Enter the path to the labeling ZIP.",
    import_bad_name='Invalid file name. The name must not contain "..".',
    batch_split="Batch split",
    include_predictions="Train pre-labeling",
    predictions_help=(
        "Only for the training split. These are model hints, not ground truth "
        "and not finished labels. The validation and final-evaluation splits download without hints."
    ),
    step_label_title="Labeling",
    step_snapshot_title="Snapshot",
    step_episodes_text=(
        "An episode is one haul-truck pass or the wait between passes. "
        "Seconds run from the start of the clip. The end is not included."
    ),
    step_split_text=(
        "Place training, validation, and final-evaluation blocks. "
        "A second outside the blocks has no role. Freeze once."
    ),
    step_sample_text=(
        "Boxes are not drawn here. The program picks sparse frames so you do not "
        "label hundreds that are almost the same."
    ),
    step_label_text="Boxes are drawn in CVAT, not on this page. Download the batch first.",
    step_snapshot_text=(
        "The snapshot is the dataset for the Training page. It includes only "
        "reviewed frames. The button opens when every video is frozen and the "
        "whole project has at least one reviewed frame in training and in validation."
    ),
    step_now="now",
    step_done="done",
    step_wait="waiting",
    step_wait_episode="Add an episode first.",
    step_wait_frozen="Freeze the split first.",
    step_wait_frames="Sample frames first.",
    step_snapshot_open=(
        "The button can already be used. The snapshot still contains only reviewed frames."
    ),
    step_snapshot_written="The snapshot is already written.",
    episode_group_help=(
        "Use the same code when passes happen together and cannot be "
        "placed in different parts."
    ),
    episode_label_help=(
        'A short name for yourself, for example "loading". It does not affect detector classes.'
    ),
    freeze_once="After you freeze, this clip's boundaries do not change.",
    split_name_train="training",
    split_name_validation="validation",
    split_name_test="final evaluation",
    equipment_block_help=(
        "The training block must contain every equipment unit and every part "
        "the model should recognize."
    ),
    action_block_help="The training block must contain every action.",
    block_guard_help=(
        "Leave at least 10 s without a role between blocks of different roles. "
        "That gap is sampled into no split. "
        "An operation that crosses a boundary goes entirely into one block."
    ),
    block_guard_gap=(
        "The gap between {earlier} ({earlier_end} s) and {later} ({later_start} s) is "
        "{gap} s, under {minimum} s. The split is frozen as is."
    ),
    block_unassigned="unassigned",
    block_overlap="Blocks overlap. One second does not belong to two roles.",
    add_block="Add a block",
    save_blocks="Save blocks",
    block_start="Start",
    block_end="End",
    block_role="Role",
    blocks_saved="Blocks saved.",
    blocks_save_failed="Could not save the blocks.",
    sample_plan_help=(
        "Diagnostic is the first import check. Content is the main set. "
        "Expanded is the larger set. Plan counts are labeling load, not an accuracy estimate."
    ),
    label_class_names=(
        "Classes from the project catalog. The name is the id. "
        "Title and box are shown as stored."
    ),
    add_class="Add class",
    class_added="Class added.",
    add_class_name="Class name",
    add_class_title="Title",
    add_class_box="Box",
    class_box_lines={
        "excavator_body": (
            "Cab, rotating platform, counterweight, and the visible undercarriage, "
            "without the boom, stick, or bucket."
        ),
        "bucket": (
            "The visible metal of the bucket, including the teeth, "
            "without the stick, falling rock, or shadow."
        ),
        "haul_truck": "The whole visible machine: cab, chassis, wheels, and bed.",
        "truck_bed": (
            "The visible cargo bed as a structure, not only the load surface "
            "and not only the open mouth."
        ),
    },
    label_review_rule=(
        "On every frame set review_status=reviewed. Without it the frame is not "
        "used for training. A reviewed frame with no objects is empty. "
        "An unreviewed frame has no labeling yet."
    ),
    label_zip_back=(
        "Upload a ZIP that contains the original frames_manifest.csv, the same JPEGs, "
        "and the CVAT COCO file (images, annotations, categories). "
        "A CVAT subset does not assign the split."
    ),
    export_path_note="After export, the notice on the page shows the zip path.",
    download_batch="Download the batch",
    contact_no_boxes="Sampled frames. There are no boxes on them.",
    snapshot_dataset="Dataset snapshot",
    snapshot_locked="The dataset snapshot stays closed until the split is frozen.",
    snapshot_ready="The split is frozen. The dataset snapshot can be built.",
    snapshot_other=(
        "The video \"{name}\" has no frozen split yet. "
        "The snapshot covers the whole project."
    ),
    snapshot_no_train=(
        "No reviewed frames in the training block. "
        "Reviewed trainable: training {train}, validation {validation}."
    ),
    snapshot_no_validation=(
        "No reviewed frames in the validation block. "
        "Reviewed trainable: training {train}, validation {validation}."
    ),
    snapshot_gate_scope=(
        "The snapshot gate is project-wide: reviewed frames are required "
        "in training and in validation, not in one video."
    ),
    label_counts_this_video="Counts for this video",
    label_counts_project="Counts for the whole project",
    label_counts_snapshot="Latest snapshot check",
    review_heading="Label review",
    count_reviewed="Reviewed",
    count_unreviewed="Unreviewed",
    count_excluded="Excluded",
    count_reviewed_empty="Reviewed empty",
    guard_prefix="Guard interval:",
    file_prefix="File:",
    plan_diagnostic_prefix="Diagnostic:",
    plan_content_prefix="Content:",
    episode_boundary_error="The episode has no numeric boundary.",
    column_episode="Episode",
    column_start_s="Start, s",
    column_end_s="End, s",
    column_label="Label",
    column_split_group="Pass group",
    column_split="Part",
    sample_plan_labels={
        "diagnostic": "Diagnostic",
        "content": "Content",
        "expanded": "Expanded",
    },
    training_empty="The loss curve is not built: the log has no rows. Time estimate: not calculated.",
    train_dataset="Dataset version",
    train_dataset_empty="no snapshot",
    train_variant="Variant",
    train_profile="Profile",
    train_device="Device",
    train_start="Start",
    train_cancel="Cancel",
    train_status="Status",
    train_epoch="Epoch",
    train_checkpoint="Checkpoint",
    train_eta="Time estimate",
    train_jobs="Jobs",
    train_jobs_empty="There are no training jobs.",
    train_queued="Training queued.",
    train_cancelled="Cancel requested.",
    train_resolution="Resolution",
    train_epochs="Epochs",
    train_batch="Batch size",
    train_accumulation="Gradient accumulation",
    train_lr="Learning rate",
    train_encoder_lr="Encoder learning rate",
    train_workers="Loader processes",
    train_seed="Seed",
    smoke_help="Smoke checks that the queue and the worker are connected. This is not an accuracy result and not model acceptance.",
    pilot_help="pilot-small is the starting Small profile. Opening the page does not start it.",
    device_help="auto selects one device: CUDA, then MPS, then CPU. A device failure does not restart training on another.",
    train_start_help="Queues training in a separate process. Starting again and refreshing the page do not create a second job.",
    train_cancel_help="Stops the current job and releases the lock.",
    dataset_not_ready=(
        "The snapshot is not ready for training: there are no reviewed frames "
        "in the training block or in the validation block. Training does not start. "
        "The loss curve is not built."
    ),
    snapshot_not_built=(
        "The dataset snapshot has not been built yet. Training does not start. "
        "The loss curve is not built."
    ),
    train_log_broken="The metrics log cannot be read. The loss curve is not built.",
    analysis_empty="There is no run. Source time, model version, and the share of observed frames: not calculated. The box list is not built.",
    analysis_no_video="There is no video. Add a file on the Video page. A link to the video is not downloaded.",
    analysis_interval_unknown="Source interval: not calculated.",
    analysis_fps_label="Analysis rate, frame/s",
    analysis_fps_help="How often the detector takes a frame. The default is 5. This is not the source video rate and not the encoded MP4 rate.",
    analysis_player_help="A long view is one encoded MP4. Precise diagnosis is one source time, one frame, and the table from detections.parquet. The player does not redraw every source frame.",
    analysis_start="Start",
    analysis_start_help="Queues analysis in a separate process. Starting again and refreshing the page do not create a second job. Weights are not loaded into this process.",
    analysis_export="Build MP4",
    analysis_export_help="Encodes an already computed run into one MP4. The detector is not run for this.",
    analysis_queued="Analysis queued.",
    export_queued="Encoding queued.",
    player_waiting="There is no MP4 yet. Source frames are not redrawn here.",
    custom_needs_bundle="A custom model requires a bundle with weights.pth and the classes of the selected catalog. Without a bundle the run does not start. The baseline model is not substituted.",
    model_mode="Model",
    mode_baseline="Baseline",
    mode_custom="Custom",
    bundle_label="Bundle",
    bundle_synthetic="synthetic wiring",
    synthetic_card_use=(
        "Synthetic wiring run: labels, the snapshot, and the job reach a bundle. "
        "This is not a model and not an accuracy measurement."
    ),
    synthetic_card_role=(
        "Metric role: synthetic_wiring. "
        "mAP and AP are not calculated. This is not a model card."
    ),
    threshold_new_run="Changing a threshold does not rewrite a finished run. A new run is required.",
    bt_high="ByteTrack high-confidence threshold",
    bt_activation="Track activation threshold",
    bt_lost="Lost buffer, s",
    bt_min="Confirmation, observations",
    display_threshold="Display threshold",
    toggle_boxes="Boxes",
    toggle_classes="Classes",
    toggle_scores="Scores",
    toggle_ids="Identifiers",
    toggle_trail="Trajectory",
    toggle_zones="Zones",
    toggle_help="Toggles go into the encoded MP4, not into the detections table.",
    source_time="Source time",
    model_version="Model version",
    observed_share="Share of observed frames",
    observed_bucket="bucket",
    observed_all="all classes",
    observed_none="no observed frames out of {total}",
    observed_of="{observed} of {total}",
    progress_time="Source time in progress",
    run_choice="Run",
    diagnostic_time="Frame source time",
    detections_heading="Detections",
    predicted_heading="predicted",
    detections_none="This frame has no boxes in detections.parquet.",
    frame_unread="The frame at this source time was not read. The table comes from detections.parquet.",
    zone_heading="Camera zones",
    zone_help="Vertices are fractions of the frame width and height. They become pixels only on the preview and in the MP4, using this frame's size. A side of the frame is not assigned as the working face.",
    zone_name="Zone",
    zone_x="x",
    zone_y="y",
    zone_view="View identifier",
    zone_save="Save zones",
    zone_view_required="Enter the view identifier.",
    zone_size_unknown="Frame size is not calculated. The zone preview is not built.",
    zone_saved="Zones saved.",
    zone_export_size="Frame size is not calculated. Zones are not substituted into the MP4.",
    zone_coordinate_file="A zone coordinate is outside the frame fraction. The file was not written.",
    zone_coordinate_preview="A zone coordinate is outside the frame fraction. The preview is not built.",
    zone_three_vertices="The zone was not saved: three vertices and a non-zero area are required.",
    invalid_view_id="Invalid view identifier.",
    thresholds_unset="Thresholds are not set. The finished run is unchanged.",
    export_no_run="There is no run. The MP4 is not encoded.",
    frame_list_not_built="The box list is not built.",
    quality_empty="There is no evaluation file. Quality metrics: not calculated. Boxes are not labeled as a model result.",
    quality_run="Evaluation run",
    quality_compare="Run comparison",
    quality_run_a="First run",
    quality_run_b="Second run",
    quality_compare_same="Two different runs are required. The metric difference is not shown.",
    quality_delta="Change by class",
    quality_visibility="Visibility",
    quality_errors="Errors",
    quality_classes="Classes",
    error_kind="Type",
    error_frame="Frame",
    error_class="Class",
    error_split="Split",
    error_kind_labels={
        "false_positive": "false positive",
        "false_negative": "false negative",
    },
    send_train_only="Only train frames are sent to labeling for further training. Validation, test, and frames with no role stay unavailable.",
    thresholds_differ="The score floor and the operating threshold are different quantities. The selected run's values are shown.",
    events_unavailable="The Events section is unavailable. Events and indicators: not calculated.",
    event_indicator_labels=(
        "Confirmed buckets",
        "Estimated bucket count",
        "Buckets per pass",
        "Excavator buckets",
        "Haul truck presence time",
        "Service time",
        "Dump window into the haul truck",
        "Total loading time",
        "Interval between dumps",
        "Full bucket cycle",
        "Average time per haul truck",
        "Share of unknown",
    ),
    events_v02_caption="Confirming, rejecting, correcting the time, assigning a pass, and adding a missed event will be possible in v0.2.",
    theme_label="Theme",
    theme_light="Light",
    theme_dark="Dark",
    theme_display_only="Changes only the appearance.",
    theme_write_failed="Could not save the theme. This session's appearance is kept.",
    leads={
        "video": "The clip and the list of runs.",
        "data": "Four steps: split blocks, frame sampling, labeling in CVAT, and a snapshot for training.",
        "training": "Training profile, status, and checkpoint. The loss curve comes only from the log.",
        "analysis": "An encoded result and one diagnostic frame.",
        "quality": "Measured detector metrics and errors. Without an evaluation, numbers are not substituted.",
        "actions": "The action catalog and intervals on the clip timeline.",
        "events": "Event candidates and indicators. This section is off for now.",
    },
    locale_label="Language",
    locale_caption="Changes only the interface language, not the data, the model, or the measurements.",
    locale_write_failed="Could not save the language. This session's language is kept.",
    seconds_suffix="s",
    process_lost="The process disappeared",
    partial_result="The result is partial",
    events_unavailable_caption="Events and indicators are not calculated in version 0.1.",
    create_project_heading="New project",
    create_project_name="Project name",
    create_project_name_hint="Latin letters, digits, _ or -, starting with a letter. This is the id. The display name is changed on a project that already exists.",
    create_project_button="Create project",
    catalog_existing="Existing catalog",
    catalog_new="New catalog",
    catalog_id_label="Catalog id",
    catalog_copy_label="Copy a catalog",
    catalog_copy_none="do not copy",
    catalog_none="no catalog selected",
    catalog_required="Choose a catalog or create one. A project without a catalog is not created.",
    catalog_id_required="Enter an id for the new catalog.",
    class_row_incomplete="A class row needs name, title, and box.",
    project_name_required="Enter a project name.",
    add_class_row="Add a row",
    class_name_column="name",
    class_title_column="title",
    class_box_column="box",
    catalog_classes_heading="Catalog classes",
    project_write_failed="Could not save the project. This session's project is kept.",
    rename_project_label="Display name",
    rename_project_button="Rename",
    delete_project_button="Delete project",
    delete_project_confirm="I confirm deleting the project and its files",
    delete_confirm_required="Check the confirmation to delete the project.",
    project_manager_title="Manage projects",
    project_manager_close="Close",
    project_manager_list="Projects",
    home_title="Projects",
    home_lead="Open a project to reach the video. This list does not count training or quality.",
    home_empty="There are no projects yet. Create a project to open a video.",
    open_project="Open",
    back_to_projects="Projects",
    home_current="Current",
    home_created_column="Created, UTC",
    home_name_column="Name",
    home_id_column="Identifier",
    home_catalog_column="Catalog",
    home_videos_column="Videos",
    home_episodes_column="Episodes",
    home_datasets_column="Datasets",
    home_models_column="Models",
    label_page_title="Frame labeling",
    label_open_app="Label in the app",
    label_loading="Frame labeling is not loaded yet.",
    label_back_to_data="Back to data and labeling",
    label_progress=(
        "{split}: reviewed {reviewed} of {total}, "
        "unreviewed {unreviewed}, excluded {excluded}."
    ),
    label_mark_reviewed="Reviewed",
    label_mark_excluded="Exclude",
    label_mark_unreviewed="Unreviewed",
    label_unreviewed_hint="An unreviewed frame is stored and is not a negative example.",
    label_visibility_clear="fully visible",
    label_visibility_partial="partly visible",
    label_visibility_unlabelable="cannot be labeled",
    label_unlabelable_excludes=(
        "Visibility “cannot be labeled” stores the frame as excluded. "
        "It stays out of detector training."
    ),
    label_truncated="cut off by the frame edge",
    label_delete_box="Delete box",
    label_prev="Previous frame",
    label_next="Next frame",
    label_annotator="Annotator",
    label_saved="Frame saved.",
    label_save_failed="Could not save the frame.",
    label_not_frozen="Freeze the blocks first. The frame is not saved.",
    label_frame_not_sampled="This frame is not in the sample. Sample frames first.",
    label_fit_width_note=(
        "The frame is shown fitted to the panel width. "
        "Screen scale does not change image-pixel coordinates."
    ),
    label_save="Save",
    label_note="Frame note",
    label_box_note="Box note",
    label_instance="Instance id",
    label_class="Class",
    label_visibility="Visibility",
    label_invalid_json="The body needs review_status and a boxes list.",
    label_invalid_review="Review status must be unreviewed, reviewed, or excluded.",
    label_invalid_visibility="Visibility must be clear, partial, or unlabelable.",
    label_invalid_box="The box is outside the image or has a non-positive size.",
    label_unknown_class=(
        "That class name is not in the project catalog. "
        "A nearby class is not substituted."
    ),
    label_unknown_video="The video is not in this project.",
    label_unknown_frame="The frame is not in this video's sample.",
    label_frame_unassigned="The frame does not fall inside a block.",
    label_image_missing="This frame's image is not in the sample.",
    label_annotator_invalid="Enter the annotator name as one line.",
    label_unsupported_crowd=(
        "iscrowd other than 0 is not stored and is not treated as occlusion."
    ),
    label_unsupported_media_type="The request body must be JSON.",
    label_cross_site="A request from another site is not accepted.",
    label_confirm_empty="A frame with no objects becomes a negative example",
    label_excluded_hint="An excluded frame is stored and stays out of detector training.",
    label_filter_status="Show",
    label_filter_all="all",
    label_filter_unreviewed="unreviewed",
    label_filter_reviewed="reviewed",
    label_filter_excluded="excluded",
    label_filter_split="Block",
    label_filter_all_splits="all blocks",
    label_shortcuts_title="Keys",
    label_shortcut_class="class of the selected box, or of the next box",
    label_shortcut_visibility="cycle visibility",
    label_shortcut_truncated="cut off by the frame edge",
    label_shortcut_reviewed="save as reviewed and open the next frame",
    label_shortcut_excluded="save as excluded and open the next frame",
    label_shortcut_unreviewed="save as unreviewed; the boxes stay",
    label_shortcut_nav="previous or next frame",
    label_shortcut_delete="delete the selected box",
    label_shortcut_help="show the key list",
    label_reviewed_at="Reviewed at",
    label_frame_index="Frame",
    label_frames_heading="Sampled frames",
    label_review_group="Review status",
    label_suggest_button="Suggest boxes (train only)",
    label_suggest_model="Model",
    label_suggest_accept="Accept",
    label_suggest_accept_all="Accept all",
    label_suggest_reject_all="Reject all",
    label_suggest_none="There are no box suggestions yet. There is nothing to accept or reject.",
    label_not_train_frame=(
        "Suggestions are only for train frames. "
        "Validation and test frames do not receive them."
    ),
    label_job_running=(
        "A job is already running in this workspace. "
        "Box suggestions are not built."
    ),
    label_bundle_missing=(
        "There is no custom model for suggestions. "
        "Baseline and synthetic bundles are not used."
    ),
    label_suggest_dropped=(
        "Names missing from the catalog: {count}. They were not turned into boxes."
    ),
    action_catalog_heading="Action catalog",
    action_name_column="Name",
    action_title_column="Title",
    action_seed="Fill with the example",
    action_name_help="name is an English token. title is shown as stored, without translation.",
    action_add="Add",
    action_delete="Delete",
    action_save="Save",
    action_edit_title="Save title",
    action_set_start="Start = current time",
    action_set_end="End = current time",
    action_select="Action",
    action_overlap="Action intervals overlap. One second does not belong to two actions.",
    action_crosses_block="The interval crosses a block boundary. It is not clipped.",
    action_outside_block="The interval is not inside a role block. It is not clipped.",
    action_invalid_interval="The interval must be half-open: the end is after the start.",
    action_stale="The interval list is stale. The replacement was not saved.",
    action_save_failed="The intervals were not saved.",
    action_blocks_unfrozen=(
        "Role blocks are not frozen. Intervals are accepted against the draft "
        "and will be checked against the frozen split before training."
    ),
    action_coverage_missing="missing from training or validation",
    action_coverage_hint=(
        "Every seeded action should appear at least once in training and once in validation. "
        "swing_loaded and swing_empty are different actions; label both swings."
    ),
    action_table_start="Start, s",
    action_table_end="End, s",
    action_table_duration="Duration, s",
    action_table_action="Action",
    action_table_role="Role",
    action_shortcuts=(
        "Keys: i sets the start, o sets the end, Enter adds, Delete removes, "
        "1–9 picks an action, comma and period step one frame."
    ),
    action_snap_note=(
        "The frame step is only on screen: the shown edge rounds to the frame-rate step. "
        "The file stores the source seconds the server computed, not a frame number / fps."
    ),
    action_reload="Reload",
    action_draft="draft",
    action_no_video="This project has no video.",
    action_step_catalog="Catalog",
    action_step_labels="Intervals",
    action_step_train="Training",
    action_step_results="Results",
    action_step_catalog_text="Add actions. The name is an English token. The title is stored as entered.",
    action_step_labels_text=(
        "Mark intervals under the player. One moment is one action, inside one role block."
    ),
    action_step_train_wait="Action classifier training is not connected on this screen yet.",
    action_step_results_wait="Results and evaluation appear after training. There are none now.",
    action_step_need_catalog="Fill in the action catalog first.",
    action_step_train_text=(
        "Check completed custom runs of one detector and one rate. "
        "Start waits for the server readiness answer."
    ),
    action_step_results_text=(
        "Model intervals, a comparison with train and validation labels, then evaluation. "
        "The final-evaluation block opens only after its file is written."
    ),
    action_runs_heading="Analysis runs",
    action_no_runs="There is no completed custom analysis run for this video.",
    action_ready_heading="Readiness",
    action_ready_ok="The selected runs can be sent to training.",
    action_start="Start",
    action_cancel="Cancel",
    action_advanced="Advanced",
    action_epochs="Epochs",
    action_window_s="Window, s",
    action_min_confidence="Min confidence",
    action_min_duration="Min duration, s",
    action_models_heading="Action models",
    action_no_model="There is no action model yet.",
    action_support_train="Train windows",
    action_support_validation="Validation windows",
    action_classes_scored="Classes in macro-F1",
    action_selection_unstable="Too few validation windows: checkpoint selection is unreliable.",
    action_selection_stable="Enough validation windows to select a checkpoint.",
    action_infer_heading="Recognition",
    action_other_detector="other detector",
    action_infer_start="Run",
    action_inferences_heading="Recognition jobs",
    action_no_inference="There is no recognition yet.",
    action_select_runs="Select at least one run.",
    action_unknown_share="unknown share",
    action_export_csv="Download CSV",
    action_compare_heading="Labelled intervals",
    action_test_hidden="final-evaluation block intervals stay hidden until the final check",
    action_predicted_heading="Model intervals",
    action_predicted_empty="There are no model intervals yet.",
    action_evaluate_validation="Evaluate on the validation block",
    action_evaluate_final="Final evaluation (test)",
    action_final_once="The file is written once and is not changed again.",
    action_eval_empty="There is no validation evaluation yet.",
    action_precision="Precision",
    action_recall="Recall",
    action_f1="F1",
    action_support_frames="Frames",
    action_support_intervals="Intervals",
    action_accuracy="Frame accuracy",
    action_macro_f1="macro-F1",
    action_evaluated_frames="Evaluated frames",
    action_unlabelled_frames="Unlabelled frames",
    action_unknown_rate="unknown rate",
    action_mean_confidence="Mean confidence",
    action_progress="Training progress",
    action_no_chart="The curve appears after the first epoch.",
    action_epoch="Epoch",
    action_run_fps="fps",
    job_already_running=(
        "A job is already running in this workspace. "
        "A new training or analysis run does not start "
        "until the current one finishes or is cancelled."
    ),
)


def catalog(locale: str | None = None) -> Catalog:
    if locale == "en":
        return EN
    return RU


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)


def _compare_value(ru_value: object, en_value: object, path: str) -> None:
    if isinstance(ru_value, str):
        if not isinstance(en_value, str):
            raise TypeError(f"{path}: English value is not a string")
        if _has_cyrillic(ru_value):
            if LANGUAGE_RUSSIAN in en_value:
                raise RuntimeError(
                    f"{path}: shared language endonym is an English catalog sentence"
                )
            if en_value == ru_value or _has_cyrillic(en_value):
                raise RuntimeError(
                    f"{path}: English must translate the Russian text and contain no Cyrillic"
                )
            return
        if en_value != ru_value:
            raise RuntimeError(f"{path}: Latin invariant changed")
        return
    if isinstance(ru_value, dict):
        if not isinstance(en_value, dict) or list(ru_value) != list(en_value):
            raise RuntimeError(f"{path}: keys differ")
        for key, item in ru_value.items():
            _compare_value(item, en_value[key], f"{path}.{key}")
        return
    if isinstance(ru_value, tuple):
        if not isinstance(en_value, tuple) or len(ru_value) != len(en_value):
            raise RuntimeError(f"{path}: tuple length differs")
        for index, item in enumerate(ru_value):
            _compare_value(item, en_value[index], f"{path}[{index}]")
        return
    raise RuntimeError(f"{path}: unsupported catalog value {type(ru_value).__name__}")


def _check_catalogs(ru: Catalog, en: Catalog) -> None:
    """Fail import when a Cyrillic Russian string is copied into English."""
    if type(ru) is type(en):
        raise RuntimeError("Russian and English catalogs must be different types")
    for field in fields(Catalog):
        _compare_value(getattr(ru, field.name), getattr(en, field.name), field.name)
    for name in _MESSAGE_NAMES:
        ru_method = getattr(_RussianCatalog, name)
        en_method = getattr(_EnglishCatalog, name)
        if ru_method is getattr(Catalog, name) or en_method is getattr(Catalog, name):
            raise RuntimeError(f"{name} is missing on a catalog")
        if ru_method is en_method:
            raise RuntimeError(f"{name} must not be one shared format string")
        for args in _MESSAGE_SAMPLES[name]:
            _compare_value(ru_method(ru, *args), en_method(en, *args), f"{name}{args}")
    for book in (ru, en):
        try:
            book.lead("missing")
        except ValueError:
            pass
        else:
            raise RuntimeError("unknown page id did not fail")
        for page_id in _PAGE_IDS:
            if book.lead(page_id) != book.leads[page_id]:
                raise RuntimeError(f"lead({page_id}) drifted")
    if ru.language_options is not en.language_options:
        raise RuntimeError("language options are not shared")
    if ru.language_options != (LANGUAGE_RUSSIAN, LANGUAGE_ENGLISH):
        raise RuntimeError("language options drifted")
    if ru.not_calculated != "не рассчитано" or en.not_calculated != "not calculated":
        raise RuntimeError("missing-measurement text drifted")
    english_titles = (
        en.page_video,
        en.page_data,
        en.page_training,
        en.page_analysis,
        en.page_quality,
        en.page_actions,
        en.page_events,
    )
    if english_titles != (
        "Video",
        "Data and labeling",
        "Training",
        "Analysis",
        "Quality",
        "Actions",
        "Events",
    ):
        raise RuntimeError("English page titles drifted")
    if en.class_labels != {
        "excavator_body": "excavator body",
        "bucket": "bucket",
        "haul_truck": "haul truck",
        "truck_bed": "truck bed",
    }:
        raise RuntimeError("English class labels drifted")
    if en.job_status_labels != {
        "queued": "Queued",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "cancel_requested": "Cancel requested",
        "cancelled": "Cancelled",
    }:
        raise RuntimeError("English job-status labels drifted")
    if ru.locale_label != "Язык" or en.locale_label != "Language":
        raise RuntimeError("locale label drifted")
    if (
        ru.locale_caption
        != "Меняет только язык интерфейса, не данные, модель и измерения."
    ):
        raise RuntimeError("Russian locale caption drifted")
    if en.locale_caption != (
        "Changes only the interface language, not the data, the model, or the measurements."
    ):
        raise RuntimeError("English locale caption drifted")
    if ru.locale_write_failed != (
        "Не удалось записать язык, язык этого сеанса сохранён."
    ):
        raise RuntimeError("Russian locale write-failed drifted")
    if en.locale_write_failed != (
        "Could not save the language. This session's language is kept."
    ):
        raise RuntimeError("English locale write-failed drifted")
    shared = (
        ("product_title", "Cadence"),
        ("theme_light", "Light"),
        ("theme_dark", "Dark"),
        ("kicker_project", "PROJECT"),
        ("kicker_workflow", "WORKFLOW"),
        ("kicker_appearance", "APPEARANCE"),
        ("not_evaluated", "not_evaluated"),
        ("not_evaluated_caption", "not evaluated"),
        ("predicted_heading", "predicted"),
    )
    for name, expected in shared:
        if getattr(ru, name) != expected or getattr(en, name) != expected:
            raise RuntimeError(f"{name} must stay {expected!r}")


def _check_lookup() -> None:
    for locale in (None, "fr", "", "EN", "ru"):
        if catalog(locale) is not RU:
            raise RuntimeError(f"catalog({locale!r}) did not return Russian")
    if catalog("en") is not EN:
        raise RuntimeError("catalog('en') did not return English")


_ACTION_SEED_ORDER = (
    "digging",
    "swing_loaded",
    "dumping",
    "swing_empty",
    "idle",
)

action_seed_titles: dict[str, dict[str, str]] = {
    "ru": {
        "digging": "Копание",
        "swing_loaded": "Поворот с грузом",
        "dumping": "Выгрузка",
        "swing_empty": "Поворот порожняком",
        "idle": "Простой",
    },
    "en": {
        "digging": "Digging",
        "swing_loaded": "Swing loaded",
        "dumping": "Dumping",
        "swing_empty": "Swing empty",
        "idle": "Idle",
    },
}


def _check_action_seed_titles() -> None:
    ru_titles = action_seed_titles["ru"]
    en_titles = action_seed_titles["en"]
    if tuple(ru_titles) != _ACTION_SEED_ORDER or tuple(en_titles) != _ACTION_SEED_ORDER:
        raise RuntimeError("action seed titles drifted")
    for name in _ACTION_SEED_ORDER:
        if (
            not _has_cyrillic(ru_titles[name])
            or _has_cyrillic(en_titles[name])
            or ru_titles[name] == en_titles[name]
        ):
            raise RuntimeError(f"action seed title {name} locale drifted")


_check_catalogs(RU, EN)
_check_lookup()
_check_action_seed_titles()

NOT_CALCULATED = RU.not_calculated
NOT_EVALUATED = RU.not_evaluated
NOT_EVALUATED_CAPTION = RU.not_evaluated_caption
PAGE_QUALITY = RU.page_quality
SEND_TO_LABELING = RU.send_to_labeling
TARGET_CAPTION = RU.target_caption
ERRORS_UNAVAILABLE = RU.errors_unavailable
ERRORS_FROM_MATCHES = RU.errors_from_matches
METRICS_BROKEN = RU.metrics_broken
METRICS_OBJECT = RU.metrics_object
MATCH_BROKEN = RU.match_broken
SAMPLE_BROKEN = RU.sample_broken
INVALID_FRAME = RU.invalid_frame
INVALID_BATCH = RU.invalid_batch
CLASS_LABELS = RU.class_labels
METRIC_LABELS = RU.metric_labels
EPISODE_OVERLAP = RU.episode_overlap
PRODUCT_TITLE = RU.product_title
KICKER_PROJECT = RU.kicker_project
KICKER_WORKFLOW = RU.kicker_workflow
KICKER_APPEARANCE = RU.kicker_appearance
PAGE_VIDEO = RU.page_video
PAGE_DATA = RU.page_data
PAGE_TRAINING = RU.page_training
PAGE_ANALYSIS = RU.page_analysis
PAGE_ACTIONS = RU.page_actions
PAGE_EVENTS = RU.page_events
PROJECT_LABEL = RU.project_label
NO_PROJECT = RU.no_project
PROJECT_HELP = RU.project_help
ADD_LOCAL_VIDEO = RU.add_local_video
LOCAL_PATH_LABEL = RU.local_path_label
LOCAL_PATH_PLACEHOLDER = RU.local_path_placeholder
VIDEO_METADATA_UNKNOWN = RU.video_metadata_unknown
VIDEO_REGISTERED = RU.video_registered
ALLOWED_DIRECTORY = RU.allowed_directory
ALLOWED_DIRECTORY_NO_PROJECT = RU.allowed_directory_no_project
PATH_HELP = RU.path_help
ADD_VIDEO_BUTTON = RU.add_video_button
LOAD_VIDEO_BUTTON = RU.load_video_button
ADD_ANOTHER = RU.add_another
PATH_REQUIRED = RU.path_required
NEED_PROJECT = RU.need_project
VIDEO_ADDED = RU.video_added
UPLOAD_LABEL = RU.upload_label
UPLOAD_BUTTON = RU.upload_button
UPLOAD_HELP = RU.upload_help
UPLOAD_MISSING = RU.upload_missing
UPLOAD_EMPTY = RU.upload_empty
UPLOAD_TOO_BIG = RU.upload_too_big
UPLOAD_BAD_NAME = RU.upload_bad_name
ADVANCED = RU.advanced
SOURCE_URL_LABEL = RU.source_url_label
SOURCE_URL_HELP = RU.source_url_help
ROTATION_LABEL = RU.rotation_label
META_DURATION = RU.meta_duration
META_FPS = RU.meta_fps
META_FPS_NOMINAL = RU.meta_fps_nominal
META_FPS_AVERAGE = RU.meta_fps_average
META_FRAME = RU.meta_frame
META_CODEC = RU.meta_codec
META_SHA = RU.meta_sha
META_DEVICE = RU.meta_device
HARDWARE_PROBES = RU.hardware_probes
OFFSET_LABEL = RU.offset_label
INTERVAL_START = RU.interval_start
INTERVAL_END = RU.interval_end
INTERVAL_HELP = RU.interval_help
INTERVAL_REQUIRED = RU.interval_required
BASELINE_BUTTON = RU.baseline_button
BASELINE_CAPTION = RU.baseline_caption
BASELINE_HELP = RU.baseline_help
BASELINE_QUEUED = RU.baseline_queued
PROBE_BUTTON = RU.probe_button
REFRESH_RUNS = RU.refresh_runs
RUNS_HEADING = RU.runs_heading
RUNS_EMPTY = RU.runs_empty
RUNS_NO_BOXES = RU.runs_no_boxes
VIDEO_CHOICE = RU.video_choice
PLAYBACK_FILE = RU.playback_file
PLAYBACK_PREVIEW = RU.playback_preview
PLAYBACK_FAILED = RU.playback_failed
CHOOSE_FILE_BUTTON = RU.choose_file_button
CHOOSE_FILE_PROMPT = RU.choose_file_prompt
CHOOSE_FAILED = RU.choose_failed
JOB_STATUS_LABELS = RU.job_status_labels
DATA_EMPTY = RU.data_empty
FREEZE_SPLIT = RU.freeze_split
PROPOSE_SPLIT = RU.propose_split
FREEZE_DONE = RU.freeze_done
FREEZE_NOTE = RU.freeze_note
PROPOSE_DONE = RU.propose_done
PROPOSE_KEPT_FROZEN = RU.propose_kept_frozen
SPLIT_HASH = RU.split_hash
PROPOSAL_MISSING = RU.proposal_missing
PROPOSAL_BROKEN = RU.proposal_broken
PROPOSAL_HEADING = RU.proposal_heading
FROZEN_HEADING = RU.frozen_heading
INDEPENDENCE_LABEL = RU.independence_label
EPISODES_HEADING = RU.episodes_heading
EPISODES_EMPTY = RU.episodes_empty
EPISODE_ID = RU.episode_id
EPISODE_GROUP = RU.episode_group
EPISODE_LABEL = RU.episode_label
EPISODE_CONCURRENT = RU.episode_concurrent
ADD_EPISODE = RU.add_episode
EPISODE_ADDED = RU.episode_added
DEFAULT_EPISODE_LABEL = RU.default_episode_label
CUT_HERE = RU.cut_here
CUT_HINT = RU.cut_hint
CONCURRENT_PASS = RU.concurrent_pass
REMOVE_CUT = RU.remove_cut
SAVE_EPISODE_LABEL = RU.save_episode_label
EPISODES_FROZEN_NOTE = RU.episodes_frozen_note
FRAME_STEP_LABEL = RU.frame_step_label
ARROW_FRAME_HINT = RU.arrow_frame_hint
NUDGE_HINT = RU.nudge_hint
EPISODES_SAVED = RU.episodes_saved
EPISODES_SAVE_FAILED = RU.episodes_save_failed
GUARDS_HEADING = RU.guards_heading
GUARDS_EMPTY = RU.guards_empty
SPLIT_HEADING = RU.split_heading
SAMPLE_HEADING = RU.sample_heading
SAMPLE_RULE = RU.sample_rule
SAMPLE_PLAN = RU.sample_plan
SAMPLE_PLAN_BUTTON = RU.sample_plan_button
SAMPLE_SPLIT = RU.sample_split
SAMPLE_COUNT = RU.sample_count
SAMPLE_COUNT_BUTTON = RU.sample_count_button
SAMPLE_SPACING = RU.sample_spacing
PLAN_COUNTS_CAPTION = RU.plan_counts_caption
CONTACT_HEADING = RU.contact_heading
CONTACT_EMPTY = RU.contact_empty
CVAT_HEADING = RU.cvat_heading
EXPORT_CVAT = RU.export_cvat
IMPORT_COCO = RU.import_coco
IMPORT_FILE = RU.import_file
IMPORT_PATH_REQUIRED = RU.import_path_required
BATCH_SPLIT = RU.batch_split
INCLUDE_PREDICTIONS = RU.include_predictions
PREDICTIONS_HELP = RU.predictions_help
SNAPSHOT_DATASET = RU.snapshot_dataset
SNAPSHOT_LOCKED = RU.snapshot_locked
SNAPSHOT_READY = RU.snapshot_ready
REVIEW_HEADING = RU.review_heading
COUNT_REVIEWED = RU.count_reviewed
COUNT_UNREVIEWED = RU.count_unreviewed
COUNT_EXCLUDED = RU.count_excluded
COUNT_REVIEWED_EMPTY = RU.count_reviewed_empty
GUARD_PREFIX = RU.guard_prefix
FILE_PREFIX = RU.file_prefix
PLAN_DIAGNOSTIC_PREFIX = RU.plan_diagnostic_prefix
PLAN_CONTENT_PREFIX = RU.plan_content_prefix
EPISODE_BOUNDARY_ERROR = RU.episode_boundary_error
COLUMN_EPISODE = RU.column_episode
COLUMN_START_S = RU.column_start_s
COLUMN_END_S = RU.column_end_s
COLUMN_LABEL = RU.column_label
COLUMN_SPLIT_GROUP = RU.column_split_group
COLUMN_SPLIT = RU.column_split
SAMPLE_PLAN_LABELS = RU.sample_plan_labels
TRAINING_EMPTY = RU.training_empty
TRAIN_DATASET = RU.train_dataset
TRAIN_DATASET_EMPTY = RU.train_dataset_empty
TRAIN_VARIANT = RU.train_variant
TRAIN_PROFILE = RU.train_profile
TRAIN_DEVICE = RU.train_device
TRAIN_START = RU.train_start
TRAIN_CANCEL = RU.train_cancel
TRAIN_STATUS = RU.train_status
TRAIN_EPOCH = RU.train_epoch
TRAIN_CHECKPOINT = RU.train_checkpoint
TRAIN_ETA = RU.train_eta
TRAIN_JOBS = RU.train_jobs
TRAIN_JOBS_EMPTY = RU.train_jobs_empty
TRAIN_QUEUED = RU.train_queued
TRAIN_CANCELLED = RU.train_cancelled
TRAIN_RESOLUTION = RU.train_resolution
TRAIN_EPOCHS = RU.train_epochs
TRAIN_BATCH = RU.train_batch
TRAIN_ACCUMULATION = RU.train_accumulation
TRAIN_LR = RU.train_lr
TRAIN_ENCODER_LR = RU.train_encoder_lr
TRAIN_WORKERS = RU.train_workers
TRAIN_SEED = RU.train_seed
SMOKE_HELP = RU.smoke_help
PILOT_HELP = RU.pilot_help
DEVICE_HELP = RU.device_help
TRAIN_START_HELP = RU.train_start_help
TRAIN_CANCEL_HELP = RU.train_cancel_help
DATASET_NOT_READY = RU.dataset_not_ready
TRAIN_LOG_BROKEN = RU.train_log_broken
ANALYSIS_EMPTY = RU.analysis_empty
ANALYSIS_NO_VIDEO = RU.analysis_no_video
ANALYSIS_INTERVAL_UNKNOWN = RU.analysis_interval_unknown
ANALYSIS_FPS_LABEL = RU.analysis_fps_label
ANALYSIS_FPS_HELP = RU.analysis_fps_help
ANALYSIS_PLAYER_HELP = RU.analysis_player_help
ANALYSIS_START = RU.analysis_start
ANALYSIS_START_HELP = RU.analysis_start_help
ANALYSIS_EXPORT = RU.analysis_export
ANALYSIS_EXPORT_HELP = RU.analysis_export_help
ANALYSIS_QUEUED = RU.analysis_queued
EXPORT_QUEUED = RU.export_queued
PLAYER_WAITING = RU.player_waiting
CUSTOM_NEEDS_BUNDLE = RU.custom_needs_bundle
MODEL_MODE = RU.model_mode
MODE_BASELINE = RU.mode_baseline
MODE_CUSTOM = RU.mode_custom
BUNDLE_LABEL = RU.bundle_label
THRESHOLD_NEW_RUN = RU.threshold_new_run
BT_HIGH = RU.bt_high
BT_ACTIVATION = RU.bt_activation
BT_LOST = RU.bt_lost
BT_MIN = RU.bt_min
DISPLAY_THRESHOLD = RU.display_threshold
TOGGLE_BOXES = RU.toggle_boxes
TOGGLE_CLASSES = RU.toggle_classes
TOGGLE_SCORES = RU.toggle_scores
TOGGLE_IDS = RU.toggle_ids
TOGGLE_TRAIL = RU.toggle_trail
TOGGLE_ZONES = RU.toggle_zones
TOGGLE_HELP = RU.toggle_help
SOURCE_TIME = RU.source_time
MODEL_VERSION = RU.model_version
OBSERVED_SHARE = RU.observed_share
OBSERVED_BUCKET = RU.observed_bucket
OBSERVED_ALL = RU.observed_all
OBSERVED_NONE = RU.observed_none
OBSERVED_OF = RU.observed_of
PROGRESS_TIME = RU.progress_time
RUN_CHOICE = RU.run_choice
DIAGNOSTIC_TIME = RU.diagnostic_time
DETECTIONS_HEADING = RU.detections_heading
PREDICTED_HEADING = RU.predicted_heading
DETECTIONS_NONE = RU.detections_none
FRAME_UNREAD = RU.frame_unread
ZONE_HEADING = RU.zone_heading
ZONE_HELP = RU.zone_help
ZONE_NAME = RU.zone_name
ZONE_X = RU.zone_x
ZONE_Y = RU.zone_y
ZONE_VIEW = RU.zone_view
ZONE_SAVE = RU.zone_save
ZONE_VIEW_REQUIRED = RU.zone_view_required
ZONE_SIZE_UNKNOWN = RU.zone_size_unknown
ZONE_SAVED = RU.zone_saved
ZONE_EXPORT_SIZE = RU.zone_export_size
ZONE_COORDINATE_FILE = RU.zone_coordinate_file
ZONE_COORDINATE_PREVIEW = RU.zone_coordinate_preview
ZONE_THREE_VERTICES = RU.zone_three_vertices
INVALID_VIEW_ID = RU.invalid_view_id
THRESHOLDS_UNSET = RU.thresholds_unset
EXPORT_NO_RUN = RU.export_no_run
FRAME_LIST_NOT_BUILT = RU.frame_list_not_built
QUALITY_EMPTY = RU.quality_empty
QUALITY_RUN = RU.quality_run
QUALITY_COMPARE = RU.quality_compare
QUALITY_RUN_A = RU.quality_run_a
QUALITY_RUN_B = RU.quality_run_b
QUALITY_COMPARE_SAME = RU.quality_compare_same
QUALITY_DELTA = RU.quality_delta
QUALITY_VISIBILITY = RU.quality_visibility
QUALITY_ERRORS = RU.quality_errors
QUALITY_CLASSES = RU.quality_classes
ERROR_KIND = RU.error_kind
ERROR_FRAME = RU.error_frame
ERROR_CLASS = RU.error_class
ERROR_SPLIT = RU.error_split
ERROR_KIND_LABELS = RU.error_kind_labels
SEND_TRAIN_ONLY = RU.send_train_only
THRESHOLDS_DIFFER = RU.thresholds_differ
EVENTS_UNAVAILABLE = RU.events_unavailable
EVENT_INDICATOR_LABELS = RU.event_indicator_labels
EVENTS_V02_CAPTION = RU.events_v02_caption
EVENTS_UNAVAILABLE_CAPTION = RU.events_unavailable_caption
SECONDS_SUFFIX = RU.seconds_suffix
PROCESS_LOST = RU.process_lost
PARTIAL_RESULT = RU.partial_result
THEME_LABEL = RU.theme_label
THEME_LIGHT = RU.theme_light
THEME_DARK = RU.theme_dark
THEME_DISPLAY_ONLY = RU.theme_display_only
THEME_WRITE_FAILED = RU.theme_write_failed
PROJECT_WRITE_FAILED = RU.project_write_failed
LABEL_PAGE_TITLE = RU.label_page_title
LABEL_OPEN_APP = RU.label_open_app
LABEL_LOADING = RU.label_loading
LABEL_BACK_TO_DATA = RU.label_back_to_data

PAGE_TITLES: tuple[str, ...] = RU.page_titles

PAGE_DESCRIPTIONS: dict[str, str] = {
    PAGE_VIDEO: catalog("ru").lead("video"),
    PAGE_DATA: catalog("ru").lead("data"),
    PAGE_TRAINING: catalog("ru").lead("training"),
    PAGE_ANALYSIS: catalog("ru").lead("analysis"),
    PAGE_QUALITY: catalog("ru").lead("quality"),
    PAGE_ACTIONS: catalog("ru").lead("actions"),
    PAGE_EVENTS: catalog("ru").lead("events"),
}


def not_train_frame_message(frame_id: str, split: str) -> str:
    """Train is the only split that may return to labeling."""
    return catalog("ru").not_train_frame_message(frame_id, split)


def frame_not_found_message(frame_id: str) -> str:
    return catalog("ru").frame_not_found_message(frame_id)


def frame_ambiguous_message(frame_id: str) -> str:
    return catalog("ru").frame_ambiguous_message(frame_id)


def eval_not_found_message(eval_id: str) -> str:
    return catalog("ru").eval_not_found_message(eval_id)


def sent_to_labeling_message(frame_id: str) -> str:
    """One train frame accepted into the labeling queue."""
    return catalog("ru").sent_to_labeling_message(frame_id)


def episode_ambiguous_message(episode_id: str) -> str:
    return catalog("ru").episode_ambiguous_message(episode_id)
