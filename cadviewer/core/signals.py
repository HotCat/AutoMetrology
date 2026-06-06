"""
Global signal bus for decoupled inter-component communication.

All cross-module events flow through this singleton to avoid
tight coupling between UI, rendering, and model layers.
"""

from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    # ── Existing: feature/view signals ──
    feature_selected = Signal(str)       # feature_id
    feature_deselected = Signal()
    features_loaded = Signal(int)         # count
    view_fit_all = Signal()
    view_fit_feature = Signal(str)        # feature_id
    highlight_feature = Signal(str)       # feature_id
    unhighlight_all = Signal()
    property_update = Signal(dict)        # property dict

    # ── Registration group signals ──
    group_created = Signal(str)                  # group_id
    group_deleted = Signal(str)                  # group_id
    group_renamed = Signal(str)                  # group_id
    group_contents_changed = Signal(str)         # group_id
    groups_cleared = Signal()

    # ── Image and registration signals ──
    image_loaded = Signal(str)                   # file path
    registration_completed = Signal(dict)        # result dict
    registration_failed = Signal(str)            # error message

    # ── Correspondence signals ──
    correspondence_updated = Signal()

    # ── Teach pose signals ──
    teach_point_added = Signal(dict)              # {"phase": str, "world": [x,y], "pixel": [x,y]}
    teach_mode_started = Signal()
    teach_mode_completed = Signal(dict)           # {"cad_points": [...], "img_points": [...]}
    teach_mode_cancelled = Signal()

    # ── Measurement signals ──
    queries_evaluated = Signal(int)              # count of evaluated queries

    # ── DWG conversion signals ──
    dwg_conversion_started = Signal(str)         # dwg_path
    dwg_conversion_progress = Signal(str, int)   # stage_name, percent
    dwg_conversion_completed = Signal(dict)      # ConversionResult as dict
    dwg_conversion_failed = Signal(str)          # error message
    dwg_conversion_cancelled = Signal()

    # ── DXF validation signals ──
    dxf_validation_completed = Signal(dict)      # ValidationReport as dict
    dxf_validation_warning = Signal(str)         # warning message

    # ── ODA configuration signals ──
    oda_path_changed = Signal(str)               # new executable path


bus = SignalBus()
