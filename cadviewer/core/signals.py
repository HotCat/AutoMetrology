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

    # ── Measurement signals ──
    queries_evaluated = Signal(int)              # count of evaluated queries


bus = SignalBus()
