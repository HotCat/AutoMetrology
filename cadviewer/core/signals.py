"""
Global signal bus for decoupled inter-component communication.

All cross-module events flow through this singleton to avoid
tight coupling between UI, rendering, and model layers.
"""

from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    feature_selected = Signal(str)       # feature_id
    feature_deselected = Signal()
    features_loaded = Signal(int)         # count
    view_fit_all = Signal()
    view_fit_feature = Signal(str)        # feature_id
    highlight_feature = Signal(str)       # feature_id
    unhighlight_all = Signal()
    property_update = Signal(dict)        # property dict


bus = SignalBus()
