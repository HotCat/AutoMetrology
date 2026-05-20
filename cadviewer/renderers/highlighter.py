"""
FeatureHighlighter — manages visual highlighting of selected features.

Handles:
  - Setting highlight color (bright cyan/green) and thick line width
  - Managing per-feature display attributes
  - Restoring original appearance when deselected
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_CYAN1, Quantity_NOC_GREEN
    from OCC.Core.AIS import AIS_Shape, AIS_InteractiveObject
    from OCC.Core.Prs3d import Prs3d_LineAspect
    from OCC.Core.Graphic3d import Graphic3d_NOM_PLASTIC
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

from ..models.feature import CADFeature


class FeatureHighlighter:
    """Manages visual highlighting of selected CAD features."""

    HIGHLIGHT_COLOR = Quantity_NOC_CYAN1
    HIGHLIGHT_LINE_WIDTH = 3.0
    DEFAULT_LINE_WIDTH = 1.0

    def __init__(self, occ_context=None) -> None:
        self._context = occ_context
        self._highlighted_id: Optional[str] = None
        self._original_attrs: Dict[str, dict] = {}  # feature_id → saved attributes

    def set_context(self, occ_context) -> None:
        self._context = occ_context

    def highlight(self, feature: CADFeature) -> None:
        """Apply highlight styling to a feature's AIS object."""
        if not OCC_AVAILABLE or not self._context or not feature.ais_object:
            return

        ais = feature.ais_object
        fid = feature.feature_id

        # Save original attributes if first highlight
        if fid not in self._original_attrs:
            self._original_attrs[fid] = {
                "color": ais.Color(),
                "width": ais.Attributes().LineAspect().Width(),
            }

        try:
            # Apply highlight style
            ais.SetColor(Quantity_Color(self.HIGHLIGHT_COLOR))
            drawer = ais.Attributes()
            drawer.SetLineAspect(Prs3d_LineAspect(
                Quantity_Color(self.HIGHLIGHT_COLOR),
                drawer.LineAspect().TypeOfLine(),
                self.HIGHLIGHT_LINE_WIDTH,
            ))
            self._context.Redisplay(ais, True)
        except Exception:
            pass

        self._highlighted_id = fid

    def unhighlight(self, feature: CADFeature) -> None:
        """Restore original appearance."""
        if not OCC_AVAILABLE or not self._context or not feature.ais_object:
            return

        ais = feature.ais_object
        fid = feature.feature_id
        saved = self._original_attrs.pop(fid, None)

        if saved:
            try:
                ais.SetColor(saved["color"])
                drawer = ais.Attributes()
                drawer.SetLineAspect(Prs3d_LineAspect(
                    saved["color"],
                    drawer.LineAspect().TypeOfLine(),
                    saved["width"],
                ))
                self._context.Redisplay(ais, True)
            except Exception:
                pass

        if self._highlighted_id == fid:
            self._highlighted_id = None

    def unhighlight_all(self) -> None:
        """Clear all highlights."""
        if self._context:
            try:
                self._context.ClearSelected(True)
            except Exception:
                pass
        self._highlighted_id = None

    @property
    def highlighted_id(self) -> Optional[str]:
        return self._highlighted_id
