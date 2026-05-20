"""
CAD Inspection Application
A metrology-oriented DXF feature inspection tool for machine vision alignment.

Architecture (MVC):
- Parser Layer: DXFImporter (parsers/dxf_importer.py)
- Model Layer: CADFeature, FeatureRepository (models/)
- Rendering Layer: OCCViewerWidget, FeatureHighlighter (renderers/)
- UI Layer: MainWindow, FeatureTreePanel, PropertyPanel (ui/)
"""

__version__ = "1.0.0"
__author__ = "CAD Inspection Tool"
