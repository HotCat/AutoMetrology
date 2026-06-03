"""
Calibration module — residual distortion compensation for industrial metrology.

Pipeline:
  Raw Image → OpenCV Calibration → Undistorted Image
           → Residual Distortion Compensation Map
           → Measurement Pipeline (corrected edge points)

Classes:
  ResidualDistortionMap  — TPS-based correction map
  CalibrationManager     — orchestrates full calibration + residual sampling
  CalibrationReport      — error analysis and reporting
"""

from .residual_map import ResidualDistortionMap
from .calibration_manager import CalibrationManager
from .report import CalibrationReport
