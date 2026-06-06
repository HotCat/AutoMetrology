"""
Calibration module — distortion compensation for industrial metrology.

Pipeline:
  Raw Image → OpenCV Calibration → Undistorted Image
           → Coordinate Correction (homography/affine models)
           → Measurement Pipeline

Classes:
  HomographyCalibrationModel  — projective correction from chessboard data
  AffineCalibrationModel      — linear correction from chessboard data
  CoordinateTransformer       — unified interface for correction models
  CalibrationValidator        — validate by measuring known grid distances
"""

from .coordinate_correction import (
    HomographyCalibrationModel,
    AffineCalibrationModel,
    CoordinateTransformer,
    CalibrationValidator,
    CalibrationMetadata,
)
from .calibration_manager import CalibrationManager
from .report import CalibrationReport
