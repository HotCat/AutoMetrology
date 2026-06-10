"""
SQLite-backed production measurement log storage.

Each production run stores:
  - measured query rows
  - persisted camera image path
  - CAD file path/name
  - registration, affine, calibration, camera, and production profile params
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ..models.query import QueryInstruction, QueryResult, QueryType

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_loads(text: str, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _default_data_dir() -> Path:
    override = os.environ.get("CADVIEWER_PRODUCTION_LOG_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "cadviewer" / "production_logs"


class ProductionLogStore:
    """Persistent production measurement log repository."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        base_dir = _default_data_dir()
        self._db_path = Path(db_path) if db_path else base_dir / "production_logs.sqlite3"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def image_dir(self) -> Path:
        return self._db_path.parent / "images"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS production_records (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    created_date TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    query_count INTEGER NOT NULL,
                    ok_count INTEGER NOT NULL,
                    ng_count INTEGER NOT NULL,
                    no_measurement_count INTEGER NOT NULL,
                    error_count INTEGER NOT NULL,
                    cad_path TEXT NOT NULL,
                    cad_filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    source_image_path TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    pixel_size_mm REAL,
                    affine_json TEXT NOT NULL,
                    registration_json TEXT NOT NULL,
                    production_profile_json TEXT NOT NULL,
                    calibration_json TEXT NOT NULL,
                    camera_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS production_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    feature_id_1 TEXT NOT NULL,
                    feature_id_2 TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    value REAL,
                    nominal REAL,
                    deviation REAL,
                    tolerance_abs REAL,
                    unit TEXT NOT NULL,
                    status TEXT NOT NULL,
                    geometry_source TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES production_records(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_production_records_date "
                "ON production_records(created_date, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_production_results_record "
                "ON production_results(record_id, row_index)"
            )

    def save_image(
        self,
        record_id: str,
        source_path: str,
        image: Optional[np.ndarray],
        created_at: datetime,
    ) -> str:
        """Persist the camera image for a production record and return its path."""
        dest_dir = self.image_dir / created_at.strftime("%Y") / created_at.strftime("%m")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{record_id}.png"

        if image is not None and HAS_CV2:
            img = image
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(dest), img)
            return str(dest)

        if source_path and Path(source_path).exists():
            suffix = Path(source_path).suffix or ".png"
            dest = dest.with_suffix(suffix)
            shutil.copy2(source_path, dest)
            return str(dest)

        return source_path or ""

    def create_record(
        self,
        *,
        results: Iterable[QueryResult],
        query_text: str,
        cad_path: str,
        source_image_path: str,
        image: Optional[np.ndarray],
        pixel_size_mm: Optional[float],
        affine: Any,
        registration: dict,
        production_profile: dict,
        calibration: dict,
        camera: dict,
    ) -> str:
        created = datetime.now()
        record_id = str(uuid.uuid4())
        results_list = list(results)
        ok_count = sum(1 for r in results_list if r.status == "ok")
        ng_count = sum(1 for r in results_list if r.status == "ng")
        no_meas_count = sum(1 for r in results_list if r.status == "no_measurement")
        error_count = len(results_list) - ok_count - ng_count - no_meas_count
        overall = "ok" if len(results_list) > 0 and ok_count == len(results_list) else "ng"
        image_path = self.save_image(record_id, source_image_path, image, created)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO production_records (
                    id, created_at, created_date, overall_status, query_count,
                    ok_count, ng_count, no_measurement_count, error_count,
                    cad_path, cad_filename, image_path, image_filename,
                    source_image_path, query_text, pixel_size_mm, affine_json,
                    registration_json, production_profile_json, calibration_json,
                    camera_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    created.isoformat(timespec="seconds"),
                    created.date().isoformat(),
                    overall,
                    len(results_list),
                    ok_count,
                    ng_count,
                    no_meas_count,
                    error_count,
                    cad_path or "",
                    Path(cad_path).name if cad_path else "",
                    image_path,
                    Path(image_path).name if image_path else "",
                    source_image_path or "",
                    query_text or "",
                    pixel_size_mm,
                    _json_dumps(affine),
                    _json_dumps(registration or {}),
                    _json_dumps(production_profile or {}),
                    _json_dumps(calibration or {}),
                    _json_dumps(camera or {}),
                ),
            )
            for index, result in enumerate(results_list):
                conn.execute(
                    """
                    INSERT INTO production_results (
                        record_id, row_index, raw_text, query_type, feature_id_1,
                        feature_id_2, line_number, value, nominal, deviation,
                        tolerance_abs, unit, status, geometry_source,
                        error_message, audit_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._result_row(record_id, index, result),
                )
        return record_id

    def _result_row(self, record_id: str, index: int, result: QueryResult) -> tuple:
        inst = result.instruction
        return (
            record_id,
            index,
            inst.raw_text if inst else "",
            inst.query_type.name if inst else "",
            inst.feature_id_1 if inst else "",
            inst.feature_id_2 if inst else "",
            inst.line_number if inst else 0,
            result.value,
            result.nominal,
            result.deviation,
            result.tolerance_abs,
            result.unit,
            result.status,
            result.geometry_source,
            result.error_message,
            _json_dumps(result.feature_geometry_audit or {}),
        )

    def month_day_counts(self, year: int, month: int) -> dict[str, dict[str, int]]:
        start = f"{year:04d}-{month:02d}-01"
        if month == 12:
            end = f"{year + 1:04d}-01-01"
        else:
            end = f"{year:04d}-{month + 1:02d}-01"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_date,
                       COUNT(*) AS total,
                       SUM(CASE WHEN overall_status='ok' THEN 1 ELSE 0 END) AS ok,
                       SUM(CASE WHEN overall_status!='ok' THEN 1 ELSE 0 END) AS ng
                FROM production_records
                WHERE created_date >= ? AND created_date < ?
                GROUP BY created_date
                """,
                (start, end),
            ).fetchall()
        return {
            row["created_date"]: {
                "total": int(row["total"] or 0),
                "ok": int(row["ok"] or 0),
                "ng": int(row["ng"] or 0),
            }
            for row in rows
        }

    def records_for_day(self, date_text: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM production_records
                WHERE created_date=?
                ORDER BY created_at DESC
                """,
                (date_text,),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get_record(self, record_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM production_records WHERE id=?",
                (record_id,),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def get_results(self, record_id: str) -> list[QueryResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM production_results
                WHERE record_id=?
                ORDER BY row_index ASC
                """,
                (record_id,),
            ).fetchall()
        return [self._result_from_row(row) for row in rows]

    def _record_from_row(self, row: sqlite3.Row) -> dict:
        record = dict(row)
        record["affine"] = _json_loads(record.pop("affine_json", ""), None)
        record["registration"] = _json_loads(record.pop("registration_json", ""), {})
        record["production_profile"] = _json_loads(
            record.pop("production_profile_json", ""), {}
        )
        record["calibration"] = _json_loads(record.pop("calibration_json", ""), {})
        record["camera"] = _json_loads(record.pop("camera_json", ""), {})
        return record

    def _result_from_row(self, row: sqlite3.Row) -> QueryResult:
        query_type_name = row["query_type"]
        inst = None
        if query_type_name:
            query_type = QueryType[query_type_name]
            inst = QueryInstruction(
                raw_text=row["raw_text"],
                query_type=query_type,
                feature_id_1=row["feature_id_1"],
                feature_id_2=row["feature_id_2"],
                line_number=int(row["line_number"] or 0),
                tolerance_abs=row["tolerance_abs"],
            )
        return QueryResult(
            instruction=inst,
            value=row["value"],
            unit=row["unit"],
            status=row["status"],
            error_message=row["error_message"],
            nominal=row["nominal"],
            deviation=row["deviation"],
            tolerance_abs=row["tolerance_abs"],
            geometry_source=row["geometry_source"],
            feature_geometry_audit=_json_loads(row["audit_json"], {}),
        )
