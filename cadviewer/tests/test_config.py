"""Tests for resilient application config persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pathlib import PureWindowsPath
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cadviewer.core import config as config_mod
from cadviewer.core.config import CameraConfig, LightChannelConfig, LightControllerConfig
from cadviewer.ui.main_window import MainWindow
from cadviewer.ui.light_control_dialog import LightControlDialog
from cadviewer.ui.registration_panel import RegistrationPanel


class AppConfigPersistenceTest(unittest.TestCase):
    def test_load_recovers_from_backup_when_primary_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_path = root / "settings.json"
            backup_path = root / "settings.json.bak"
            good = {
                "pixel_size_mm": 0.025,
                "camera": {"exposure_us": 123, "gamma": 4, "contrast": 99,
                            "analog_gain": 2, "ae_enabled": False,
                            "reverse_x": False, "reverse_y": False},
            }
            main_path.write_text("{not valid json", encoding="utf-8")
            backup_path.write_text(json.dumps(good), encoding="utf-8")

            with patch.object(config_mod, "_CONFIG_DIR", root), \
                 patch.object(config_mod, "_CONFIG_FILE", main_path), \
                 patch.object(config_mod, "_CONFIG_BACKUP_FILE", backup_path):
                cfg = config_mod.AppConfig.load()

            self.assertAlmostEqual(cfg.pixel_size_mm, 0.025)
            self.assertEqual(cfg.camera.exposure_us, 123)

    def test_save_refuses_to_overwrite_when_last_load_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_path = root / "settings.json"
            main_path.write_text(json.dumps({"pixel_size_mm": 0.0433}), encoding="utf-8")
            original = main_path.read_text(encoding="utf-8")

            with patch.object(config_mod, "_CONFIG_DIR", root), \
                 patch.object(config_mod, "_CONFIG_FILE", main_path), \
                 patch.object(config_mod, "_CONFIG_BACKUP_FILE", root / "settings.json.bak"), \
                 patch.object(config_mod, "_LAST_LOAD_STATUS", "load_failed"):
                cfg = config_mod.AppConfig()
                cfg.pixel_size_mm = 0.01
                cfg.save()

            self.assertEqual(main_path.read_text(encoding="utf-8"), original)

    def test_save_writes_backup_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_path = root / "settings.json"
            backup_path = root / "settings.json.bak"

            with patch.object(config_mod, "_CONFIG_DIR", root), \
                 patch.object(config_mod, "_CONFIG_FILE", main_path), \
                 patch.object(config_mod, "_CONFIG_BACKUP_FILE", backup_path), \
                 patch.object(config_mod, "_LAST_LOAD_STATUS", "ok"):
                cfg = config_mod.AppConfig()
                cfg.pixel_size_mm = 0.033
                cfg.save()

            self.assertTrue(main_path.exists())
            self.assertTrue(backup_path.exists())
            saved = json.loads(main_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(saved["pixel_size_mm"], 0.033)

    def test_windows_config_and_data_dirs_use_appdata(self) -> None:
        with patch.object(config_mod.os, "name", "nt"), \
             patch.object(config_mod, "Path", PureWindowsPath), \
             patch.dict(config_mod.os.environ, {"APPDATA": r"C:\Users\Tester\AppData\Roaming"}, clear=False):
            self.assertEqual(
                config_mod.get_config_dir(),
                PureWindowsPath(r"C:\Users\Tester\AppData\Roaming") / "cadviewer",
            )
            self.assertEqual(
                config_mod.get_data_dir("production_logs"),
                PureWindowsPath(r"C:\Users\Tester\AppData\Roaming") / "cadviewer" / "production_logs",
            )

    def test_measurement_queries_update_active_profile_only(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._config = SimpleNamespace(
            measurement_queries="old global",
            line_fit_side_overrides={},
            active_production_profile="Product A",
            production_profiles=[
                {"name": "Product A", "measurement_queries": "old A"},
                {"name": "Product B", "measurement_queries": "old B"},
            ],
        )

        window._set_active_profile_measurement_queries("new A")

        self.assertEqual(window._config.measurement_queries, "new A")
        self.assertEqual(
            window._config.production_profiles[0]["measurement_queries"],
            "new A",
        )
        self.assertEqual(
            window._config.production_profiles[1]["measurement_queries"],
            "old B",
        )

    def test_line_band_overrides_update_active_profile_only(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._config = SimpleNamespace(
            line_fit_side_overrides={},
            active_production_profile="Product A",
            production_profiles=[
                {"name": "Product A", "line_fit_side_overrides": {"A1": "positive"}},
                {"name": "Product B", "line_fit_side_overrides": {"B1": "negative"}},
            ],
        )

        window._set_active_profile_line_fit_side_overrides({"A2": "negative"})

        self.assertEqual(window._config.line_fit_side_overrides, {"A2": "negative"})
        self.assertEqual(
            window._config.production_profiles[0]["line_fit_side_overrides"],
            {"A2": "negative"},
        )
        self.assertEqual(
            window._config.production_profiles[1]["line_fit_side_overrides"],
            {"B1": "negative"},
        )

    def test_query_settings_update_active_profile_only(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._config = SimpleNamespace(
            dual_light_orientation_guard_enabled=True,
            active_production_profile="Product A",
            production_profiles=[
                {"name": "Product A", "query_settings": {
                    "force_nearest_line_bias": False,
                    "dual_light_orientation_guard_enabled": True,
                    "line_fit_side_mode": "auto",
                }},
                {"name": "Product B", "query_settings": {
                    "force_nearest_line_bias": True,
                    "dual_light_orientation_guard_enabled": False,
                    "line_fit_side_mode": "negative",
                }},
            ],
        )

        window._set_active_profile_query_settings({
            "force_nearest_line_bias": True,
            "dual_light_orientation_guard_enabled": False,
            "line_fit_side_mode": "positive",
        })

        self.assertFalse(window._config.dual_light_orientation_guard_enabled)
        self.assertEqual(
            window._config.production_profiles[0]["query_settings"],
            {
                "force_nearest_line_bias": True,
                "dual_light_orientation_guard_enabled": False,
                "line_fit_side_mode": "positive",
            },
        )
        self.assertEqual(
            window._config.production_profiles[1]["query_settings"],
            {
                "force_nearest_line_bias": True,
                "dual_light_orientation_guard_enabled": False,
                "line_fit_side_mode": "negative",
            },
        )

    def test_profile_measurement_queries_falls_back_to_legacy_global(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._config = SimpleNamespace(
            measurement_queries="legacy",
            line_fit_side_overrides={"LEG": "positive"},
        )

        self.assertEqual(window._profile_measurement_queries({}), "legacy")
        self.assertEqual(
            window._profile_line_fit_side_overrides({}),
            {"LEG": "positive"},
        )

    def test_profile_query_settings_falls_back_to_legacy_guard(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._config = SimpleNamespace(
            dual_light_orientation_guard_enabled=False,
        )

        settings = window._profile_query_settings({})
        self.assertFalse(settings["dual_light_orientation_guard_enabled"])
        self.assertFalse(settings["force_nearest_line_bias"])
        self.assertEqual(settings["line_fit_side_mode"], "auto")

    def test_legacy_global_queries_migrate_only_to_active_profile(self) -> None:
        panel = RegistrationPanel.__new__(RegistrationPanel)
        panel._config = SimpleNamespace(
            measurement_queries="active query",
            line_fit_side_overrides={"ACTIVE": "positive"},
            dual_light_orientation_guard_enabled=False,
            light_controller=LightControllerConfig(
                device="/dev/ttyUSB9",
                baud=19200,
                timeout_s=0.5,
                backlight_settle_delay_ms=321,
                ring_light_settle_delay_ms=654,
                ring_ch1=LightChannelConfig(brightness=11, enabled=True),
                ring_ch2=LightChannelConfig(brightness=22, enabled=False),
                backlight_ch4=LightChannelConfig(brightness=44, enabled=True),
            ),
            active_production_profile="Product A",
            production_profiles=[
                {"name": "Product A"},
                {"name": "Product B"},
            ],
        )

        profiles = panel._ensure_production_profiles()

        self.assertEqual(profiles[0]["measurement_queries"], "active query")
        self.assertEqual(
            profiles[0]["line_fit_side_overrides"],
            {"ACTIVE": "positive"},
        )
        self.assertEqual(
            profiles[0]["query_settings"],
            {
                "force_nearest_line_bias": False,
                "dual_light_orientation_guard_enabled": False,
                "line_fit_side_mode": "auto",
            },
        )
        self.assertEqual(profiles[0]["light_controller"]["device"], "/dev/ttyUSB9")
        self.assertEqual(profiles[0]["light_controller"]["ring_ch1"]["brightness"], 11)
        self.assertEqual(profiles[1]["measurement_queries"], "")
        self.assertEqual(profiles[1]["line_fit_side_overrides"], {})
        self.assertEqual(
            profiles[1]["query_settings"],
            {
                "force_nearest_line_bias": False,
                "dual_light_orientation_guard_enabled": True,
                "line_fit_side_mode": "auto",
            },
        )
        self.assertEqual(profiles[1]["light_controller"]["ring_ch1"]["brightness"], 180)

    def test_profile_defaults_include_cad_block(self) -> None:
        panel = RegistrationPanel.__new__(RegistrationPanel)
        panel._config = SimpleNamespace(
            camera=CameraConfig(),
            measurement_queries="",
            line_fit_side_overrides={},
            dual_light_orientation_guard_enabled=True,
            light_controller=LightControllerConfig(),
        )

        profile = panel._default_production_profile()

        self.assertEqual(profile["version"], 3)
        self.assertEqual(profile["cad"], {
            "source_path": "",
            "stored_path": "",
            "filename": "",
            "source_type": "dxf",
        })

    def test_ensure_profiles_migrates_cad_block(self) -> None:
        panel = RegistrationPanel.__new__(RegistrationPanel)
        panel._config = SimpleNamespace(
            measurement_queries="active query",
            line_fit_side_overrides={},
            dual_light_orientation_guard_enabled=True,
            light_controller=LightControllerConfig(),
            active_production_profile="Product A",
            production_profiles=[
                {"name": "Product A"},
                {"name": "Product B"},
            ],
        )

        profiles = panel._ensure_production_profiles()

        self.assertIn("cad", profiles[0])
        self.assertIn("cad", profiles[1])
        self.assertEqual(profiles[0]["cad"]["stored_path"], "")
        self.assertEqual(profiles[1]["cad"]["stored_path"], "")

    def test_save_active_cad_profile_stores_per_profile_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.dxf"
            source.write_text("cad data", encoding="utf-8")
            cad_root = root / "cad-data"

            panel = RegistrationPanel.__new__(RegistrationPanel)
            panel._config = SimpleNamespace(
                active_production_profile="Product A",
                production_profiles=[{"name": "Product A"}],
                save=lambda: None,
            )
            panel._current_profile_name = lambda: "Product A"
            panel._find_production_profile = lambda name: panel._config.production_profiles[0]
            panel._snapshot_production_profile = lambda name: {"name": name}
            captured = {}
            panel._upsert_production_profile = lambda profile, silent=False: captured.update(profile)

            with patch.object(config_mod, "get_profile_data_dir", lambda profile_name, *parts: cad_root / profile_name / Path(*parts)):
                stored = panel.save_active_cad_profile(str(source), source_type="dxf", silent=True)

            self.assertIsNotNone(stored)
            stored_path = Path(stored)
            self.assertTrue(stored_path.exists())
            self.assertEqual(stored_path.read_text(encoding="utf-8"), "cad data")
            self.assertEqual(captured["cad"]["source_path"], str(source))
            self.assertEqual(captured["cad"]["stored_path"], str(stored_path))
            self.assertEqual(captured["cad"]["source_type"], "dxf")

    def test_profile_cad_restore_loads_stored_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cad_file = root / "stored.dxf"
            cad_file.write_text("cad data", encoding="utf-8")

            window = MainWindow.__new__(MainWindow)
            window._suppress_profile_cad_restore = False
            window._reg_panel = SimpleNamespace(
                profile_cad_path=lambda profile: str(cad_file)
            )
            window._last_dxf_path = str(root / "old.dxf")
            calls = []
            window._load_dxf = lambda path, **kwargs: calls.append((path, kwargs))

            window._restore_profile_cad({"name": "Product A"})

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], str(cad_file))
            self.assertEqual(calls[0][1]["update_active_profile_cad"], False)

    def test_light_control_dialog_saves_connection_only(self) -> None:
        dialog = LightControlDialog.__new__(LightControlDialog)
        saved = []
        dialog._config = SimpleNamespace(
            light_controller=LightControllerConfig(
                device="/dev/ttyUSB0",
                baud=9600,
                timeout_s=0.7,
                backlight_settle_delay_ms=200,
                ring_light_settle_delay_ms=200,
                ring_ch1=LightChannelConfig(brightness=11, enabled=True),
                ring_ch2=LightChannelConfig(brightness=22, enabled=False),
                backlight_ch4=LightChannelConfig(brightness=44, enabled=True),
            ),
            save=lambda: saved.append(True),
        )
        dialog._current_key = lambda: ("/dev/ttyUSB9", 115200, 1.2)
        dialog._backlight_delay_spin = SimpleNamespace(value=lambda: 321)
        dialog._ring_delay_spin = SimpleNamespace(value=lambda: 654)
        dialog.light_profile_changed = SimpleNamespace(emit=lambda: None)

        dialog._save_config()

        light = dialog._config.light_controller
        self.assertEqual(light.device, "/dev/ttyUSB9")
        self.assertEqual(light.baud, 115200)
        self.assertAlmostEqual(light.timeout_s, 1.2)
        self.assertEqual(light.backlight_settle_delay_ms, 321)
        self.assertEqual(light.ring_light_settle_delay_ms, 654)
        self.assertEqual(light.ring_ch1.brightness, 11)
        self.assertTrue(saved)

    def test_save_active_light_controller_profile_uses_current_config_state(self) -> None:
        panel = RegistrationPanel.__new__(RegistrationPanel)
        panel._config = SimpleNamespace(
            light_controller=LightControllerConfig(
                device="/dev/ttyUSB1",
                baud=19200,
                timeout_s=0.5,
                backlight_settle_delay_ms=111,
                ring_light_settle_delay_ms=222,
                ring_ch1=LightChannelConfig(brightness=9, enabled=True),
                ring_ch2=LightChannelConfig(brightness=19, enabled=False),
                backlight_ch4=LightChannelConfig(brightness=29, enabled=True),
            ),
            active_production_profile="Profile A",
            production_profiles=[
                {
                    "name": "Profile A",
                    "light_controller": {
                        "device": "/dev/ttyUSB9",
                        "baud": 115200,
                        "timeout_s": 1.2,
                        "backlight_settle_delay_ms": 321,
                        "ring_light_settle_delay_ms": 654,
                        "ring_ch1": {"brightness": 11, "enabled": False},
                        "ring_ch2": {"brightness": 22, "enabled": True},
                        "backlight_ch4": {"brightness": 44, "enabled": False},
                    },
                }
            ],
        )

        captured = {}
        panel._current_profile_name = lambda: "Profile A"
        panel._find_production_profile = lambda name: panel._config.production_profiles[0]
        panel._snapshot_production_profile = lambda name: {"name": name}
        panel._upsert_production_profile = lambda profile, silent=False: captured.update(profile)

        panel.save_active_light_controller_profile(silent=True)

        self.assertEqual(captured["light_controller"]["device"], "/dev/ttyUSB1")
        self.assertEqual(captured["light_controller"]["baud"], 19200)
        self.assertEqual(captured["light_controller"]["ring_ch1"]["brightness"], 9)
        self.assertNotIn("backlight_settle_delay_ms", captured["light_controller"])
        self.assertNotIn("ring_light_settle_delay_ms", captured["light_controller"])
        self.assertEqual(panel._config.light_controller.backlight_settle_delay_ms, 111)
        self.assertEqual(panel._config.light_controller.ring_light_settle_delay_ms, 222)

    def test_apply_light_controller_profile_keeps_global_settle_delays(self) -> None:
        panel = RegistrationPanel.__new__(RegistrationPanel)
        panel._config = SimpleNamespace(
            light_controller=LightControllerConfig(
                device="/dev/ttyUSB1",
                baud=19200,
                timeout_s=0.5,
                backlight_settle_delay_ms=111,
                ring_light_settle_delay_ms=222,
                ring_ch1=LightChannelConfig(brightness=9, enabled=True),
                ring_ch2=LightChannelConfig(brightness=19, enabled=False),
                backlight_ch4=LightChannelConfig(brightness=29, enabled=True),
            )
        )
        panel._live_window = None

        panel._apply_light_controller_profile({
            "device": "/dev/ttyUSB9",
            "baud": 115200,
            "timeout_s": 1.2,
            "backlight_settle_delay_ms": 321,
            "ring_light_settle_delay_ms": 654,
            "ring_ch1": {"brightness": 11, "enabled": False},
            "ring_ch2": {"brightness": 22, "enabled": True},
            "backlight_ch4": {"brightness": 44, "enabled": False},
        })

        light = panel._config.light_controller
        self.assertEqual(light.device, "/dev/ttyUSB9")
        self.assertEqual(light.baud, 115200)
        self.assertAlmostEqual(light.timeout_s, 1.2)
        self.assertEqual(light.backlight_settle_delay_ms, 111)
        self.assertEqual(light.ring_light_settle_delay_ms, 222)
        self.assertEqual(light.ring_ch1.brightness, 11)
        self.assertFalse(light.ring_ch1.enabled)


if __name__ == "__main__":
    unittest.main()
