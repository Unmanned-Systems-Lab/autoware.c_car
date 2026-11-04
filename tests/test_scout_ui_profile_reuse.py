"""Exercise the real UI launcher against an isolated HTTP server, without ROS."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
import urllib.request


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "src/launcher/autoware_launch/ui/start_ui_server.sh"


class ProfileReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scout-ui-profile-")
        self.map_dir = str((Path(self.temp.name) / "map with spaces").resolve())
        self.mode = "universe"
        self.allow = "0"
        self.map_file = "ndt_map.pcd"
        self.status_code = 200
        self.raw_json = None
        self.status = self.make_status()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    code, data = 200, b'{"status":"ok"}'
                elif self.path == "/autoware_status":
                    code = outer.status_code
                    data = outer.raw_json if outer.raw_json is not None else json.dumps(outer.status).encode()
                else:
                    code, data = 404, b"{}"
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def make_status(self):
        enabled = self.allow == "1"
        return {
            "localization_backend": self.mode,
            "ui_actions_enabled": enabled,
            "runtime_profile": {
                "localization_backend": self.mode,
                "runtime_map_dir": self.map_dir,
                "pointcloud_map_file": self.map_file,
                "ui_actions_enabled": enabled,
            },
        }

    def launch(self, runtime_map_dir=None):
        environment = dict(os.environ)
        environment.update({
            "SCOUT_LOCALIZATION_MODE": self.mode,
            "SCOUT_UI_ALLOW_VEHICLE_ACTIONS": self.allow,
            "AUTOWARE_UI_POINTCLOUD_MAP_FILE": self.map_file,
        })
        environment.pop("AUTOWARE_UI_RUNTIME_MAP_DIR", None)
        result = subprocess.run(
            ["bash", str(SCRIPT), "127.0.0.1", str(self.server.server_port), "false",
             self.map_dir if runtime_map_dir is None else runtime_map_dir],
            cwd=self.temp.name, env=environment, text=True, capture_output=True, timeout=8,
        )
        # Even rejected profiles must leave the existing process untouched.
        with urllib.request.urlopen(f"http://127.0.0.1:{self.server.server_port}/health", timeout=1) as reply:
            self.assertEqual(reply.status, 200)
        return result

    def assert_rejected(self):
        result = self.launch()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Existing UI cannot be reused", result.stderr)

    def test_same_universe_profile_is_reused(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Reusing the existing process", result.stdout)

    def test_same_fixposition_profile_is_reused(self):
        self.mode, self.allow, self.map_file = "fixposition", "1", "pointcloud_map.pcd"
        self.status = self.make_status()
        self.assertEqual(self.launch().returncode, 0)

    def test_relative_request_map_path_is_canonicalized(self):
        result = self.launch("./nested/../map with spaces")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_backend_mismatch_is_rejected(self):
        self.status["runtime_profile"]["localization_backend"] = "fixposition"
        self.status["localization_backend"] = "fixposition"
        self.assert_rejected()

    def test_runtime_map_and_pointcloud_file_mismatch_are_rejected(self):
        baseline = copy.deepcopy(self.status)
        for field, value in (("runtime_map_dir", self.map_dir + "-other"),
                             ("pointcloud_map_file", "old_map.pcd")):
            with self.subTest(field=field):
                self.status = copy.deepcopy(baseline)
                self.status["runtime_profile"][field] = value
                self.assert_rejected()

    def test_permissions_must_match_in_both_directions(self):
        self.status["runtime_profile"]["ui_actions_enabled"] = True
        self.status["ui_actions_enabled"] = True
        self.assert_rejected()
        self.allow = "1"
        self.status = self.make_status()
        self.status["runtime_profile"]["ui_actions_enabled"] = False
        self.status["ui_actions_enabled"] = False
        self.assert_rejected()

    def test_legacy_ui_without_profile_is_rejected(self):
        self.status.pop("runtime_profile")
        self.assert_rejected()

    def test_contradictory_top_level_fields_are_rejected(self):
        baseline = copy.deepcopy(self.status)
        for field, value in (("localization_backend", "fixposition"), ("ui_actions_enabled", True)):
            with self.subTest(field=field):
                self.status = copy.deepcopy(baseline)
                self.status[field] = value
                self.assert_rejected()

    def test_bad_json_http_error_and_incorrect_field_type_are_rejected(self):
        self.raw_json = b"{bad json"
        self.assert_rejected()
        self.raw_json = None
        self.status_code = 503
        self.assert_rejected()
        self.status_code = 200
        self.status["runtime_profile"]["ui_actions_enabled"] = "false"
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
