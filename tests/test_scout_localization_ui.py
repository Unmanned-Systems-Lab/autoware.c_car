"""Backend compatibility and action gates, without hardware or control calls."""
import importlib.util
import os
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from fixposition_driver_msgs.msg import FpaOdomstatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

UI_PATH = Path(__file__).resolve().parents[1] / 'src/launcher/autoware_launch/ui/generate_autoware_map.py'
SPEC = importlib.util.spec_from_file_location('scout_localization_ui_under_test', UI_PATH)
UI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UI
SPEC.loader.exec_module(UI)


class LocalizationUiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.mode = patch.object(UI, 'SCOUT_LOCALIZATION_MODE', 'universe')
        self.mode.start()
        self.addCleanup(self.mode.stop)
        self.env = patch.dict(os.environ, {'AUTOWARE_UI_RUNTIME_MAP_DIR': self.temp.name,
                                          'AUTOWARE_UI_POINTCLOUD_MAP_FILE': 'pointcloud_map.pcd'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.write_map()
        (self.directory / 'map_projector_info.yaml').write_text(
            'projector_type: LocalCartesian\nmap_origin:\n'
            '  latitude: 29.6\n  longitude: 106.28\n  altitude: 255.0\n')
        self.now = time.time_ns()
        self.received = time.monotonic()
        self.odom = Odometry()
        self.odom.header.frame_id = 'map'
        self.odom.child_frame_id = 'base_link'
        self.odom.header.stamp.sec, self.odom.header.stamp.nanosec = divmod(self.now, 10**9)
        self.odom.pose.pose.orientation.w = 1.0
        self.odom.pose.pose.position.x = 2.0
        self.ndt = PoseWithCovarianceStamped()
        self.ndt.header = self.odom.header
        self.ndt.pose = self.odom.pose
        self.client = UI.AutowareRuntimeClient()
        self.client._ensure_ready = MagicMock()
        self.client._spin_locked = MagicMock()
        self.client._node = MagicMock()
        self.client._node.get_clock().now().nanoseconds = self.now
        self.client._kinematic_state = self.odom
        self.client._kinematic_state_received_monotonic = self.received
        self.client._ndt_pose = self.ndt
        self.client._ndt_pose_received_monotonic = self.received
        self.client._localization_init_state = LocalizationInitializationState.INITIALIZED
        self.client._pointcloud_map_signature = ('map', 1500)
        self.client._fixposition_status = None

    def write_map(self, count=1500, truncate=False):
        header = ('VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                  f'COUNT 1 1 1 1\nWIDTH {count}\nHEIGHT 1\nPOINTS {count}\nDATA binary\n')
        body = struct.pack('<ffff', 1.0, 2.0, 3.0, 4.0) * (1 if truncate else count)
        (self.directory / 'pointcloud_map.pcd').write_bytes(header.encode() + body)

    def route_payload(self):
        return {'data': [{'latitude': 29.6, 'longitude': 106.28},
                         {'latitude': 29.6001, 'longitude': 106.2801}]}

    def test_healthy_universe_does_not_need_fixposition(self):
        status = self.client.localization_snapshot()
        self.assertTrue(status['ready'], status)
        vehicle = self.client.vehicle_state_snapshot()
        self.assertTrue(vehicle['received'])
        self.assertFalse(vehicle['fixposition_ready'])
        self.assertEqual(vehicle['localization_backend'], 'universe')
        self.assertEqual(vehicle['pose']['position']['x'], 2.0)
        self.assertEqual(vehicle['geo_source'], 'universe_map_projection')

    def test_fixposition_action_guard_is_preserved(self):
        context = UI.load_runtime_map_context({})
        with patch.object(UI, 'SCOUT_LOCALIZATION_MODE', 'fixposition'), \
                patch.object(UI, 'require_live_fp_enu0_alignment') as original:
            UI.require_active_localization(context, map_label='active map')
            original.assert_called_once_with(context.origin, map_label='active map')

    def test_fixposition_vehicle_still_requires_ready_fix(self):
        with patch.object(UI, 'SCOUT_LOCALIZATION_MODE', 'fixposition'):
            self.assertFalse(self.client.vehicle_state_snapshot()['received'])
            fix = FpaOdomstatus()
            fix.init_status = fix.consts.INIT_STATUS_GLOBAL_INIT
            fix.gnss1_status = fix.consts.GNSS_STATUS_NO_FIX + 1
            self.client._fixposition_status = fix
            self.client._fixposition_status_received_monotonic = self.received
            self.assertTrue(self.client.vehicle_state_snapshot()['received'])

    def test_initialized_good_fixed_map_allows_route(self):
        self.client.set_route = MagicMock(return_value={'status': 'ok'})
        with patch.object(UI, 'AUTOWARE_RUNTIME_CLIENT', self.client):
            result = UI.set_route_from_payload(self.route_payload())
        self.assertEqual(result['status'], 'ok')
        self.client.set_route.assert_called_once()

    def test_uninitialized_pose_cannot_route(self):
        self.client._localization_init_state = LocalizationInitializationState.INITIALIZING
        self.client.set_route = MagicMock()
        with patch.object(UI, 'AUTOWARE_RUNTIME_CLIENT', self.client):
            with self.assertRaisesRegex(UI.MapGenerationError, 'not initialized'):
                UI.set_route_from_payload(self.route_payload())
        self.client.set_route.assert_not_called()

    def test_odom_header_and_receipt_must_both_be_fresh(self):
        self.client._kinematic_state_received_monotonic = self.received - 3.0
        self.assertFalse(self.client.localization_snapshot()['ready'])
        self.client._kinematic_state_received_monotonic = self.received
        self.odom.header.stamp.sec -= 3
        self.assertFalse(self.client.localization_snapshot()['ready'])

    def test_fresh_ekf_with_stale_ndt_is_not_ready(self):
        self.client._ndt_pose_received_monotonic = self.received - 3.0
        status = self.client.localization_snapshot()
        self.assertTrue(status['kinematic_recent'])
        self.assertFalse(status['ready'])
        self.assertFalse(status['ndt_recent'])

    def test_wrong_frame_child_quaternion_and_nonfinite_are_rejected(self):
        for key, value in [('frame', 'odom'), ('child', 'rslidar'), ('quat', 0.0),
                           ('position', float('nan'))]:
            with self.subTest(key=key):
                self.odom.header.frame_id = 'map'
                self.odom.child_frame_id = 'base_link'
                self.odom.pose.pose.orientation.w = 1.0
                self.odom.pose.pose.position.x = 2.0
                if key == 'frame': self.odom.header.frame_id = value
                if key == 'child': self.odom.child_frame_id = value
                if key == 'quat': self.odom.pose.pose.orientation.w = value
                if key == 'position': self.odom.pose.pose.position.x = value
                self.assertFalse(self.client.localization_snapshot()['ready'])

    def test_loaded_map_must_match_fixed_file_geometry(self):
        for signature in [None, ('odom', 1500), ('map', 1499)]:
            self.client._pointcloud_map_signature = signature
            status = self.client.localization_snapshot()
            self.assertFalse(status['ready'])
            self.assertFalse(status['map_loaded'])

    def test_placeholder_truncated_and_missing_maps_are_rejected(self):
        self.write_map(count=105)
        self.assertFalse(self.client.localization_snapshot()['map_valid'])
        self.write_map(truncate=True)
        self.assertFalse(self.client.localization_snapshot()['map_valid'])
        (self.directory / 'pointcloud_map.pcd').unlink()
        self.assertFalse(self.client.localization_snapshot()['map_valid'])

    def test_map_mutation_is_disabled_before_any_reload_or_writes(self):
        with patch.object(UI.subprocess, 'run') as reload_map:
            for operation in [lambda: UI.generate_map_from_payload({}),
                              lambda: UI.activate_generated_map_from_payload({}),
                              lambda: UI.switch_autoware_map(self.directory)]:
                with self.assertRaisesRegex(UI.MapGenerationError, 'restart'):
                    operation()
            reload_map.assert_not_called()

    def test_runtime_profile_is_explicit_before_localization_is_ready(self):
        self.client.snapshot = MagicMock(return_value={"status": "ok"})
        self.client._localization_init_state = None
        with patch.object(UI, "AUTOWARE_RUNTIME_CLIENT", self.client):
            result = UI.get_autoware_status()
        profile = result["runtime_profile"]
        self.assertEqual(profile["localization_backend"], "universe")
        self.assertEqual(profile["runtime_map_dir"], str(self.directory.resolve()))
        self.assertEqual(profile["pointcloud_map_file"], "pointcloud_map.pcd")
        self.assertEqual(profile["ui_actions_enabled"], UI.SCOUT_UI_ALLOW_VEHICLE_ACTIONS)
        self.assertFalse(result["localization"]["ready"])

    @unittest.skipUnless(shutil.which("node"), "Node is only needed for browser action-gate tests")
    def test_browser_backend_readiness_and_map_action_gates(self):
        harness = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const html=fs.readFileSync(process.argv[1], 'utf8');
const code=Array.from(html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)).map(x=>x[1]).join('\n');
const elements={};
const document={getElementById(id){return elements[id] || (elements[id]={disabled:false,value:'map1',textContent:'',title:''});}};
const context={document,window:{location:{port:'8090',search:''},addEventListener(){}},URLSearchParams,console};
vm.createContext(context);new vm.Script(code).runInContext(context);
context.scheduleResponsiveUiUpdate=()=>{};
context.setRuntimeStatus=()=>{};context.setMapStatus=()=>{};
context.apiRequest=()=>{throw new Error('Unexpected backend action');};
function status(backend,ready,enabled){context.updateAutowareStatusPanel({localization_backend:backend,localization:{ready,error:'waiting'},ui_actions_enabled:enabled,map_switch_supported:backend==='fixposition',runtime_map:{status:'ok',output_dir:'/maps/one',origin_source:'survey'}});}
status('fixposition',false,true);
assert.equal(elements['send-route-button'].disabled,false);
assert.equal(elements['generate-autoware-map-button'].disabled,false);
status('universe',false,true);
assert.equal(elements['send-route-button'].disabled,true);
assert.equal(elements['enter-autonomous-button'].disabled,true);
assert.equal(elements['generate-autoware-map-button'].disabled,true);
assert.equal(elements['activate-map-button'].disabled,true);
assert.match(elements['autoware-localization'].textContent,/Universe/);
context.sendRouteToAutoware();context.enterAutonomousMode();context.generateAutowareMap();context.activateGeneratedMap();
status('universe',true,true);
assert.equal(elements['send-route-button'].disabled,false);
assert.equal(elements['enter-autonomous-button'].disabled,false);
assert.equal(elements['generate-autoware-map-button'].disabled,true);
assert.equal(elements['activate-map-button'].disabled,true);
status('universe',true,false);
assert.equal(elements['send-route-button'].disabled,true);
assert.equal(elements['enter-autonomous-button'].disabled,true);
"""
        result = subprocess.run(["node", "-e", harness, str(UI_PATH.with_name("index.html"))],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_gps_does_not_make_stale_universe_pose_available(self):
        gps = NavSatFix()
        gps.status.status = 0
        gps.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN
        gps.latitude, gps.longitude, gps.altitude = 29.6, 106.28, 255.0
        self.client._raw_gps = gps
        self.client._raw_gps_received_monotonic = self.received
        self.client._kinematic_state_received_monotonic = self.received - 3.0
        vehicle = self.client.vehicle_state_snapshot()
        self.assertFalse(vehicle['received'])
        self.assertFalse(vehicle['geo_available'])
        self.assertTrue(vehicle['raw_gps']['received'])


if __name__ == '__main__':
    unittest.main()
