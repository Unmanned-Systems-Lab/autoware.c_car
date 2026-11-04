import math
import struct
import unittest

from scout_localization_support.can_feedback import (
    CAN_EFF_FLAG, CAN_RTR_FLAG, CAN_ERR_FLAG, CLASSIC_CAN_FRAME,
    CanFeedbackState, SCOUT_MOTION_FILTER_MASK, SCOUT_V2_MOTION_ID, is_scout_motion_feedback,
)


def frame(can_id=0x221, dlc=8):
    return CLASSIC_CAN_FRAME.pack(can_id, dlc, b'\x00' * 8)


class CanFeedbackTest(unittest.TestCase):
    def test_sdk_motion_id_and_kernel_data_filter_contract(self):
        self.assertEqual(SCOUT_V2_MOTION_ID, 0x221)
        self.assertEqual(SCOUT_MOTION_FILTER_MASK & CAN_ERR_FLAG, 0)
        self.assertTrue(SCOUT_MOTION_FILTER_MASK & CAN_EFF_FLAG)
        self.assertTrue(SCOUT_MOTION_FILTER_MASK & CAN_RTR_FLAG)

    def test_zero_motion_feedback_is_a_real_valid_feedback_frame(self):
        self.assertTrue(is_scout_motion_feedback(frame()))

    def test_unrelated_ids_and_frame_kinds_cannot_refresh_gate(self):
        for invalid in (
            frame(0x211), frame(0x241), frame(0x221 | CAN_EFF_FLAG),
            frame(0x221 | CAN_RTR_FLAG), frame(0x221 | CAN_ERR_FLAG),
            frame(dlc=0), frame(dlc=7), frame(dlc=9),
            frame()[:-1], frame() + b'\x00', b'\x00' * 72,
        ):
            with self.subTest(invalid=invalid): self.assertFalse(is_scout_motion_feedback(invalid))

    def test_first_feedback_required_and_timeout_boundary(self):
        state = CanFeedbackState(.5)
        self.assertFalse(state.is_fresh(10_000_000_000))
        self.assertTrue(state.observe(frame(), 10_000_000_000))
        self.assertTrue(state.is_fresh(10_500_000_000))
        self.assertFalse(state.is_fresh(10_500_000_001))
        self.assertFalse(state.is_fresh(9_999_999_999))

    def test_unrelated_traffic_and_cached_ros_odometry_do_not_extend_time(self):
        state = CanFeedbackState(.5)
        state.observe(frame(), 10_000_000_000)
        self.assertFalse(state.observe(frame(0x211), 10_400_000_000))
        for now in [10_100_000_000, 10_300_000_000, 10_600_000_000]:
            state.is_fresh(now)  # Frequent ROS callbacks do not renew CAN time.
        self.assertFalse(state.is_fresh(10_600_000_000))

    def test_interface_error_blocks_even_after_valid_frame(self):
        state = CanFeedbackState(.5)
        state.observe(frame(), 10_000_000_000)
        state.mark_interface_error()
        self.assertFalse(state.is_fresh(10_000_000_000))
        self.assertFalse(state.observe(frame(), 10_100_000_000))

    def test_invalid_timeout_is_rejected(self):
        for value in [0.0, -1.0, math.nan, math.inf]:
            with self.assertRaises(ValueError): CanFeedbackState(value)


if __name__ == '__main__':
    unittest.main()
