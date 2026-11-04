"""Classic CAN motion-feedback recognition and an independent freshness gate."""

from dataclasses import dataclass
import math
import struct


CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x7FF
SCOUT_V2_MOTION_ID = 0x221
SCOUT_MOTION_FILTER_MASK = CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG
CLASSIC_CAN_FRAME = struct.Struct("=IB3x8s")


def is_scout_motion_feedback(frame):
    """Accept only standard, non-RTR/non-error Scout V2 0x221 DLC=8 frames."""
    if len(frame) != CLASSIC_CAN_FRAME.size:
        return False
    can_id, dlc, _ = CLASSIC_CAN_FRAME.unpack(frame)
    if can_id & (CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG):
        return False
    return can_id == SCOUT_V2_MOTION_ID and dlc == 8


@dataclass
class CanFeedbackState:
    timeout_sec: float = 0.5
    last_feedback_ns: object = None
    interface_error: bool = False

    def __post_init__(self):
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ValueError("can_feedback_timeout_sec must be finite and positive")

    def observe(self, frame, monotonic_ns):
        if self.interface_error or not is_scout_motion_feedback(frame):
            return False
        self.last_feedback_ns = monotonic_ns
        return True

    def is_fresh(self, monotonic_ns):
        if self.interface_error or self.last_feedback_ns is None:
            return False
        age = monotonic_ns - self.last_feedback_ns
        return 0 <= age <= int(self.timeout_sec * 1e9)

    def mark_interface_error(self):
        self.interface_error = True
        self.last_feedback_ns = None
