import argparse
import csv
import sys
from pathlib import Path
from typing import Optional
from typing import TextIO

from ins_driver_c_car.protocol import POSITION_FRAME_IDS
from ins_driver_c_car.protocol import decode_s32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latitude/longitude entries from a raw CAN TXT log."
    )
    parser.add_argument("input", type=Path, help="Input TXT file in CANID#DATA format.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional CSV output path. Defaults to stdout.",
    )
    return parser.parse_args()


def open_output(path: Optional[Path]) -> TextIO:
    if path is None:
        return sys.stdout

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def parse_can_line(line: str, line_number: int) -> Optional[tuple[int, bytes, str]]:
    text = line.strip()
    if not text or text.startswith("#"):
        return None

    can_id_text, separator, payload_text = text.partition("#")
    if separator != "#":
        raise ValueError(f"Line {line_number}: invalid CAN line, expected CANID#DATA.")

    payload_text = payload_text.strip().upper()
    if len(payload_text) % 2 != 0:
        raise ValueError(f"Line {line_number}: payload hex length must be even.")

    try:
        can_id = int(can_id_text, 16)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid CAN ID '{can_id_text}'.") from exc

    try:
        payload = bytes.fromhex(payload_text)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid payload hex '{payload_text}'.") from exc

    return can_id, payload, payload_text


def decode_lat_lon(payload: bytes, line_number: int) -> tuple[float, float]:
    if len(payload) < 8:
        raise ValueError(
            f"Line {line_number}: position frame must contain 8 bytes, got {len(payload)}."
        )

    latitude_deg = decode_s32(payload, 0) * 1e-7
    longitude_deg = decode_s32(payload, 4) * 1e-7
    return latitude_deg, longitude_deg


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"Input TXT file not found: {args.input}", file=sys.stderr)
        return 1

    extracted_count = 0

    with args.input.open("r", encoding="utf-8") as input_handle:
        with open_output(args.output) as output_handle:
            writer = csv.writer(output_handle)
            writer.writerow(
                [
                    "entry_index",
                    "line_number",
                    "can_id",
                    "latitude_deg",
                    "longitude_deg",
                    "raw_payload",
                ]
            )

            for line_number, line in enumerate(input_handle, start=1):
                try:
                    parsed = parse_can_line(line, line_number)
                except ValueError as exc:
                    print(exc, file=sys.stderr)
                    return 1

                if parsed is None:
                    continue

                can_id, payload, payload_text = parsed
                if can_id not in POSITION_FRAME_IDS:
                    continue

                try:
                    latitude_deg, longitude_deg = decode_lat_lon(payload, line_number)
                except ValueError as exc:
                    print(exc, file=sys.stderr)
                    return 1

                extracted_count += 1
                writer.writerow(
                    [
                        extracted_count,
                        line_number,
                        f"0x{can_id:03X}",
                        f"{latitude_deg:.7f}",
                        f"{longitude_deg:.7f}",
                        payload_text,
                    ]
                )

    if extracted_count == 0:
        print("No latitude/longitude frames (0x20B/0x21B) were found.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
